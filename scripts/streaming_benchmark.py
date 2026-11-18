"""One startup warmup followed by one sequential pass over cached features."""
from collections import OrderedDict
from dataclasses import asdict
import time

import numpy as np
import torch

from scripts.evaluate_benchmarks import CHECKPOINT, digest, feature_setup, load_dataset, save, stats
from scripts.inference import Matcher

PROTOCOL = 'startup-warmup-cached-stream-v1'
DATASETS = ('megadepth1500', 'scannet1500', 'hardmatch')

MATCHER_SCOPE = ('CUDA-event elapsed time around model.match after input H2D and before output D2H. '
                 'Excludes disk IO, feature extraction, feature transfers, CPU output conversion, '
                 'startup warmup and GT scoring. Includes device-timeline gaps inside matching, '
                 'including early-exit host decisions; not a sum of kernel-only durations.')


def summarize_matcher_timing(output, datasets=DATASETS):
    """Derive the primary no-IO comparison without changing measured records."""
    import json
    from pathlib import Path
    output = Path(output)
    variants = ('float_full', 'quant_full', 'quant_early')
    combined = {v: [] for v in variants}
    sources, results = {}, {}

    def summarize(values):
        timings = {v: stats(a) for v, a in values.items()}
        f, q, e = (timings[v]['mean'] for v in variants)
        return dict(pairs=len(values['float_full']), gpu_ms=timings,
                    full_speedup=f / q, full_latency_reduction_percent=100 * (1 - q / f),
                    early_latency_change_percent=100 * (e / q - 1))

    for dataset in datasets:
        values, reference_ids = {}, None
        for variant in variants:
            path = output / dataset / variant / 'inference.json'
            report = json.loads(path.read_text())
            assert report['state'] == 'complete' and report['timing_protocol'] == PROTOCOL
            ids = [r['pair_id'] for r in report['results']]
            assert len(ids) == len(set(ids)) == report['pairs']
            if reference_ids is None:
                reference_ids = ids
            assert ids == reference_ids
            assert all(len(r['gpu_ms']) == 1 for r in report['results'])
            values[variant] = [r['gpu_ms'][0] for r in report['results']]
            assert all(np.isfinite(v) and v > 0 for v in values[variant])
            combined[variant].extend(values[variant])
            sources[str(path.relative_to(output))] = digest(path)
        results[dataset] = summarize(values)
    result = dict(state='complete', primary_metric='gpu_ms', unit='ms per pair',
                  scope=MATCHER_SCOPE, datasets=results, overall=summarize(combined),
                  source_reports_sha256=sources,
                  note='Derived from recorded CUDA events; no new inference or timing run.')
    save(output / 'matcher_comparison.json', result)
    return result


def compiler_counts():
    from torch._dynamo.utils import counters
    return {category: dict(values) for category, values in counters.items()}


@torch.inference_mode()
def infer_stream(args):
    from flexglue_quantize import FlexGlue
    cfg, provenance, cache = feature_setup(args)
    selected = [args.dataset] if args.dataset else list(DATASETS)
    cfg.early_stop_decay = .5 if args.variant == 'quant_early' else None
    torch._inductor.config.compile_threads = 4
    if args.variant == 'float_full':
        matcher = Matcher(cfg)
        matcher.model.requires_grad_(False)
    else:
        matcher = Matcher.__new__(Matcher)
        matcher.config, matcher._cache = cfg, OrderedDict()
        matcher.model = FlexGlue(weights=CHECKPOINT, device='cuda',
            cute_preprocess='qkv', head_backend='cute-precise',
            early_exit_backend='cute' if cfg.early_stop_decay is not None else 'none',
            early_stop_decay=cfg.early_stop_decay)
    matcher.model.compile()
    loaded, all_pairs, features = {}, [], {}
    original_dataset = args.dataset
    for name in selected:
        args.dataset = name
        root, pairs, images = load_dataset(args)
        loaded[name] = (root, pairs)
        for image_name in sorted({p[k] for p in pairs for k in ('image0', 'image1')}):
            features[name, image_name] = torch.load(cache / (images[image_name]['sha256'] + '.pt'), weights_only=True)
        all_pairs.extend((name, pair) for pair in pairs)
    args.dataset = original_dataset

    # Fixed uniform positions, independent of GT, model scores, or exit layers.
    positions = np.linspace(0, len(all_pairs) - 1, min(args.warmup_pairs, len(all_pairs)), dtype=int).tolist()
    warmup = dict(selection='Uniform positions in concatenated manifests; independent of GT and predictions',
                  rounds=args.warmup_rounds, pairs=[dict(dataset=all_pairs[i][0], pair_id=all_pairs[i][1]['pair_id']) for i in positions])
    print('WARMUP', args.variant, len(positions), 'pairs x', args.warmup_rounds, 'rounds', flush=True)
    torch.cuda.synchronize()
    before = time.perf_counter()
    for _ in range(args.warmup_rounds):
        for i in positions:
            name, pair = all_pairs[i]
            inputs = [matcher.input(features[name, pair[k]]) for k in ('image0', 'image1')]
            out = matcher.model.match(*inputs, sparse=True)
            del out, inputs
    torch.cuda.synchronize()
    warmup['seconds'] = time.perf_counter() - before
    warmup['calls'] = len(positions) * args.warmup_rounds
    # Identical empty LRU at stream start; warmup must not privilege its pairs.
    matcher.clear_cache()
    initial_counters = compiler_counts()
    stream = dict(state='running', protocol=PROTOCOL, variant=args.variant,
        datasets=selected, pairs=len(all_pairs), warmup=warmup,
        feature_cache=str(cache), gpu_cache_images=cfg.gpu_cache_images,
        primary_metric='matcher_gpu_ms', primary_scope=MATCHER_SCOPE,
        secondary_wall_scope='Cached-feature stream including GPU-LRU transfers and CPU match outputs; retained only as diagnostic wall time.',
        compile_counters_before=initial_counters)
    stream_path = args.output / f'{args.variant}_stream.json'
    save(stream_path, stream)
    reports, predictions = {}, {}
    for name, (root, pairs) in loaded.items():
        report = dict(state='inference', dataset=name, variant=args.variant,
            pairs=len(pairs), config=asdict(cfg), provenance=provenance, repeats=1,
            timing_protocol=PROTOCOL, timing_scope=stream['primary_scope'], primary_timing_metric='gpu_ms',
            checkpoint_sha256=digest(CHECKPOINT) if args.variant != 'float_full' else digest(cfg.weights),
            pair_manifest_sha256=digest(root / 'pairs.jsonl'), feature_cache=str(cache),
            backend=dict(attention='SDPA' if args.variant == 'float_full' else matcher.model.attention_backend,
                         linears='float' if args.variant == 'float_full' else 'frozen static FP8',
                         compile=True, cute=args.variant != 'float_full'), results=[])
        reports[name], predictions[name] = report, []
        folder = args.output / name / args.variant
        folder.mkdir(parents=True, exist_ok=True)
        save(folder / 'inference.json', report)

    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    overall = time.perf_counter()
    for name, (_, pairs) in loaded.items():
        rows, outputs = reports[name]['results'], predictions[name]
        dataset_start = time.perf_counter()
        for i, pair in enumerate(pairs, 1):
            pair_start = time.perf_counter()
            inputs = [matcher.input(features[name, pair[k]]) for k in ('image0', 'image1')]
            start.record()
            out = matcher.model.match(*inputs, sparse=True)
            end.record()
            # These D2H copies provide the synchronization required by the real
            # CPU-output API. Do not insert an additional per-pair synchronize.
            mapping = out['matches0'][0].cpu()
            scores = out['matching_scores0'][0].cpu()
            covis = float(out['covisibility_score'][0])
            source = torch.where(mapping >= 0)[0]
            indices = torch.stack((source, mapping[source]), -1)
            scores = scores[source]
            rejected = covis < cfg.min_covisibility
            if rejected:
                indices, scores = indices[:0], scores[:0]
            points = [features[name, pair[k]]['keypoints'][indices[:, j]].clone()
                      for j, k in enumerate(('image0', 'image1'))]
            outputs.append(dict(pair_id=pair['pair_id'], indices=indices, scores=scores,
                                points0=points[0], points1=points[1]))
            layer = int(out['stop_layer'])
            elapsed_ms = (time.perf_counter() - pair_start) * 1000
            device_ms = start.elapsed_time(end)
            rows.append(dict(pair_id=pair['pair_id'], matches=len(indices), covisibility=covis,
                             rejected=rejected, stop_layer=layer, wall_ms=[elapsed_ms], gpu_ms=[device_ms]))
            del out, inputs
            if i % 100 == 0 or i == len(pairs):
                print('STREAM', args.variant, name, i, '/', len(pairs), flush=True)
        seconds = time.perf_counter() - dataset_start
        reports[name]['stream'] = dict(seconds=seconds, pairs=len(pairs),
            mean_ms_per_pair=seconds * 1000 / len(pairs), pairs_per_second=len(pairs) / seconds,
            matcher_calls=len(pairs), warmup_calls_during_stream=0)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - overall
    after_counters = compiler_counts()
    stream.update(state='complete', seconds=seconds, mean_ms_per_pair=seconds * 1000 / len(all_pairs),
        pairs_per_second=len(all_pairs) / seconds, matcher_calls=len(all_pairs),
        warmup_calls_during_stream=0, compile_counters_after=after_counters,
        new_unique_graphs_during_stream=after_counters.get('stats', {}).get('unique_graphs', 0) - initial_counters.get('stats', {}).get('unique_graphs', 0),
        peak_allocated_bytes=torch.cuda.max_memory_allocated())
    stream['matcher_gpu_ms'] = stats([v for r in reports.values() for row in r['results'] for v in row['gpu_ms']])
    # Large report/prediction writes happen only after the complete stream.
    for name, report in reports.items():
        assert len(report['results']) == report['pairs']
        report.update(state='complete', completed=report['pairs'], startup_warmup=warmup)
        folder = args.output / name / args.variant
        torch.save(predictions[name], folder / 'predictions.pt')
        save(folder / 'inference.json', report)
    save(stream_path, stream)
    print('STREAM COMPLETE', args.variant, seconds, 'seconds;', stream['new_unique_graphs_during_stream'], 'new graphs', flush=True)
