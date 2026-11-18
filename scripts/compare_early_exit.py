"""Measure original-float early decisions on the exact full-benchmark features.

This is a decision/accuracy comparison, not a new latency benchmark. Existing
quantized early predictions are reused; all original-float pairs run once.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import time

import torch

from scripts.evaluate_benchmarks import ROOT, digest, feature_setup, load_dataset, save
from scripts.inference import Matcher


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=ROOT / 'results/full_benchmarks_20261118')
    p.add_argument('--output', type=Path, default=ROOT / 'results/early_float_comparison_20261118')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    packages = {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions()}
    cfg, provenance, cache = feature_setup(args)
    cfg.early_stop_decay = .5
    torch._inductor.config.compile_threads = 4
    matcher = Matcher(cfg)
    matcher.model.requires_grad_(False)
    matcher.model.compile()
    overall = time.perf_counter()
    report = dict(state='running', reference=str(args.reference), config=asdict(cfg),
                  provenance=provenance, packages_before=packages, datasets={},
                  scope='One original-float compiled early-exit inference per pair; no latency claim. Same cached features, thresholds, weights and empty-match early rejection semantics as the prior benchmark.')
    (args.output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(args.output / 'report.json', report)
    for name in ('megadepth1500', 'scannet1500', 'hardmatch'):
        args.dataset = name
        root, pairs, images = load_dataset(args)
        references = {v: json.loads((args.reference / name / v / 'report.json').read_text())
                      for v in ('float_full', 'quant_full', 'quant_early')}
        full, quant_full, quant = (references[v] for v in ('float_full', 'quant_full', 'quant_early'))
        assert full['feature_cache'] == quant['feature_cache'] == str(cache)
        assert full['checkpoint_sha256'] == digest(cfg.weights)
        assert full['provenance']['source_sha256'] == provenance['source_sha256']
        for field, value in asdict(cfg).items():
            assert quant['config'][field] == value, field
        assert all(r['pair_manifest_sha256'] == digest(root / 'pairs.jsonl') for r in references.values())
        predictions = torch.load(args.reference / name / 'float_full/predictions.pt', weights_only=True)
        names = {pair[k] for pair in pairs for k in ('image0', 'image1')}
        features = {n: torch.load(cache / (images[n]['sha256'] + '.pt'), weights_only=True) for n in names}
        rows = []
        dataset = dict(state='running', pairs=len(pairs), rows=rows)
        report['datasets'][name] = dataset
        for i, (pair, f, q, qf, fp) in enumerate(zip(pairs, full['results'], quant['results'], quant_full['results'], predictions), 1):
            assert pair['pair_id'] == f['pair_id'] == q['pair_id'] == qf['pair_id'] == fp['pair_id']
            inputs = [matcher.input(features[pair[k]]) for k in ('image0', 'image1')]
            out = matcher.model.match(*inputs, sparse=True)
            layer = int(out['stop_layer'])
            score = float(out['covisibility_score'][0])
            mapping = out['matches0'][0].cpu()
            source = torch.where(mapping >= 0)[0]
            indices = torch.stack((source, mapping[source]), -1)
            rejected = score < cfg.min_covisibility
            if rejected:
                indices = indices[:0]
            if layer < 9:
                assert len(indices) == 0
            else:
                # Extra early checks must not alter the full-depth float result.
                assert torch.equal(indices, fp['indices']), (name, pair['pair_id'])
                assert score == f['covisibility'], (name, pair['pair_id'], score, f['covisibility'])
            rows.append(dict(pair_id=pair['pair_id'], float_layer=layer, quant_layer=q['stop_layer'],
                float_early=layer < 9, quant_early=q['stop_layer'] < 9,
                float_score_at_exit=score, quant_score_at_exit=q['covisibility'],
                float_matches=len(indices), quant_matches=q['matches'],
                float_rejected=rejected, quant_rejected=q['rejected'],
                float_full_rejected=f['rejected'], quant_full_rejected=qf['rejected']))
            del out
            if i % 100 == 0 or i == len(pairs):
                dataset['completed'] = i
                save(args.output / 'report.json', report)
                print(name, i, '/', len(pairs), 'float exits', sum(r['float_early'] for r in rows), flush=True)
        counts = Counter((r['float_early'], r['quant_early']) for r in rows)
        summary = dict(pairs=len(rows), float_exits=sum(r['float_early'] for r in rows),
            quant_exits=sum(r['quant_early'] for r in rows), both_exit=counts[True, True],
            float_only_exit=counts[True, False], quant_only_exit=counts[False, True], neither_exit=counts[False, False],
            trigger_agreement=sum(r['float_early'] == r['quant_early'] for r in rows) / len(rows),
            exact_layer_agreement=sum(r['float_layer'] == r['quant_layer'] for r in rows) / len(rows),
            both_exit_same_layer=sum(r['float_early'] and r['quant_early'] and r['float_layer'] == r['quant_layer'] for r in rows),
            quant_earlier_layer=sum(r['quant_layer'] < r['float_layer'] for r in rows),
            quant_later_layer=sum(r['quant_layer'] > r['float_layer'] for r in rows),
            float_layer_counts=dict(sorted(Counter(str(r['float_layer']) for r in rows).items())),
            quant_layer_counts=dict(sorted(Counter(str(r['quant_layer']) for r in rows).items())),
            float_mean_blocks=sum(r['float_layer'] for r in rows) / len(rows),
            quant_mean_blocks=sum(r['quant_layer'] for r in rows) / len(rows),
            float_early_but_float_full_accepted=sum(r['float_early'] and not r['float_full_rejected'] for r in rows),
            quant_early_but_quant_full_accepted=sum(r['quant_early'] and not r['quant_full_rejected'] for r in rows))
        dataset.update(state='complete', summary=summary,
            changed_trigger_pairs=[r for r in rows if r['float_early'] != r['quant_early']],
            changed_layer_pairs=[r for r in rows if r['float_layer'] != r['quant_layer']])
        save(args.output / 'report.json', report)
        print('SUMMARY', name, json.dumps(summary), flush=True)
        matcher.clear_cache()
        del features, predictions
    after = {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions()}
    assert after == packages
    for file, expected in provenance['weights_sha256'].items():
        assert digest(file) == expected
    for file, expected in provenance['source_sha256'].items():
        assert digest(ROOT / file) == expected
    report.update(state='complete', seconds=time.perf_counter() - overall, packages_after=after,
                  non_exited_float_predictions_exactly_match_full_baseline=True)
    save(args.output / 'report.json', report)
    save(args.output / 'summary.json', {n: d['summary'] for n, d in report['datasets'].items()})
    print('ALL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
