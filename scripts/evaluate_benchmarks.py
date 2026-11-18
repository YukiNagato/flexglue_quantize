"""Full frozen-checkpoint evaluation; no fitting or calibration on benchmarks.

Run as python -m scripts.evaluate_benchmarks. Geometry uses original pixel
coordinates. HardMatch's F metric follows its upstream MAGSAC/SED protocol;
all pairs undergo inference even when that metric has no applicable GT.
"""
import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('float_full', 'quant_full', 'quant_early')
CHECKPOINT = ROOT / 'results/eval100_ptq_best_posenc_float/model_fp8.pt'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def stats(values):
    a = np.asarray(values)
    return dict(mean=float(a.mean()), median=float(np.median(a)), p95=float(np.percentile(a, 95)))


def load_dataset(args):
    root = ROOT / 'data/benchmarks' / args.dataset
    manifest = json.loads((root / 'manifest.json').read_text())
    assert digest(root / 'pairs.jsonl') == manifest['pair_manifest_sha256']
    pairs = [json.loads(s) for s in (root / 'pairs.jsonl').read_text().splitlines()]
    expected = json.loads((ROOT / 'configs/benchmark_data.json').read_text())['datasets'][args.dataset]['expected_pairs']
    assert len(pairs) == expected
    return root, pairs, json.loads((root / 'images.json').read_text())


def feature_setup(args):
    import torch
    from scripts.inference import Config, provenance
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    cfg = Config()
    prov = provenance(cfg)
    # All variants share exact features, including the float descriptor.
    info = dict(config=asdict(cfg), weights=prov['weights_sha256'], torch=str(torch.__version__),
                sources=prov['source_sha256'])
    fingerprint = hashlib.sha256(json.dumps(info, sort_keys=True).encode()).hexdigest()
    cache = ROOT / 'results' / ('benchmark_features_' + fingerprint[:16])
    cache.mkdir(exist_ok=True)
    save(cache / 'manifest.json', info)
    return cfg, prov, cache


def feature_path(cache, meta):
    return cache / (meta['sha256'] + '.pt')


def extract(args):
    import torch
    from scripts.inference import FeatureExtractor
    cfg, prov, cache = feature_setup(args)
    root, pairs, images = load_dataset(args)
    names = sorted({p[k] for p in pairs for k in ('image0', 'image1')})
    output = args.output / args.dataset
    output.mkdir(exist_ok=True, parents=True)
    report = dict(state='extracting', total=len(names), cache=str(cache), provenance=prov)
    save(output / 'features.json', report)
    extractor = None
    t = time.perf_counter()
    for i, name in enumerate(names, 1):
        target = feature_path(cache, images[name])
        if not target.exists():
            assert digest(root / name) == images[name]['sha256']
            if extractor is None:
                extractor = FeatureExtractor(cfg)
            feature = extractor.extract(root / name)
            assert torch.isfinite(feature['features']).all() and torch.isfinite(feature['keypoints']).all()
            tmp = target.with_suffix('.tmp')
            torch.save(feature, tmp)
            tmp.replace(target)
        if i % 50 == 0 or i == len(names):
            report.update(completed=i, seconds=time.perf_counter() - t)
            save(output / 'features.json', report)
            print('FEATURES', args.dataset, i, '/', len(names), round(report['seconds'], 1), flush=True)
    report.update(state='complete', seconds=time.perf_counter() - t)
    save(output / 'features.json', report)


def infer(args):
    from scripts.streaming_benchmark import infer_stream
    return infer_stream(args)


def calibrated_pose_error(x, y, pair, seed):
    if len(x) < 5:
        return dict(success=False, reason='fewer_than_5_matches')
    K0, K1 = (np.asarray(pair[k], dtype=np.float64) for k in ('K0', 'K1'))
    a = cv2.undistortPoints(x.reshape(-1, 1, 2), K0, None).reshape(-1, 2)
    b = cv2.undistortPoints(y.reshape(-1, 1, 2), K1, None).reshape(-1, 2)
    focal = np.mean([K0[0, 0], K0[1, 1], K1[0, 0], K1[1, 1]])
    cv2.setRNGSeed(seed)
    E, mask = cv2.findEssentialMat(a, b, np.eye(3), method=cv2.RANSAC,
                                  prob=.999, threshold=1. / focal, maxIters=1000)
    if E is None or mask is None:
        return dict(success=False, reason='essential_estimation_failed')
    T = np.asarray(pair['T_0to1'], dtype=np.float64)
    if np.linalg.norm(T[:3, 3]) < 1e-8:
        return dict(success=False, reason='zero_reference_baseline')
    best = None
    for essential in E.reshape(-1, 3, 3):
        # Match standard image-matching evaluation: avoid OpenCV's default
        # 50-baseline distance cutoff discarding distant scene points.
        n, R, t, _, _ = cv2.recoverPose(essential, a, b, np.eye(3),
                                      distanceThresh=1e9, mask=mask.copy())
        if n == 0:
            continue
        rotation = float(np.degrees(np.arccos(np.clip((np.trace(R @ T[:3, :3].T) - 1) / 2, -1, 1))))
        cosine = abs(float(t[:, 0] @ T[:3, 3] / (np.linalg.norm(t) * np.linalg.norm(T[:3, 3]))))
        translation = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
        row = dict(success=True, rotation_deg=rotation, translation_deg=translation,
                   error_deg=max(rotation, translation), inliers=int(n))
        if best is None or n > best['inliers']:
            best = row
    return best or dict(success=False, reason='no_cheirality_inliers')


def score_pair(root, pair, pred, index):
    from scripts.matching_metrics import sampson_distance_px
    x, y = (pred[k].numpy().astype(np.float64) for k in ('points0', 'points1'))
    if 'K0' in pair:
        T = np.asarray(pair['T_0to1'])
        c0 = dict(K=pair['K0'], camera_to_world=np.eye(4).tolist())
        c1 = dict(K=pair['K1'], camera_to_world=np.linalg.inv(T).tolist())
        errors = sampson_distance_px(x, y, c0, c1)
        return dict(pair_id=pair['pair_id'], pose=calibrated_pose_error(x, y, pair, seed=20261118 + index),
                    correct={str(th): int((errors <= th).sum()) for th in (1, 3, 5)})
    eligible = pair['upstream_fundamental_metric_eligible']
    if not eligible:
        return dict(pair_id=pair['pair_id'], metric_eligible=False,
                    reason='fewer_than_8_gt' if pair['gt_correspondences'] < 8 else 'dynamic')
    gt = np.loadtxt(root / pair['correspondences']).reshape(-1, 4)
    cv2.setRNGSeed(20261118 + index)
    F = None
    if len(x) >= 8:
        try:
            F, _ = cv2.findFundamentalMat(x, y, cv2.USAC_MAGSAC, .25, .999, 100000)
        except cv2.error:
            pass
    success = bool(F is not None and F.shape == (3, 3) and np.isfinite(F).all())
    if not success:
        # Upstream's deterministic failure fallback, retained for comparability.
        F = np.array([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    a, b = np.c_[gt[:, :2], np.ones(len(gt))], np.c_[gt[:, 2:], np.ones(len(gt))]
    l1, l0 = a @ F.T, b @ F
    with np.errstate(divide='ignore', invalid='ignore'):
        errors = np.sqrt((b * l1).sum(1)**2 *
                         (1 / (l1[:, :2]**2).sum(1) + 1 / (l0[:, :2]**2).sum(1)) + 1e-8)
    accuracy = [(errors <= th).mean().item() for th in range(20)]
    return dict(pair_id=pair['pair_id'], metric_eligible=True, fundamental_success=success,
                accuracy_by_px=accuracy, asset_source=pair['asset_source'])


def score(args):
    import torch
    from scripts.matching_metrics import pose_auc
    cv2.setNumThreads(1)
    root, pairs, images = load_dataset(args)
    folder = args.output / args.dataset / args.variant
    report = json.loads((folder / 'inference.json').read_text())
    assert report['state'] == 'complete'
    predictions = torch.load(folder / 'predictions.pt', weights_only=True)
    assert [p['pair_id'] for p in predictions] == [p['pair_id'] for p in pairs]
    metrics = None
    if args.reference_results:
        reference = args.reference_results / args.dataset / args.variant
        previous = json.loads((reference / 'report.json').read_text())
        for key in ('pair_manifest_sha256', 'checkpoint_sha256', 'config', 'feature_cache'):
            assert report[key] == previous[key], key
        old = torch.load(reference / 'predictions.pt', weights_only=True)
        assert [r['pair_id'] for r in old] == [r['pair_id'] for r in predictions]
        changed = [a['pair_id'] for a, b in zip(predictions, old)
                   if not all(torch.equal(a[k], b[k]) for k in ('indices', 'points0', 'points1'))]
        report['reference_comparison'] = dict(reference=str(reference), changed_prediction_pairs=changed,
            exact_match_coordinates=not changed, reused_geometry=not changed,
            reference_report_sha256=digest(reference / 'report.json'))
        if not changed:
            metrics = previous['geometry']
    if metrics is None:
        with ThreadPoolExecutor(4) as pool:
            metrics = list(pool.map(lambda a: score_pair(root, *a), zip(pairs, predictions, range(len(pairs)))))
    rows = report['results']
    summary = dict(pairs=len(rows), wall_ms=stats([v for r in rows for v in r['wall_ms']]),
        gpu_ms=stats([v for r in rows for v in r['gpu_ms']]),
        rejected_pairs=sum(r['rejected'] for r in rows), total_matches=sum(r['matches'] for r in rows),
        stop_layer_counts=dict(Counter(str(r['stop_layer']) for r in rows)))
    summary['primary_timing_metric'] = 'gpu_ms'
    if 'stream' in report:
        summary['cached_stream'] = report['stream']
    if args.dataset != 'hardmatch':
        errors = [r['pose'].get('error_deg', float('inf')) for r in metrics]
        summary.update(pose_auc={str(th): pose_auc(errors, th) for th in (5, 10, 20)},
            pose_successes=sum(r['pose']['success'] for r in metrics),
            sampson_precision={str(th): sum(r['correct'][str(th)] for r in metrics) / max(1, summary['total_matches']) for th in (1, 3, 5)})
    else:
        valid = [r for r in metrics if r['metric_eligible']]
        accuracy = np.mean([r['accuracy_by_px'] for r in valid], axis=0)
        strict = np.mean([r['accuracy_by_px'] if r['fundamental_success'] else np.zeros(20) for r in valid], axis=0)
        summary.update(metric_pairs=len(valid), fundamental_successes=sum(r['fundamental_success'] for r in valid),
                       mAA_5px=float(accuracy[:6].mean()), mAA_10px=float(accuracy[:11].mean()),
                       strict_failure_zero_mAA_10px=float(strict[:11].mean()),
                       metric_exclusions=dict(Counter(r['reason'] for r in metrics if not r['metric_eligible'])))
        official_assets = [r for r in valid if r['asset_source'] == 'official_archive']
        summary['official_assets_only_mAA_10px'] = float(np.mean([r['accuracy_by_px'][:11] for r in official_assets]))
    report.update(state='complete', summary=summary, geometry=metrics, scorer_sha256=digest(__file__),
        metric_protocol=dict(pose='OpenCV RANSAC essential matrix: 1 original-image pixel, confidence .999, maxIters 1000; recoverPose distanceThresh=1e9, require positive cheirality inliers; max(rotation, sign-invariant translation) error. Same fixed protocol for all variants; not an exact paper reproduction.',
            hardmatch='Upstream MAGSAC F: .25px, confidence .999, maxIters 100000; symmetric epipolar distance on annotated GT, mean accuracy over integer thresholds 0..10; original split labels retained; all 1000 pairs inferred.',
            coordinate_offset=0, opencv=cv2.__version__))
    save(folder / 'report.json', report)
    print('SCORE', args.dataset, args.variant, json.dumps(summary), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'results/streaming_benchmarks_20261118')
    p.add_argument('--stage', choices=('run', 'extract', 'infer', 'score'), default='run')
    p.add_argument('--dataset', choices=('megadepth1500', 'scannet1500', 'hardmatch'))
    p.add_argument('--variant', choices=VARIANTS)
    p.add_argument('--repeats', type=int, default=1, help='Must be 1: one call per pair after startup warmup')
    p.add_argument('--warmup-pairs', type=int, default=32)
    p.add_argument('--warmup-rounds', type=int, default=2)
    p.add_argument('--reference-results', type=Path, help='Reuse prior GT scoring only after exact predicted-coordinate parity')
    args = p.parse_args()
    if args.repeats != 1 or args.warmup_pairs < 1 or args.warmup_rounds < 1:
        p.error('Use exactly one timed traversal and positive startup warmup counts')
    args.output.mkdir(exist_ok=True, parents=True)
    if args.stage != 'run':
        assert (args.dataset or args.stage == 'infer') and (args.stage == 'extract' or args.variant)
        return globals()[args.stage](args)
    packages = {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions()}
    run = dict(state='running', started=time.time(), packages_before=packages,
               evaluator_sha256=digest(__file__), checkpoint_sha256=digest(CHECKPOINT), cases=[])
    save(args.output / 'progress.json', run)
    (args.output / 'evaluate_benchmarks.py').write_bytes(Path(__file__).read_bytes())
    (args.output / 'streaming_benchmark.py').write_bytes((ROOT / 'scripts/streaming_benchmark.py').read_bytes())
    environment = dict(os.environ)
    environment['TORCHINDUCTOR_COMPILE_THREADS'] = '4'
    datasets = [args.dataset] if args.dataset else ['megadepth1500', 'scannet1500', 'hardmatch']
    jobs = [(d, None, 'extract') for d in datasets]
    for variant in VARIANTS:
        jobs.append((args.dataset, variant, 'infer'))
        jobs.extend((d, variant, 'score') for d in datasets)
    for dataset, variant, stage in jobs:
        folder = args.output / dataset if dataset else args.output
        target = (args.output / f'{variant}_stream.json' if stage == 'infer' else
                  folder / 'features.json' if stage == 'extract' else folder / variant / 'report.json')
        if target.exists() and json.loads(target.read_text())['state'] == 'complete':
            continue
        folder.mkdir(exist_ok=True, parents=True)
        command = [sys.executable, '-m', 'scripts.evaluate_benchmarks', '--stage', stage,
                   '--output', str(args.output), '--warmup-pairs', str(args.warmup_pairs),
                   '--warmup-rounds', str(args.warmup_rounds)]
        if dataset:
            command += ['--dataset', dataset]
        if variant:
            command += ['--variant', variant]
        if args.reference_results:
            command += ['--reference-results', str(args.reference_results)]
        case = dict(dataset=dataset or 'all', variant=variant, stage=stage, state='running', started=time.time())
        run['cases'].append(case)
        save(args.output / 'progress.json', run)
        print('START', dataset or 'all', variant, stage, flush=True)
        with (folder / f'{variant or "features"}_{stage}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        case.update(state='complete' if result.returncode == 0 else 'failed', seconds=time.time() - case['started'])
        if result.returncode:
            run['state'] = 'failed'
        save(args.output / 'progress.json', run)
        if result.returncode:
            raise RuntimeError(f'Failed {case}; inspect log in {folder}')
    summaries = {d: {v: json.loads((args.output / d / v / 'report.json').read_text())['summary'] for v in VARIANTS} for d in datasets}
    save(args.output / 'summary.json', summaries)
    from scripts.streaming_benchmark import summarize_matcher_timing
    summarize_matcher_timing(args.output, datasets)
    after = {d.metadata['Name'].lower(): d.version for d in importlib.metadata.distributions()}
    assert after == packages
    assert digest(CHECKPOINT) == run['checkpoint_sha256']
    run.update(state='complete', seconds=time.time() - run['started'], packages_after=after)
    save(args.output / 'progress.json', run)
    print('ALL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
