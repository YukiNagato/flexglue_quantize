"""Standalone RaCo + FlexGlue sparse inference using editable local sources."""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from collections import OrderedDict
from dataclasses import asdict, dataclass
import hashlib
import inspect
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

from flexglue_quantize import DINOv3Descriptor
from flexglue_quantize.core import FlexGlue
from scripts.raco import RaCo

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    max_keypoints: int = 4096
    detection_resize: int = 1024
    feature_resize: int = 512
    n_layers: int = 9
    min_covisibility: float = 0.9
    early_stop_decay: float | None = None
    filter_threshold: float = 0.1
    max_distance_px: float = 3.0
    gpu_cache_images: int = 32
    mp: str | None = 'float16'
    device: str = 'cuda'
    weights: str = str(ROOT / 'weights/flexglue.pth')
    detector_weights: str = str(ROOT / 'weights/raco.pth')

    def __post_init__(self):
        if self.max_keypoints < 1 or self.detection_resize < 32:
            raise ValueError('Require max_keypoints > 0 and detection_resize >= 32')
        if self.feature_resize < 32 or self.feature_resize % 32:
            raise ValueError('feature_resize must be a positive multiple of 32')
        if not 0 <= self.n_layers <= 9 or not 0 <= self.min_covisibility <= 1:
            raise ValueError('Require 0..9 layers and covisibility in [0, 1]')
        if self.early_stop_decay is not None and (not 0 <= self.early_stop_decay < 1 or self.min_covisibility == 0):
            raise ValueError('Early stopping requires decay in [0, 1) and positive covisibility')
        if self.gpu_cache_images < 2:
            raise ValueError('gpu_cache_images must be at least 2')
        for path in (self.weights, self.detector_weights):
            if not Path(path).is_file():
                raise FileNotFoundError(path)


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def timed(fn, device):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return result, time.perf_counter() - start


class FeatureExtractor:
    def __init__(self, config):
        self.config = config
        self.detector = RaCo(max_num_keypoints=config.max_keypoints,
                             weights=config.detector_weights).eval().to(config.device)
        self.descriptor = DINOv3Descriptor(resize=(config.feature_resize,) * 2,
            describe=False, weights=config.weights).eval().to(config.device)

    @torch.inference_mode()
    def extract(self, path):
        with Image.open(path) as image:
            pixels = np.array(image.convert('RGB'))
        image = torch.from_numpy(pixels).to(self.config.device).float().permute(2, 0, 1) / 255
        detections = self.detector.extract(image, resize=self.config.detection_resize)
        dense = self.descriptor({'image': image[None]})
        return dict(keypoints=detections['keypoints'][0].cpu(),
                    features=dense['features'][0].cpu(), image_size=tuple(image.shape[-2:]))


class Matcher:
    def __init__(self, config):
        self.config = config
        self.model = FlexGlue(describe=False, refine=False, weights=config.weights,
            n_layers=config.n_layers, min_covisibility=config.min_covisibility,
            early_stop_decay=config.early_stop_decay, filter_threshold=config.filter_threshold,
            max_distance_px=config.max_distance_px, mp=config.mp).eval().to(config.device)
        self._cache = OrderedDict()

    def input(self, feature):
        key = id(feature)
        entry = self._cache.pop(key, None)
        if entry is None:
            height, width = feature['image_size']
            entry = (feature, dict(features=feature['features'][None].to(self.config.device),
                keypoints=feature['keypoints'][None].to(self.config.device),
                image_size=torch.tensor([[width, height]], dtype=torch.float32, device=self.config.device)))
        self._cache[key] = entry
        while len(self._cache) > self.config.gpu_cache_images:
            self._cache.popitem(last=False)
        return entry[1]

    def clear_cache(self):
        self._cache.clear()

    @torch.inference_mode()
    def match(self, feature0, feature1):
        if min(len(feature0['keypoints']), len(feature1['keypoints'])) == 0:
            return dict(indices=torch.empty((0, 2), dtype=torch.long), scores=torch.empty(0),
                        metadata=dict(empty_input=True))
        pred = self.model.match(self.input(feature0), self.input(feature1), sparse=True)
        score = float(pred['covisibility_score'][0])
        rejected = score < self.config.min_covisibility
        mapping = pred['matches0'][0]
        source = torch.where(mapping >= 0)[0]
        indices = torch.stack((source, mapping[source]), -1)
        scores = pred['matching_scores0'][0, source]
        if rejected:
            indices, scores = indices[:0], scores[:0]
        return dict(indices=indices.cpu(), scores=scores.cpu(), metadata=dict(
            covisibility_score=score, rejected=rejected, stop_layer=int(pred['stop_layer'])))


def add_model_arguments(parser):
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--weights', type=str, default=str(ROOT / 'weights/flexglue.pth'))
    parser.add_argument('--detector-weights', type=str, default=str(ROOT / 'weights/raco.pth'))
    parser.add_argument('--max-keypoints', type=int, default=4096)
    parser.add_argument('--detection-resize', type=int, default=1024)
    parser.add_argument('--feature-resize', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=9)
    parser.add_argument('--min-covisibility', type=float, default=0.9)
    parser.add_argument('--early-stop-decay', type=float, default=None)
    parser.add_argument('--filter-threshold', type=float, default=0.1)
    parser.add_argument('--max-distance-px', type=float, default=3.0)
    parser.add_argument('--gpu-cache-images', type=int, default=32)
    parser.add_argument('--mp', choices=['float16', 'bfloat16', 'float32'], default='float16',
                        help='Transformer precision; float32 disables autocast')
    parser.add_argument('--threads', type=int, default=4)


def config_from_args(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    values = {key: getattr(args, key) for key in Config.__dataclass_fields__}
    values['mp'] = None if args.mp == 'float32' else args.mp
    return Config(**values)


def provenance(config):
    paths = {name: str(Path(inspect.getfile(cls)).resolve()) for name, cls in
             [('FlexGlue', FlexGlue), ('DINOv3Descriptor', DINOv3Descriptor), ('RaCo', RaCo)]}
    for path in paths.values():
        if not any(Path(path).is_relative_to(ROOT / folder) for folder in ('flexglue_quantize', 'scripts/raco')):
            raise RuntimeError(f'Model loaded outside project: {path}')
    if any(name == 'torch_sfm' or name.startswith('torch_sfm.') or
           name in ('flexglue', 'raco') or name.startswith(('flexglue.', 'raco.')) for name in sys.modules):
        raise RuntimeError('Unexpected external model or torch_sfm import')
    sources = [Path(__file__), *sorted((ROOT / 'flexglue_quantize').rglob('*.py')), *sorted((ROOT / 'scripts/raco').rglob('*.py'))]
    return dict(config=asdict(config), model_paths=paths, external_model_imports=False,
        torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(config.device) if torch.device(config.device).type == 'cuda' else None,
        threads=torch.get_num_threads(), matmul_precision=torch.get_float32_matmul_precision(),
        tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        weights_sha256={str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for p in (config.weights, config.detector_weights)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--images', type=Path, help='Image list; relative paths resolved against list location')
    inputs.add_argument('--image-dir', type=Path, help='Recursively match all images in this directory')
    inputs.add_argument('--pair', type=Path, nargs=2, metavar=('IMAGE0', 'IMAGE1'))
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--repeats', type=int, default=1)
    add_model_arguments(parser)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    cfg = config_from_args(args)
    if args.images:
        paths = [Path(line.strip()) for line in args.images.read_text().splitlines() if line.strip()]
        paths = [(p if p.is_absolute() else args.images.resolve().parent / p).resolve() for p in paths]
    elif args.image_dir:
        paths = sorted(p.resolve() for p in args.image_dir.rglob('*')
                       if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.webp'))
    else:
        paths = [p.resolve() for p in args.pair]
    if len(paths) < 2 or any(not p.is_file() for p in paths):
        parser.error('Require at least two existing images')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'features').mkdir()
    (args.output / 'matches').mkdir()
    (args.output / 'images.txt').write_text(''.join(str(p) + '\n' for p in paths))
    pairs = list(itertools.combinations(range(len(paths)), 2))
    np.savetxt(args.output / 'pairs.txt', pairs, fmt='%d')
    report = provenance(cfg)
    report.update(images=len(paths), pairs=len(pairs), records=[], state='running',
        timing_scope='Warm synchronized wall time. Extraction includes JPEG decode, H2D and CPU features. '
                     'Matching includes LRU input/H2D and CPU matches. Feature/match file writes excluded. '
                     'All inputs recomputed each repeat, GPU LRU starts empty. No geometry or retrieval.')
    extractor, report['extractor_load_seconds'] = timed(lambda: FeatureExtractor(cfg), cfg.device)
    _, report['extraction_warmup_seconds'] = timed(lambda: extractor.extract(paths[0]), cfg.device)
    for repeat in range(args.repeats):
        features, extraction_s = timed(lambda: [extractor.extract(p) for p in paths], cfg.device)
        if repeat == 0:
            matcher, report['matcher_load_seconds'] = timed(lambda: Matcher(cfg), cfg.device)
            def warmup():
                for index in sorted(set([0, len(pairs)//2, len(pairs)-1])):
                    i, j = pairs[index]
                    matcher.match(features[i], features[j])
            _, report['matching_warmup_seconds'] = timed(warmup, cfg.device)
        matcher.clear_cache()
        if torch.device(cfg.device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(cfg.device)
        matches, matching_s = timed(lambda: {(i, j): matcher.match(features[i], features[j]) for i, j in pairs}, cfg.device)
        row = dict(repeat=repeat+1, extraction_seconds=extraction_s, matching_seconds=matching_s,
                   milliseconds_per_pair=matching_s*1000/len(pairs), pairs_per_second=len(pairs)/matching_s,
                   matches=sum(len(m['indices']) for m in matches.values()),
                   rejected_pairs=sum(m['metadata'].get('rejected', False) for m in matches.values()),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(cfg.device) if torch.device(cfg.device).type == 'cuda' else None)
        report['records'].append(row)
        print(json.dumps(row), flush=True)
        if repeat == args.repeats - 1:
            for i, feature in enumerate(features):
                torch.save(feature, args.output / 'features' / f'{i}.pt')
            for (i, j), match in matches.items():
                torch.save(match, args.output / 'matches' / f'{i}_{j}.pt')
        report['state'] = 'complete' if repeat == args.repeats-1 else 'running'
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Saved to', args.output.resolve())


if __name__ == '__main__':
    main()
