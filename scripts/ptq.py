"""Real-data version of the previously fastest 8-bit FlexGlue combination.

Linear modules use the editable FP8 E4M3 implementation, with aligned padding
and frozen per-tensor activation ranges plus the original 10% margin. The
validated variant retains posenc.Wr in float via keep_posenc_float=True.
Attention uses SageAttention QK INT8 / PV FP8, HND, per_thread, smooth_k=True.
No fitting is performed when applying or loading this model.
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from flexglue_quantize.linear import FP8Linear
from flexglue_quantize import load_best_ptq

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def apply_best_ptq(model, calibration, keep_posenc_float=False):
    calibration = Path(calibration)
    info = json.loads((calibration / 'calibration.json').read_text())
    assert info['state'] == 'complete' and info['validation_images_used'] == 0
    assert digest(calibration / 'observers.pt') == info['observers_sha256']
    assert digest(model.conf.weights) == info['source_checkpoint_sha256']
    state = torch.load(calibration / 'observers.pt', weights_only=True)
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    assert set(n for n, _ in linears) == set(state), 'Calibration must cover every Linear'
    coverage = []
    for name, module in linears:
        if keep_posenc_float and name == 'posenc.Wr':
            coverage.append(dict(name=name, shape=list(module.weight.shape), quantized=False,
                weight_type='Parameter', dtype=str(module.weight.dtype),
                reason='Calibration-pair diagnosis: Fourier phase is sensitive to FP8 projection errors'))
            continue
        stats = state[name]
        assert list(module.weight.shape) == info['statistics'][name]['weight_shape']
        assert info['statistics'][name]['calls'] > 0
        amax = torch.maximum(stats['min_val'].abs(), stats['max_val'].abs())
        assert amax.numel() == 1 and torch.isfinite(amax).all()
        replacement = FP8Linear(module, amax.to(module.weight.device))
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent), child, replacement)
        coverage.append(dict(name=name, shape=[module.out_features, module.in_features],
            quantized=True, weight_type='FP8Linear', dtype=str(replacement.weight.dtype),
            padding=[replacement.pad_out, replacement.pad_in],
            activation_scale=float(replacement.activation_scale.item())))
    model.eval()
    assert len(coverage) == 171
    return dict(kind='fp8custom-static', artifact=str(calibration.resolve()),
        artifact_sha256=info['observers_sha256'], calibration_pair_count=info['calibration_pairs'],
        keep_posenc_float=keep_posenc_float,
        calibration_dataset_manifest_sha256=info['calibration_manifest_sha256'],
        activation_granularity='PerTensor static, real calibration MinMax + 10% range margin',
        weight_granularity='PerTensor E4M3', linear_coverage=coverage,
        attention='SageAttention QK INT8 / PV FP8, HND, per_thread, smooth_k=True, fp32+fp16 accumulation')


@torch.inference_mode()
def export_best_ptq(model, info, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    count = sum(row['quantized'] for row in info['linear_coverage'])
    assert len([m for m in model.modules() if isinstance(m, FP8Linear)]) == count
    assert sum(isinstance(m, nn.Linear) for m in model.modules()) == 171-count
    conf = dict(vars(model.conf))
    conf['weights'] = None
    torch.save(dict(format='flexglue-best-ptq-v1', conf=conf, ptq=info,
        state_dict={n: t.detach().cpu() for n, t in model.state_dict().items()}), destination)
    return dict(path=str(destination.resolve()), sha256=digest(destination), bytes=destination.stat().st_size)


def main():
    from flexglue_quantize.core import FlexGlue
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=ROOT/'results/ptq_realdata_calibration')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT/'weights/flexglue.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--keep-posenc-float', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model = FlexGlue(weights=str(args.weights), describe=False, refine=False,
                     min_covisibility=.9).eval().to(args.device)
    info = apply_best_ptq(model, args.calibration, args.keep_posenc_float)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(export_best_ptq(model, info, args.output), indent=2))


if __name__ == '__main__':
    main()
