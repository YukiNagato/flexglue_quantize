"""Synthetic calibration helper retained for historical benchmarks."""
import torch
from torch import nn
from flexglue_quantize.linear import FP8Linear

@torch.inference_mode()
def replace_fp8(model, pairs, static=False):
    ranges, handles = {}, []
    linears = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if static:
        def hook(name):
            def observe(module, args):
                amax = args[0].detach().float().abs().amax()
                ranges[name] = amax if name not in ranges else torch.maximum(ranges[name], amax)
            return observe
        for name, module in linears:
            handles.append(module.register_forward_pre_hook(hook(name)))
        try:
            for pair in pairs:
                model.match(*pair, sparse=True)
            torch.cuda.synchronize()
        finally:
            for h in handles:
                h.remove()
    coverage = []
    for name, module in linears:
        # Unexecuted early-exit heads still get quantized; their range is unused here.
        amax = ranges.get(name, torch.ones((), device=module.weight.device)) if static else None
        parent, _, child = name.rpartition('.')
        replacement = FP8Linear(module, amax)
        setattr(model.get_submodule(parent), child, replacement)
        coverage.append(dict(name=name, shape=[module.out_features, module.in_features],
                             quantized=True, weight_type='FP8Linear', dtype=str(replacement.weight.dtype),
                             synthetic_range_observed=name in ranges,
                             padding=[replacement.pad_out, replacement.pad_in]))
    return coverage
