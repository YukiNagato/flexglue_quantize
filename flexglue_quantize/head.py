"""Validated CuTe head readout, installed on individual heads."""
import math

import torch

from .utils import ops


@torch.library.custom_op("flexglue_quantize::sorted_windows", mutates_args=())
def sorted_windows(logits: torch.Tensor, top: torch.Tensor, indices: torch.Tensor,
                   temperature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from .kernels.cute_sorted_windows import sorted_windows as kernel
    return kernel(logits, top, indices, temperature)


@sorted_windows.register_fake
def _fake(logits, top, indices, temperature):
    return (top.new_empty(top.shape), indices.new_empty((*logits.shape[:-1], 1)),
            logits.new_empty((*logits.shape[:-1], 9)))


def precise_readout(query, search, temperature, grid, hw):
    if tuple(hw) != (64, 64) or query.dtype != torch.float32 or not query.is_cuda:
        return ops.match_logits_readout(query, search, temperature, grid, hw)
    logits = query @ search.mT
    supported = (logits.dtype == torch.float32 and logits.is_contiguous() and logits.numel() > 0
                 and grid.shape == (4096, 2) and grid.dtype == torch.float32 and grid.is_contiguous())
    if not supported:
        return ops.match_logits_readout(query, search, temperature, grid, hw)
    from .kernels.cute_sorted_windows import native_readout
    top, indices = logits.topk(512, dim=-1, sorted=False)
    values, idx, window = sorted_windows(logits, top, indices, temperature)
    return native_readout(values, idx, window, grid)


def install(model, backend):
    if backend not in ("none", "cute-precise"):
        raise ValueError(f"Unknown head backend: {backend}")
    if backend == "cute-precise":
        from .kernels import cute_sorted_windows  # Check the optional dependency now.
        for head in model.heads:
            temperature = float(head.temperature.detach())
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("CuTe readout requires positive finite temperatures.")
    for head in model.heads:
        head.readout = precise_readout if backend == "cute-precise" else ops.match_logits_readout
