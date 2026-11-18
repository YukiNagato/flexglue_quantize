"""Packed FP8 linear inference with frozen activation scales."""
import torch
from torch import nn
import torch.nn.functional as F


class FP8Linear(nn.Module):
    def __init__(self, module, static_amax=None):
        super().__init__()
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.pad_in = -self.in_features % 16
        self.pad_out = -self.out_features % 16
        weight = F.pad(module.weight.detach(), (0, self.pad_in, 0, self.pad_out))
        scale = weight.abs().amax().clamp_min(1e-12) / 448.
        self.register_buffer('weight_scale', scale.reshape(1, 1))
        self.register_buffer('weight', (weight / scale).clamp(-448, 448).to(torch.float8_e4m3fn))
        self.register_buffer('bias', None if module.bias is None else F.pad(module.bias.detach(), (0, self.pad_out)))
        self.register_buffer('bias_half', None if self.bias is None else self.bias.half())
        self.register_buffer('activation_scale', None if static_amax is None else
                             (static_amax.float().clamp_min(1e-12) * 1.1 / 448.).reshape(1, 1))

    def forward(self, x):
        shape = (*x.shape[:-1], self.out_features)
        out_dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled('cuda') else x.dtype
        x = x.reshape(-1, self.in_features)
        if self.pad_in:
            x = F.pad(x, (0, self.pad_in))
        scale = self.activation_scale
        if scale is None:
            scale = (x.float().abs().amax().clamp_min(1e-12) / 448.).reshape(1, 1)
        xq = (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
        with torch.autocast('cuda', enabled=False):
            out = torch._scaled_mm(xq, self.weight.t(), scale_a=scale, scale_b=self.weight_scale,
                bias=self.bias_half if out_dtype == torch.float16 else None,
                out_dtype=out_dtype, use_fast_accum=True)
        if self.bias is not None and out_dtype != torch.float16:
            out = out + self.bias.to(out_dtype)
        return out[:, :self.out_features].reshape(shape)

