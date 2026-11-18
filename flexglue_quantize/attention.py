"""SageAttention adapters isolated to each model; no global monkeypatching."""
from functools import lru_cache
from types import FunctionType

import torch


@lru_cache(maxsize=3)
def _implementation(preprocess):
    try:
        from sageattention import sageattn_qk_int8_pv_fp8_cuda
    except ImportError as error:
        raise ImportError("The sage backend requires SageAttention with its CUDA extensions.") from error
    fn = sageattn_qk_int8_pv_fp8_cuda
    if preprocess == "none":
        return fn
    from .kernels.cute_attention_preprocess import wrap_qk, wrap_v

    # A private globals dictionary lets Sage use our prologues without changing
    # sageattention.core or the behavior of another model in the same process.
    namespace = dict(fn.__globals__)
    namespace["per_thread_int8_triton"] = wrap_qk(namespace["per_thread_int8_triton"])
    if preprocess == "qkv":
        namespace["per_channel_fp8"] = wrap_v(namespace["per_channel_fp8"])
    adapted = FunctionType(fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__)
    adapted.__kwdefaults__ = fn.__kwdefaults__
    return adapted


@torch.library.custom_op("flexglue_quantize::sage", mutates_args=())
def sage(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, preprocess: str) -> torch.Tensor:
    return _implementation(preprocess)(
        q, k, v, tensor_layout="HND", is_causal=False,
        qk_quant_gran="per_thread", pv_accum_dtype="fp32+fp16",
        smooth_k=True, smooth_v=False,
    )


@sage.register_fake
def _fake(q, k, v, preprocess):
    return torch.empty_like(q, memory_format=torch.contiguous_format)


@torch.library.custom_op("flexglue_quantize::warp_attention", mutates_args=())
def warp_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    from .kernels.warp_attention import attention
    return attention(q, k, v)


@warp_attention.register_fake
def _fake_warp(q, k, v):
    return torch.empty_like(q, memory_format=torch.contiguous_format)


@torch.library.custom_op("flexglue_quantize::cute_attention", mutates_args=())
def cute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    from .kernels.cute_attention import attention
    return attention(q, k, v)


@cute_attention.register_fake
def _fake_cute(q, k, v):
    return torch.empty_like(q, memory_format=torch.contiguous_format)


def install(model, backend="cute", preprocess=None):
    from .utils.modules import attention as sdpa

    if preprocess is None:
        preprocess = "qkv" if backend in ("cute", "warp") else "none"
    if backend not in ("sage", "sdpa", "warp", "cute"):
        raise ValueError(f"Unknown attention backend: {backend}")
    if preprocess not in ("none", "qk", "qkv"):
        raise ValueError(f"Unknown attention preprocess: {preprocess}")
    if backend == "sdpa" and preprocess != "none":
        raise ValueError("CuTe QK/V preprocessing requires attention_backend='sage'.")
    if backend == "warp":
        from .kernels import warp_attention as _warp_module
    if backend == "sage":
        _implementation(preprocess)  # Fail before inference if a dependency is missing.

    def attend(q, k, v):
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else v.dtype
        q, k, v = (t.transpose(1, 2).to(dtype).contiguous() for t in (q, k, v))
        if backend == "cute":
            out = cute_attention(q, k, v)
        elif backend == "warp":
            out = warp_attention(q, k, v)
        else:
            out = sage(q, k, v, preprocess)
        return out.transpose(1, 2).flatten(-2)

    for block in model.transformers:
        block.self_attn.attention = attend if backend != "sdpa" else sdpa
        block.cross_attn.attention = attend if backend != "sdpa" else sdpa
