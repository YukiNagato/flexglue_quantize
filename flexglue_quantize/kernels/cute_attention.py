"""Package-owned SM89+ INT8 QK / FP8 PV attention for FlexGlue (head dim 64)."""

import torch
import torch.nn.functional as F
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import from_dlpack

from .cute_attention_preprocess import quantize_qk, quantize_v


@dsl_user_op
def _qk_mma(a, b, c, *, loc=None, ip=None):
    ty = T.i32()
    result = llvm.inline_asm(
        llvm.StructType.get_literal([ty] * 4),
        [x.ir_value() for x in (*a, *b, *c)],
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32"
        " {$0,$1,$2,$3},{$4,$5,$6,$7},{$8,$9},{$10,$11,$12,$13};",
        ",".join(["=r"] * 4 + ["r"] * 10),
        asm_dialect=0,
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(ty, result, [i])) for i in range(4))


@dsl_user_op
def _pack(a, b, c, d, *, loc=None, ip=None):
    result = llvm.inline_asm(
        T.i32(),
        [x.ir_value() for x in (a, b, c, d)],
        "{ .reg .b16 lo,hi; cvt.rn.satfinite.e4m3x2.f32 lo,$2,$1; "
        "cvt.rn.satfinite.e4m3x2.f32 hi,$4,$3; mov.b32 $0,{lo,hi}; }",
        "=r,f,f,f,f",
        asm_dialect=0,
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


@dsl_user_op
def _load_words(ptr, *, count, loc=None, ip=None):
    regs = ",".join("$" + str(i) for i in range(count))
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * count),
        [ptr.toint().ir_value()],
        f"ld.global.v{count}.b32 {{{regs}}}, [${count}];",
        ",".join(["=r"] * count + ["l"]),
        asm_dialect=0,
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), result, [i])) for i in range(count))


@dsl_user_op
def _pv(a, b, c, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value() for x in (*a, *b, *c)],
        "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {$0,$1},{$2,$3,$4,$5},{$6,$7},{$8,$9};",
        "=r,=r,r,r,r,r,r,r,r,r",
        asm_dialect=0,
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(llvm.extractvalue(T.i32(), result, [i])) for i in range(2))


@dsl_user_op
def _half2(word, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [word.ir_value()],
        "{ .reg .b16 lo,hi; mov.b32 {lo,hi},$2; cvt.f32.f16 $0,lo; cvt.f32.f16 $1,hi; }",
        "=f,=f,r",
        asm_dialect=0,
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), result, [i])) for i in range(2))


@cute.kernel
def _kernel(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    qs: cute.Tensor,
    ks: cute.Tensor,
    vs: cute.Tensor,
    out: cute.Tensor,
    length: cutlass.Constexpr,
    tiles: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    blk, h, b = cute.arch.block_idx()
    group = (tid % 32) // 4
    pair = tid % 4
    row = blk * 64 + (tid // 32) * 16
    qa = cute.make_rmem_tensor((2, 4), Int32)
    for z in cutlass.range_constexpr(2):
        for r in cutlass.range_constexpr(4):
            qa[z, r] = q[b, h, row + group + 8 * (r % 2), z * 8 + (r // 2) * 4 + pair]
    scale_q = qs[b, h, row // 32 * 8 + group]
    acc = cute.make_rmem_tensor((8, 4), Float32)
    acc.fill(0.0)
    maximum = cute.make_rmem_tensor((2,), Float32)
    maximum.fill(-1.0e30)
    denom = cute.make_rmem_tensor((2,), Float32)
    denom.fill(0.0)
    # Preserve the upstream scaling order in the last two key tiles.
    # Separate loops let the compiler specialize both paths without a per-MMA branch.
    blocks = cute.ceil_div(length, 32 * tiles)
    main_blocks = max(0, blocks - 2)
    for tail in cutlass.range_constexpr(2):
        for index in range(blocks - main_blocks if tail else main_blocks):
            block = index + (main_blocks if tail else 0)
            base = block * (32 * tiles)
            dequant = scale_q * ks[b, h, block * tiles // 2 * 4 + pair]
            scale = dequant * (1.4426950408889634 / 8)
            if cutlass.const_expr(tail):
                scale = Float32(1.4426950408889634 / 8)
            score = cute.make_rmem_tensor((tiles, 4, 4), Float32)
            for t in cutlass.range_constexpr(tiles):
                for octet in cutlass.range_constexpr(4):
                    koffset = cute.crd2idx(
                        (b, h, block * tiles + t, octet * 128 + (tid % 32) * 4),
                        k.layout,
                    )
                    words = _load_words(k.iterator + koffset, count=4)
                    dot = (Int32(0), Int32(0), Int32(0), Int32(0))
                    for z in cutlass.range_constexpr(2):
                        dot = _qk_mma(
                            (qa[z, 0], qa[z, 1], qa[z, 2], qa[z, 3]),
                            (words[2 * z], words[2 * z + 1]),
                            dot,
                        )
                    for i in cutlass.range_constexpr(4):
                        x = dot[i].to(Float32)
                        if cutlass.const_expr(tail):
                            x = x * dequant
                        if cutlass.const_expr(length % (32 * tiles) != 0):
                            if base + t * 32 + octet * 8 + 2 * pair + i % 2 >= length:
                                x = Float32(-1.0e30)
                        score[t, octet, i] = x
            rescale = cute.make_rmem_tensor((2,), Float32)
            for r in cutlass.range_constexpr(2):
                m = Float32(-1.0e30)
                for t in cutlass.range_constexpr(tiles):
                    for j in cutlass.range_constexpr(4):
                        m = cute.arch.fmax(
                            m,
                            cute.arch.fmax(score[t, j, 2 * r], score[t, j, 2 * r + 1]),
                        )
                m = cute.arch.fmax(maximum[r], m * scale - 8.807)
                for shift in cutlass.range_constexpr(2):
                    m = cute.arch.fmax(
                        m, cute.arch.shuffle_sync_bfly(m, offset=1 << shift)
                    )
                alpha = cute.math.exp2(maximum[r] - m, fastmath=True)
                maximum[r] = m
                d = denom[r] * alpha
                for t in cutlass.range_constexpr(tiles):
                    for j in cutlass.range_constexpr(4):
                        for i in cutlass.range_constexpr(2):
                            p = cute.math.exp2(
                                score[t, j, 2 * r + i] * scale - m, fastmath=True
                            )
                            score[t, j, 2 * r + i] = p
                    for j in cutlass.range_constexpr(2):
                        d += (
                            (score[t, 2 * j, 2 * r] + score[t, 2 * j, 2 * r + 1])
                            + score[t, 2 * j + 1, 2 * r]
                        ) + score[t, 2 * j + 1, 2 * r + 1]
                denom[r] = d
                rescale[r] = alpha
            # Each lane packs four adjacent probability bytes as one MMA operand.
            probs = cute.make_rmem_tensor((2, 4), Int32)
            for t in cutlass.range_constexpr(2):
                for s in cutlass.range_constexpr(4):
                    j = s // 2 * 2
                    i = s % 2 * 2
                    probs[t, s] = _pack(
                        score[t, j, i],
                        score[t, j, i + 1],
                        score[t, j + 1, i],
                        score[t, j + 1, i + 1],
                    )
            for j in cutlass.range_constexpr(8):
                partial = (Int32(0), Int32(0))
                for t in cutlass.range_constexpr(2):
                    voffset = cute.crd2idx(
                        (b, h, block * 2 + t, j * 64 + (tid % 32) * 2), v.layout
                    )
                    words = _load_words(v.iterator + voffset, count=2)
                    partial = _pv(
                        (probs[t, 0], probs[t, 1], probs[t, 2], probs[t, 3]),
                        words,
                        partial,
                    )
                for r in cutlass.range_constexpr(2):
                    lo, hi = _half2(partial[r])
                    acc[j, r * 2] = acc[j, r * 2] * rescale[r] + lo
                    acc[j, r * 2 + 1] = acc[j, r * 2 + 1] * rescale[r] + hi
    for r in cutlass.range_constexpr(2):
        d = denom[r]
        for shift in cutlass.range_constexpr(2):
            d += cute.arch.shuffle_sync_bfly(d, offset=1 << shift)
        inv = cute.math.rcp(d, approx=True)
        for j in cutlass.range_constexpr(8):
            for i in cutlass.range_constexpr(2):
                col = j * 8 + 2 * pair + i
                out[b, h, row + group + 8 * r, col] = (
                    acc[j, 2 * r + i] * inv * vs[b, h, col]
                ).to(cutlass.Float16)


@cute.jit
def _launch(
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    qs: cute.Tensor,
    ks: cute.Tensor,
    vs: cute.Tensor,
    out: cute.Tensor,
    stream: cuda.CUstream,
    length: cutlass.Constexpr,
    tiles: cutlass.Constexpr,
):
    _kernel(q, k, v, qs, ks, vs, out, length, tiles).launch(
        grid=(q.shape[2] // 64, q.shape[1], q.shape[0]),
        block=(128, 1, 1),
        stream=stream,
    )


_cache = {}


def attention(q, k, v):
    if (
        not q.is_cuda
        or torch.cuda.get_device_capability(q.device) < (8, 9)
        or q.ndim != 4
        or k.shape != v.shape
        or k.ndim != 4
        or q.shape[:2] != k.shape[:2]
        or q.size(-1) != 64
        or k.size(-1) != 64
        or q.size(2) == 0
        or not 1 <= k.size(2) <= 4096
        or not all(t.device == q.device and t.dtype == torch.float16 for t in (q, k, v))
    ):
        raise ValueError(
            "CuTe attention requires SM89 or newer, FP16 HND, D=64 and 1..4096 keys"
        )
    rows, keys = q.size(2), k.size(2)
    mean = k.mean(dim=2, keepdim=True).contiguous()

    def pad(t, m):
        return F.pad(t, (0, 0, 0, -t.size(2) % m)) if t.size(2) % m else t.contiguous()

    q, k, v = pad(q, 128), pad(k, 256), pad(v, 256)
    qi, qs, ki, ks = quantize_qk(q, k, mean, mma_layout=True)
    vi, vs, _ = quantize_v(v, mma_layout=True)
    return _attention_from_packed(qi, ki, vi, qs, ks, vs, torch.empty_like(q), keys)[
        :, :, :rows
    ].contiguous()


def _attention_from_packed(qi, ki, vi, qs, ks, vs, out, keys):
    args = [
        from_dlpack(t)
        for t in (
            qi.view(torch.int32),
            ki.view(torch.int32),
            vi.view(torch.int32),
            qs,
            ks,
            vs,
            out,
        )
    ]
    stream = cuda.CUstream(torch.cuda.current_stream(out.device).cuda_stream)
    tiles = 2
    key = (out.device, tuple(qi.shape), tuple(ki.shape), keys, tiles)
    if key not in _cache:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up CuTe attention before CUDA Graph capture")
        _cache[key] = cute.compile(_launch, *args, stream, keys, tiles)
    _cache[key](*args, stream)
    return out
