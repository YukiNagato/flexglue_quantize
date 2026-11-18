"""Editable CuTe DSL kernels for SageAttention's HND FP16/D=64 prologue.

Keep Sage's per-thread scale groups and the FP16 rounding after K centering.
Unsupported inputs fall back to the original implementation. No weights change.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.memory import SmemAllocator
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda


@cute.kernel
def qk_kernel(q: cute.Tensor, k: cute.Tensor, km: cute.Tensor,
              qo: cute.Tensor, ko: cute.Tensor, qs: cute.Tensor, ks: cute.Tensor, mma_layout: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    block, head, batch = cute.arch.block_idx()
    lane = tid % 32
    warp = tid // 32
    qblocks = cute.ceil_div(q.shape[2], 32) * 2
    # Four independent scale groups per CTA; a whole warp cooperates on each.
    if block < qblocks:
        group = block * 4 + warp
        row_base = group // 8 * 32 + group % 8
        vals = cute.make_rmem_tensor((8,), cutlass.Float32)
        amax = cutlass.Float32(0.)
        for i in cutlass.range_constexpr(8):
            row = row_base + (i // 2) * 8
            col = lane + (i % 2) * 32
            x = cutlass.Float32(0.)
            if row < q.shape[2]:
                x = q[batch, head, row, col].to(cutlass.Float32)
            vals[i] = x
            amax = cute.arch.fmax(amax, cute.arch.fmax(x, -x))
        amax = cute.arch.warp_reduction_max(amax)
        scale = amax * (1. / 127.) + 0.0000001
        if lane == 0:
            qs[batch, head, group] = scale
        inv = cute.math.rcp(scale, approx=True, ftz=True)
        for i in cutlass.range_constexpr(8):
            row = row_base + (i // 2) * 8
            col = lane + (i % 2) * 32
            x = vals[i] * inv
            bias = cutlass.Float32(-0.5)
            if x >= 0:
                bias = cutlass.Float32(0.5)
            if row < q.shape[2]:
                qo[batch, head, row, col] = (x + bias).to(cutlass.Int8)
    else:
        group = (block - qblocks) * 4 + warp
        row_base = group // 4 * 64 + group % 4 * 2
        vals = cute.make_rmem_tensor((32,), cutlass.Float32)
        amax = cutlass.Float32(0.)
        for i in cutlass.range_constexpr(32):
            row = row_base + (i // 4) * 8 + (i // 2) % 2
            col = lane + (i % 2) * 32
            x = cutlass.Float32(0.)
            if row < k.shape[2]:
                # The original torch subtraction materializes FP16 before Triton.
                x = (k[batch, head, row, col].to(cutlass.Float32)
                     - km[batch, head, 0, col].to(cutlass.Float32)).to(cutlass.Float16).to(cutlass.Float32)
            vals[i] = x
            amax = cute.arch.fmax(amax, cute.arch.fmax(x, -x))
        amax = cute.arch.warp_reduction_max(amax)
        scale = amax * (1. / 127.) + 0.0000001
        if lane == 0:
            ks[batch, head, group] = scale
        inv = cute.math.rcp(scale, approx=True, ftz=True)
        for i in cutlass.range_constexpr(32):
            row = row_base + (i // 4) * 8 + (i // 2) % 2
            col = lane + (i % 2) * 32
            x = vals[i] * inv
            bias = cutlass.Float32(-0.5)
            if x >= 0:
                bias = cutlass.Float32(0.5)
            if row < k.shape[2]:
                value = (x + bias).to(cutlass.Int8)
                if cutlass.const_expr(mma_layout):
                    tile = row // 32
                    octet = row % 32 // 8
                    subgroup = row % 8
                    mma_lane = subgroup*4 + col%16//4
                    word = col//16
                    ko[batch,head,tile,octet*512+mma_lane*16+word*4+col%4] = value
                else:
                    ko[batch, head, row, col] = value


@cute.jit
def launch_qk(q: cute.Tensor, k: cute.Tensor, km: cute.Tensor,
              qo: cute.Tensor, ko: cute.Tensor, qs: cute.Tensor, ks: cute.Tensor,
              stream: cuda.CUstream, mma_layout: cutlass.Constexpr):
    blocks = cute.ceil_div(q.shape[2],32)*2 + cute.ceil_div(k.shape[2],64)
    qk_kernel(q,k,km,qo,ko,qs,ks,mma_layout).launch(grid=(blocks,q.shape[1],q.shape[0]),block=(128,1,1),stream=stream)


_compiled = {}


def quantize_qk(q, k, km, *, mma_layout=False):
    """Specialized contiguous HND FP16 path; reject unsupported direct calls."""
    if not supports_qk(q,k,km):
        raise ValueError('CuTe QK requires contiguous HND FP16/D=64 tensors with aligned lengths on SM89 or newer')
    b,h,n,d=q.shape
    nk=k.shape[2]
    qo=torch.empty_like(q,dtype=torch.int8)
    ko=torch.empty((b,h,nk//32,2048),device=k.device,dtype=torch.int8) if mma_layout else torch.empty_like(k,dtype=torch.int8)
    qs=torch.empty((b,h,((n+127)//128)*32),device=q.device,dtype=torch.float32)
    ks=torch.empty((b,h,((nk+63)//64)*4),device=q.device,dtype=torch.float32)
    tensors=[from_dlpack(t) for t in (q,k,km,qo,ko,qs,ks)]
    stream=cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    key=(q.device.index,tuple(q.shape),tuple(k.shape),mma_layout)
    if key not in _compiled:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Warm up CuTe QK specialization before CUDA graph capture')
        _compiled[key]=cute.compile(launch_qk,*tensors,stream,mma_layout)
    _compiled[key](*tensors,stream)
    return qo,qs,ko,ks


def supports_qk(q,k,km,BLKQ=128,WARPQ=32,BLKK=64,WARPK=64,tensor_layout='HND'):
    return (tensor_layout=='HND' and (BLKQ,WARPQ,BLKK,WARPK)==(128,32,64,64)
        and q.is_cuda and q.device==k.device and torch.cuda.get_device_capability(q.device) >= (8, 9)
        and q.dtype==k.dtype==torch.float16 and q.ndim==k.ndim==4
        and q.shape[:2]==k.shape[:2] and q.shape[-1]==k.shape[-1]==64
        and q.shape[2]>0 and k.shape[2]>0 and q.shape[2]%128==0 and k.shape[2]%64==0
        and q.is_contiguous() and k.is_contiguous() and km is not None
        and km.device==q.device and km.dtype==q.dtype and tuple(km.shape)==(*k.shape[:2],1,64) and km.is_contiguous())


def wrap_qk(original):
    def wrapped(q,k,km=None,BLKQ=128,WARPQ=32,BLKK=64,WARPK=64,sm_scale=None,tensor_layout='HND'):
        if supports_qk(q,k,km,BLKQ,WARPQ,BLKK,WARPK,tensor_layout):
            return quantize_qk(q,k,km)
        return original(q,k,km,BLKQ,WARPQ,BLKK,WARPK,sm_scale,tensor_layout)
    wrapped._cute_qk=True
    return wrapped


@cute.kernel
def v_kernel(v: cute.Tensor, out: cute.Tensor, scales: cute.Tensor, warp_layout: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    channel_block, head, batch = cute.arch.block_idx()
    channel = channel_block*4+tid%4
    vals = cute.make_rmem_tensor((cute.ceil_div(v.shape[2],64),),cutlass.Float32)
    amax = cutlass.Float32(0.)
    for i in cutlass.range_constexpr(cute.ceil_div(v.shape[2],64)):
        row = i*64+tid//4
        x = v[batch,head,row,channel].to(cutlass.Float32)
        vals[i] = x
        amax = cute.arch.fmax(amax,cute.arch.fmax(x,-x))
    for shift in cutlass.range_constexpr(3):
        amax = cute.arch.fmax(amax,cute.arch.shuffle_sync_bfly(amax,offset=4<<shift))
    shared = SmemAllocator().allocate_tensor(cutlass.Float32,cute.make_layout(32))
    if tid%32 < 4:
        shared[tid//32*4+tid%4] = amax
    cute.arch.sync_threads()
    amax = cutlass.Float32(0.)
    for i in cutlass.range_constexpr(8):
        amax = cute.arch.fmax(amax,shared[i*4+tid%4])
    if tid < 4:
        scales[batch,head,channel] = amax*(1./2.25)
    inv = cutlass.Float32(0.)
    if amax > 0:
        inv = cute.math.rcp(amax,approx=True,ftz=True)*2.25
    for i in cutlass.range_constexpr(cute.ceil_div(v.shape[2],64)):
        row = i*64+tid//4
        mod = row%16
        perm = (row//16)*16+(mod//8)*2+((mod//2)%4)*4+mod%2
        value = (vals[i]*inv).to(cutlass.Float8E4M3FN)
        if cutlass.const_expr(warp_layout == 2):
            lane = channel%8*4 + perm%16//4
            out[batch,head,row//32,channel//8*256+lane*8+perm%32//16*4+perm%4] = value
        elif cutlass.const_expr(warp_layout):
            out[batch, head, row//32, channel*32 + row%32] = value
        else:
            out[batch,head,channel,perm] = value


@cute.jit
def launch_v(v: cute.Tensor, out: cute.Tensor, scales: cute.Tensor, stream: cuda.CUstream, warp_layout: cutlass.Constexpr):
    v_kernel(v,out,scales,warp_layout).launch(grid=(16,v.shape[1],v.shape[0]),block=(256,1,1),stream=stream)


_compiled_v={}


def quantize_v(v, *, warp_layout=False, mma_layout=False):
    if not supports_v(v):
        raise ValueError('CuTe V requires contiguous HND FP16/D=64 and length 256..4096 divisible by 256 on SM89 or newer')
    b,h,n,d=v.shape
    layout = 2 if mma_layout else int(warp_layout)
    out=torch.empty((b,h,n//32,d*32) if layout else (b,h,d,n),device=v.device,dtype=torch.float8_e4m3fn)
    scale=torch.empty((b,h,d),device=v.device,dtype=torch.float32)
    tensors=[from_dlpack(t) for t in (v,out,scale)]
    stream=cuda.CUstream(torch.cuda.current_stream(v.device).cuda_stream)
    key=(v.device.index,tuple(v.shape),layout)
    if key not in _compiled_v:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Warm up CuTe V specialization before CUDA graph capture')
        _compiled_v[key]=cute.compile(launch_v,*tensors,stream,layout)
    _compiled_v[key](*tensors,stream)
    return out,scale,None


def supports_v(v,tensor_layout='HND',scale_max=2.25,smooth_v=False):
    return (v.is_cuda and torch.cuda.get_device_capability(v.device) >= (8, 9)
        and v.dtype==torch.float16 and v.ndim==4 and v.shape[-1]==64
        and v.is_contiguous() and 256<=v.shape[2]<=4096 and v.shape[2]%256==0
        and tensor_layout=='HND' and scale_max==2.25 and not smooth_v)


def wrap_v(original):
    def wrapped(v,tensor_layout='HND',scale_max=448.,smooth_v=True):
        if supports_v(v,tensor_layout,scale_max,smooth_v):
            return quantize_v(v)
        return original(v,tensor_layout,scale_max,smooth_v)
    wrapped._cute_v=True
    return wrapped
