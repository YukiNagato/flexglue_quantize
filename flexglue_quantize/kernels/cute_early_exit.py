"""Opt-in block-tail covisibility fusion; host early-exit decisions stay native.

Two CUDA kernels: residual + fixed FP8 quantization + per-token projection and
sigmoid, then mean reduction. The original packed weights/scales are retained.
Only FP32 residual / FP16 FFN output, 512 channels, FP16 scoring on SM89 or newer is
specialized. Unsupported cases use the original scoring operations.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.memory import SmemAllocator
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda


@cute.kernel
def tail_kernel(x0:cute.Tensor,x1:cute.Tensor,d0:cute.Tensor,d1:cute.Tensor,
                w:cute.Tensor,ws:cute.Tensor,sa:cute.Tensor,bias:cute.Tensor,
                y0:cute.Tensor,y1:cute.Tensor,prob:cute.Tensor):
    tid,_,_=cute.arch.thread_idx()
    bid,view,batch=cute.arch.block_idx()
    lane=tid%32;row=bid*4+tid//32
    acc=cutlass.Float32(0.)
    n=x0.shape[1]
    if view==1:n=x1.shape[1]
    if row<n:
        for j in cutlass.range_constexpr(16):
            col=j*32+lane
            value=cutlass.Float32(0.)
            if view==0:
                value=x0[batch,row,col]+d0[batch,row,col].to(cutlass.Float32)
                y0[batch,row,col]=value
            else:
                value=x1[batch,row,col]+d1[batch,row,col].to(cutlass.Float32)
                y1[batch,row,col]=value
            q=cutlass.min(cutlass.max(value/sa[0,0],-448.),448.).to(cutlass.Float8E4M3FN).to(cutlass.Float32)
            acc=acc+q*w[0,col].to(cutlass.Float32)
        acc=cute.arch.warp_reduction_sum(acc)
        if lane==0:
            logit=(acc*(sa[0,0]*ws[0,0])+bias[0].to(cutlass.Float32)).to(cutlass.Float16).to(cutlass.Float32)
            p=(1./(1.+cute.math.exp(-logit))).to(cutlass.Float16)
            offset=cutlass.Int32(0)
            if view==1:offset=x0.shape[1]
            prob[batch,offset+row]=p


@cute.kernel
def reduce_kernel(prob:cute.Tensor,score:cute.Tensor,stop:cute.Tensor,threshold:cutlass.Float32):
    tid,_,_=cute.arch.thread_idx();batch,_,_=cute.arch.block_idx()
    value=cutlass.Float32(0.)
    for i in cutlass.range_constexpr(cute.ceil_div(prob.shape[1],256)):
        row=i*256+tid
        if row<prob.shape[1]:value=value+prob[batch,row].to(cutlass.Float32)
    value=cute.arch.warp_reduction_sum(value)
    shared=SmemAllocator().allocate_tensor(cutlass.Float32,cute.make_layout(8))
    if tid%32==0:shared[tid//32]=value
    cute.arch.sync_threads()
    if tid<32:
        value=cutlass.Float32(0.)
        if tid<8:value=shared[tid]
        value=cute.arch.warp_reduction_sum(value)
        if tid==0:
            result=(value/prob.shape[1]).to(cutlass.Float16)
            score[batch]=result
            stop[batch]=result<threshold.to(cutlass.Float16)


@cute.jit
def launch(x0:cute.Tensor,x1:cute.Tensor,d0:cute.Tensor,d1:cute.Tensor,
           w:cute.Tensor,ws:cute.Tensor,sa:cute.Tensor,bias:cute.Tensor,
           y0:cute.Tensor,y1:cute.Tensor,prob:cute.Tensor,score:cute.Tensor,stop:cute.Tensor,threshold:cutlass.Float32,stream:cuda.CUstream):
    tail_kernel(x0,x1,d0,d1,w,ws,sa,bias,y0,y1,prob).launch(
        grid=(cute.ceil_div(max(x0.shape[1],x1.shape[1]),4),2,x0.shape[0]),block=(128,1,1),stream=stream)
    reduce_kernel(prob,score,stop,threshold).launch(grid=(x0.shape[0],1,1),block=(256,1,1),stream=stream)


_compiled={}

def run_tail(x0,x1,d0,d1,w,ws,sa,bias,threshold=0.,return_decision=False):
    y0=torch.empty_like(x0);y1=torch.empty_like(x1)
    prob=torch.empty((x0.shape[0],x0.shape[1]+x1.shape[1]),device=x0.device,dtype=torch.float16)
    score=torch.empty((x0.shape[0],),device=x0.device,dtype=torch.float16)
    stop=torch.empty((x0.shape[0],),device=x0.device,dtype=torch.bool)
    tensors=(x0,x1,d0,d1,w,ws,sa,bias,y0,y1,prob,score,stop)
    key=(x0.device.index,tuple((tuple(t.shape),tuple(t.stride()),t.dtype) for t in tensors))
    stream=cuda.CUstream(torch.cuda.current_stream(x0.device).cuda_stream)
    args=[from_dlpack(t,assumed_align=16) for t in tensors]
    if key not in _compiled:
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm up CuTe early-exit kernel before capture')
        _compiled[key]=cute.compile(launch,*args,cutlass.Float32(threshold),stream)
    _compiled[key](*args,cutlass.Float32(threshold),stream)
    return (y0,y1,score,stop) if return_decision else (y0,y1,score)


@torch.library.custom_op('flexglue_quantize::early_tail',mutates_args=())
def tail_op(x0:torch.Tensor,x1:torch.Tensor,d0:torch.Tensor,d1:torch.Tensor,
            w:torch.Tensor,ws:torch.Tensor,sa:torch.Tensor,bias:torch.Tensor,threshold:float)->tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
    return run_tail(x0,x1,d0,d1,w,ws,sa,bias,threshold,True)

@tail_op.register_fake
def _(x0,x1,d0,d1,w,ws,sa,bias,threshold):
    return (torch.empty_like(x0),torch.empty_like(x1),torch.empty((x0.shape[0],),device=x0.device,dtype=torch.float16),
            torch.empty((x0.shape[0],),device=x0.device,dtype=torch.bool))

