"""Fuse top512 sort/temperature/window gather; retain native entmax arithmetic."""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda


@cute.kernel
def sort_window_kernel(logits:cute.Tensor,top:cute.Tensor,indices:cute.Tensor,temperature:cute.Tensor,
                       sorted_values:cute.Tensor,max_indices:cute.Tensor,windows:cute.Tensor):
    tid,_,_=cute.arch.thread_idx();bid,_,_=cute.arch.block_idx()
    lane=tid%32;row=bid*4+tid//32
    if row<top.shape[0]:
        vals=cute.make_rmem_tensor((16,),cutlass.Float32)
        order=cute.make_rmem_tensor((16,),cutlass.Int32)
        nextv=cute.make_rmem_tensor((16,),cutlass.Float32)
        nexto=cute.make_rmem_tensor((16,),cutlass.Int32)
        for i in cutlass.range_constexpr(16):
            vals[i]=top[row,i*32+lane]*temperature[0];order[i]=i*32+lane
        for stage in cutlass.range_constexpr(1,10):
            for step in cutlass.range_constexpr(stage-1,-1,-1):
                for i in cutlass.range_constexpr(16):
                    a=vals[i];ai=order[i]
                    if cutlass.const_expr(step>=5):
                        b=vals[i^(1<<(step-5))];bi=order[i^(1<<(step-5))]
                    else:
                        b=cute.arch.shuffle_sync_bfly(a,offset=1<<step)
                        bi=cute.arch.shuffle_sync_bfly(ai,offset=1<<step)
                    index=i*32+lane
                    descending=((index&(1<<stage))==0)==((index&(1<<step))==0)
                    take=((a<b)|((a==b)&(ai>bi)))
                    if not descending:take=((a>b)|((a==b)&(ai<bi)))
                    nv=a;ni=ai
                    if take:nv=b;ni=bi
                    nextv[i]=nv;nexto[i]=ni
                for i in cutlass.range_constexpr(16):
                    vals[i]=nextv[i];order[i]=nexto[i]
        for i in cutlass.range_constexpr(16):sorted_values[row,i*32+lane]=vals[i]
        first=cute.arch.shuffle_sync(order[0],offset=0)
        idx=indices[row,first].to(cutlass.Int32)
        if lane==0:max_indices[row,0]=idx
        if lane<9:
            iy=cutlass.min(cutlass.max(idx//64+lane//3-1,0),63)
            ix=cutlass.min(cutlass.max(idx%64+lane%3-1,0),63)
            windows[row,lane]=logits[row,iy*64+ix]*temperature[0]


@cute.jit
def launch(logits:cute.Tensor,top:cute.Tensor,indices:cute.Tensor,temperature:cute.Tensor,
           sorted_values:cute.Tensor,max_indices:cute.Tensor,windows:cute.Tensor,stream:cuda.CUstream):
    sort_window_kernel(logits,top,indices,temperature,sorted_values,max_indices,windows).launch(
        grid=(cute.ceil_div(top.shape[0],4),1,1),block=(128,1,1),stream=stream)


_compiled={}


def sorted_windows(logits,top,indices,temperature):
    rows=logits.numel()//4096
    if (not logits.is_cuda or logits.dtype!=torch.float32 or logits.shape[-1]!=4096 or rows==0
        or not logits.is_contiguous() or top.shape!=(*logits.shape[:-1],512) or indices.shape!=top.shape
        or top.dtype!=torch.float32 or indices.dtype!=torch.int64 or temperature.numel()!=1
        or temperature.dtype!=torch.float32 or not top.is_contiguous() or not indices.is_contiguous()
        or any(t.device!=logits.device for t in (top,indices,temperature))):raise ValueError('Unsupported sort/window inputs')
    values=torch.empty_like(top);idx=torch.empty((*logits.shape[:-1],1),device=logits.device,dtype=torch.int64)
    window=logits.new_empty((*logits.shape[:-1],9))
    tensors=[from_dlpack(t) for t in (logits.reshape(-1,4096),top.reshape(-1,512),indices.reshape(-1,512),
        temperature.reshape(1),values.reshape(-1,512),idx.reshape(-1,1),window.reshape(-1,9))]
    stream=cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream);key=(logits.device.index,rows)
    if key not in _compiled:
        if torch.cuda.is_current_stream_capturing():raise RuntimeError('Warm up sorted-window kernel before graph capture')
        _compiled[key]=cute.compile(launch,*tensors,stream)
    _compiled[key](*tensors,stream)
    return values,idx,window


def native_readout(topv,idx,window,grid):
    from ..utils.ops import sorted_entmax15_threshold
    vmax=topv[...,:1]
    tau=sorted_entmax15_threshold((topv-vmax)/2)
    r=torch.arange(-1,2,device=topv.device)
    ny=(idx//64)[...,None]+r[:,None];nx=(idx%64)[...,None]+r[None,:]
    inside=((ny>=0)&(ny<64)&(nx>=0)&(nx<64)).flatten(-2)
    nidx=(ny.clamp(0,63)*64+nx.clamp(0,63)).flatten(-2)
    p=((window-vmax)/2-tau).clamp_min(0).square()
    position=(p[...,None]*grid[nidx]).sum(-2)/p.sum(-1,keepdim=True)
    return position,(p*inside).sum(-1)
