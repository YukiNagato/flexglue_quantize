"""Per-model early scoring backends; host rejection decisions remain native."""
import types
import torch

def install(model,backend):
    from .linear import FP8Linear
    if backend not in ('compiled','cute'):raise ValueError(backend)
    if backend == 'cute':
        from .kernels.cute_early_exit import tail_op
    if any(hasattr(b,'forward_with_early_score') for b in model.transformers):raise ValueError('Early backend already installed')
    def make_forward(head):
        def forward(block,x0,x1,enc0,enc1,threshold):
            if backend=='compiled':
                y0,y1=block(x0,x1,enc0,enc1)
                score=torch.sigmoid(head(torch.cat((y0,y1),1).float())).mean((1,2))
                return y0,y1,score,(score<threshold).all()
            x0=block.self_attn(x0,enc0);x1=block.self_attn(x1,enc1)
            c=block.cross_attn
            q0,q1,v0,v1=(t.unflatten(-1,(c.num_heads,-1)) for t in (c.to_qk(x0),c.to_qk(x1),c.to_v(x0),c.to_v(x1)))
            q0=c.qk_norm(q0).to(v0.dtype);q1=c.qk_norm(q1).to(v1.dtype)
            m0=c.to_out(c.attention(q0,q1,v1));m1=c.to_out(c.attention(q1,q0,v0))
            d0=c.ffn(torch.cat((x0,c.message_norm(m0)),-1));d1=c.ffn(torch.cat((x1,c.message_norm(m1)),-1))
            supported=(x0.is_cuda and torch.cuda.get_device_capability(x0.device) >= (8, 9) and x0.dtype==x1.dtype==torch.float32 and d0.dtype==d1.dtype==torch.float16
                       and x0.shape[-1]==x1.shape[-1]==512 and x0.shape[1]>0 and x1.shape[1]>0
                       and x0.is_contiguous() and x1.is_contiguous() and d0.is_contiguous() and d1.is_contiguous())
            if supported:
                y0,y1,score,stop=tail_op(x0,x1,d0,d1,head.weight,head.weight_scale,head.activation_scale,head.bias_half,threshold)
                return y0,y1,score,stop[0] if x0.shape[0]==1 else stop.all()
            y0,y1=x0+d0,x1+d1
            score=torch.sigmoid(head(torch.cat((y0,y1),1).float())).mean((1,2))
            return y0,y1,score,(score<threshold).all()
        return forward
    for k,block in enumerate(model.transformers,1):
        if k==len(model.transformers):continue
        head=model.heads[k].covisibility
        if not isinstance(head, FP8Linear) or head.activation_scale is None:
            if backend == 'cute':
                continue  # This layer uses decode_pair's ordinary covisibility check.
            raise ValueError('Requires frozen FP8 covisibility head')
        forward=make_forward(head)
        # Each layer has its own threshold and packed head. Give Dynamo a distinct
        # code object, avoiding a shared eight-specialization budget across all
        # layers/backends without changing global compiler configuration.
        name=f'early_layer_{k}_{backend}'
        forward=types.FunctionType(forward.__code__.replace(co_name=name),forward.__globals__,
                                   name=name,argdefs=forward.__defaults__,closure=forward.__closure__)
        block.forward_with_early_score=types.MethodType(forward,block)
    model.early_exit_backend=backend

