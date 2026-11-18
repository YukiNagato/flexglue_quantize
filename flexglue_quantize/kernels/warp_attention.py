"""SM89+ noncausal Sage-style attention. One warp processes 16 query rows."""
import torch
import torch.nn.functional as F
import warp as wp

from .cute_attention_preprocess import quantize_qk, quantize_v
wp.init()

# The m16n8k32 accumulator fragment holds two columns in each of two rows
# per lane. Keep the running maximum, denominator and PV result in registers.
@wp.func_native(r"""
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const int lane = threadIdx.x % 32, group = lane / 4, pair = lane % 4;
    const int row = (tile % (nq / 16)) * 16;
    const int head = tile / (nq / 16);
    const auto Q = q.data + head*nq*64;
    const auto K = k.data + head*nk*64;
    const auto V = v.data + head*nk*64;
    unsigned qa[2][4];
    #pragma unroll
    for (int half=0; half<2; ++half)
        #pragma unroll
        for (int r=0; r<4; ++r)
            qa[half][r] = *(const unsigned*)(Q+(row+group+8*(r%2))*64+half*32+(r/2)*16+pair*4);
    float maximum[2]={-1.e30f,-1.e30f}, sum[2]={}, result[8][4]={};
    const float scale_q = qs.data[head*(nq/4)+(row/32)*8+group]*0.125f;
    for (int base=0; base<length; base+=32) {
        float scores[4][4];
        #pragma unroll
        for (int oct=0; oct<4; ++oct) {
            int dot[4]={};
            #pragma unroll
            for (int half=0; half<2; ++half) {
                // Arrange pairs of K rows in the register order needed by FP8 PV.
                int key=(oct/2)*16+(group/2)*4+(oct%2)*2+group%2;
                auto b=(const unsigned*)(K+(base+key)*64+half*32+pair*4);
                asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                    : "+r"(dot[0]),"+r"(dot[1]),"+r"(dot[2]),"+r"(dot[3])
                    : "r"(qa[half][0]),"r"(qa[half][1]),"r"(qa[half][2]),"r"(qa[half][3]),"r"(b[0]),"r"(b[4]));
            }
            float scale=scale_q*ks.data[head*(nk/16)+(base/64)*4+(pair%2)*2+oct%2];
            #pragma unroll
            for (int i=0;i<4;++i)
                scores[oct][i]=(length%32==0 || base+(oct/2)*16+4*pair+(oct%2)*2+i%2<length) ? dot[i]*scale : -1.e30f;
        }
        unsigned packed[2][4];
        #pragma unroll
        for (int r=0;r<2;++r) {
            float m=maximum[r];
            #pragma unroll
            for (int j=0;j<4;++j) m=fmaxf(m,fmaxf(scores[j][r*2],scores[j][r*2+1]));
            m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,1));
            m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,2));
            float rescale=__expf(maximum[r]-m), local=0;
            maximum[r]=m;
            #pragma unroll
            for (int j=0;j<4;++j) {
                float p0=__expf(scores[j][2*r]-m), p1=__expf(scores[j][2*r+1]-m);
                local+=p0+p1;
                unsigned short bytes;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0,%1,%2;" : "=h"(bytes) : "f"(p1*448.f),"f"(p0*448.f));
                packed[r][j]=(unsigned)bytes;
            }
            local+=__shfl_xor_sync(0xffffffff,local,1);
            local+=__shfl_xor_sync(0xffffffff,local,2);
            sum[r]=sum[r]*rescale+local;
            #pragma unroll
            for (int j=0;j<8;++j) {
                result[j][2*r]*=rescale;
                result[j][2*r+1]*=rescale;
            }
        }
        #pragma unroll
        for (int j=0;j<8;++j) {
            const auto vb=(const unsigned*)(V+base*64+(j*8+group)*32+pair*4);
            auto &c=result[j];
            asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
                : "r"(packed[0][0]|(packed[0][1]<<16)),"r"(packed[1][0]|(packed[1][1]<<16)),"r"(packed[0][2]|(packed[0][3]<<16)),"r"(packed[1][2]|(packed[1][3]<<16)),"r"(vb[0]),"r"(vb[4]));
        }
    }
    #pragma unroll
    for (int j=0;j<8;++j)
        #pragma unroll
        for (int i=0;i<4;++i) {
            int col=j*8+2*pair+i%2;
            out.data[((head*nq+row+group+8*(i/2))*64)+col]=wp::float16(result[j][i]*vs.data[head*64+col]/(448.f*sum[i/2]));
        }
#endif
""")
def _tile(q: wp.array(dtype=wp.int8), k: wp.array(dtype=wp.int8),
          v: wp.array(dtype=wp.uint8), qs: wp.array(dtype=float),
          ks: wp.array(dtype=float), vs: wp.array(dtype=float),
          out: wp.array(dtype=wp.float16), tile: int, nq: int, nk: int, length: int):
    ...


@wp.kernel(enable_backward=False)
def _attention(q: wp.array(dtype=wp.int8), k: wp.array(dtype=wp.int8),
               v: wp.array(dtype=wp.uint8), qs: wp.array(dtype=float),
               ks: wp.array(dtype=float), vs: wp.array(dtype=float),
               out: wp.array(dtype=wp.float16), nq: int, nk: int, length: int):
    _tile(q, k, v, qs, ks, vs, out, wp.tid() // 32, nq, nk, length)


def attention(q, k, v):
    if (not q.is_cuda or torch.cuda.get_device_capability(q.device) < (8, 9)
            or q.ndim != 4 or k.ndim != 4 or k.shape != v.shape
            or q.shape[:2] != k.shape[:2] or q.shape[-1] != 64 or k.shape[-1] != 64
            or not all(t.dtype == torch.float16 and t.device == q.device for t in (q,k,v))
            or q.size(2) == 0 or not 1 <= k.size(2) <= 4096):
        raise ValueError("Warp attention requires SM89 or newer, FP16 HND, D=64 and 1..4096 keys")
    rows, keys = q.size(2), k.size(2)
    mean = k.mean(dim=2, keepdim=True).contiguous()
    def pad(t, multiple):
        extra = -t.size(2) % multiple
        return F.pad(t, (0, 0, 0, extra)) if extra else t.contiguous()
    q, k, v = pad(q, 128), pad(k, 256), pad(v, 256)
    qi, qs, ki, ks = quantize_qk(q, k, mean)
    vi, vs, _ = quantize_v(v, warp_layout=True)
    out = torch.empty_like(q)
    arrays = [wp.from_torch(t.flatten()) for t in (qi, ki, vi.view(torch.uint8), qs, ks, vs, out)]
    wp.launch(_attention, dim=q.numel()//64*2, inputs=[*arrays, q.size(2), k.size(2), keys],
              stream=wp.stream_from_torch(torch.cuda.current_stream(q.device)), block_dim=128)
    return out[:, :, :rows].contiguous()
