# CuTe attention on RTX 4090 D

The package's CuTe DSL backend requires compute capability 8.9 or higher, FP16 input, head dimension
64 and noncausal attention. It writes the quantized K and V values directly
in the lane order consumed by the Tensor Core instructions. Each warp
processes 16 query rows and updates online softmax over 64-key tiles. The last
two tiles preserve SageAttention's ordering of scale multiplications. Kernels
launch on the current PyTorch stream and support CUDA Graph replay after warmup.

| Measurement | Patched SageAttention 2.2.0 | Package CuTe |
|---|---:|---:|
| Attention kernel, prequantized inputs | 127.44 µs | 125.79 µs |
| Attention including centering and Q/K/V quantization | 144.48 µs | 145.33 µs |
| Full matcher, nine layers | 13.45 ms/pair | 13.47 ms/pair |
| Matcher with early rejection | 12.33 ms/pair | 12.40 ms/pair |

The attention benchmark used batch size 1, eight heads and 4096 tokens.
Each number is the median of eight CUDA-event measurements of 120 graph
replays, alternating backend order. Both kernels produced identical FP16
output on the tested random input.

The matcher benchmark used PyTorch 2.11.0+cu128, the same packed FP8
checkpoint, compiled execution with CUDA Graphs and 100 pairs sampled evenly
from the ETH Courtyard manifest. Each variant warmed up on 32 pairs for two
rounds before processing the 100 GPU-resident pairs in sequence. Timing
covers the whole sequence with one synchronization at its end. It excludes
feature extraction, host-device transfers, compilation and validation. A
separate pass checked both match-index arrays, image-0 matching scores,
covisibility scores and exit layers. All comparisons were exact for all
100 pairs in both modes. Small timing differences vary with GPU clocks.
