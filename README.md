# flexglue_quantize

**Unoffical** [FlexGlue](https://github.com/cvg/flexglue) FP8 inference with SageAttention, CuTe fusion and
early rejection.
The quantized pipeline achieved **1.65× overall matcher speedup**.
See the [full benchmark and early-exit evaluation report](docs/evaluation_2026-11-18.md)
for measured accuracy, matching speed, and float-versus-quantized exit behavior.

## Installation
The tested platform is Linux x86_64, Python 3.12, PyTorch 2.11.0 with CUDA 12.8,
and an NVIDIA RTX 4090 D (Ada, SM89). FP8 inference requires a supported NVIDIA
GPU; CPU inference is not supported. The package-owned CuTe and Warp attention backends require compute capability
8.9 or higher and a toolchain that supports the GPU. Runtime correctness and
performance have been validated on RTX 4090 D.

### Install this package

```bash
pip install .
```
This installs the default CuTe backend.

### Verify the installation

```bash
python -m pip check
python - <<'PY'
import torch
import flexglue_quantize
import cutlass.cute
import cuda.bindings.driver

assert torch.cuda.is_available()
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
print("Package:", flexglue_quantize.__file__)
print("Accelerated inference dependencies imported successfully.")
PY
```
## Inference

By default, `FlexGlue()` downloads the [v0.1 FP8 matcher weights](https://github.com/YukiNagato/flexglue_quantize/releases/download/v0.1/model_fp8.pt) on first use. For offline use with a local checkpoint, pass `FlexGlue(weights="/path/to/model_fp8.pt")`.

```python
import torch
from flexglue_quantize import FlexGlue

model = FlexGlue(
    min_covisibility=0.9,
    early_stop_decay=None,       # full nine-layer inference
).eval().cuda()
model.compile(mode="reduce-overhead")

# Each input follows the original FlexGlue dictionary contract:
# features: (B, 256, H, W), keypoints: (B, N, 2) in original-image pixels,
# image_size: (B, 2), ordered (width, height). Tensors must be on the GPU.
with torch.inference_mode():
    full = model.match(data0, data1, sparse=True)
    early = model.match(data0, data1, sparse=True, early_stop_decay=0.5)
    # Equivalent forward interface:
    early = model({"image0": data0, "image1": data1},
                  sparse=True, early_stop_decay=0.5)
```

## How to quantize
See [how to quantize](scripts/readme.md)