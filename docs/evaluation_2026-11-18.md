# FlexGlue Quantization and Early-Exit Evaluation

After one startup warmup and one matching call per pair, the frozen quantized
pipeline achieved **1.65× overall matcher speedup** across all 4,000 pairs:
**23.98 → 14.49 ms per pair**, measured with CUDA events and excluding input and
output transfers. Per-dataset speedups were 1.65–1.66×. Full-depth pose AUC and
standard HardMatch mAA changed by less than 0.42 percentage points at the
reported thresholds.

Quantized early exit averaged 15.11 ms per pair. It was slower on MegaDepth and
ScanNet, and reduced HardMatch matcher latency by 1.63%. All sequential-run
predictions, exit layers, rejection decisions, and covisibility scores exactly
matched the earlier evaluation, so the GT accuracy results remain unchanged.

This is a controlled comparison of our inference implementations against dataset
ground truth, not an exact reproduction of the FlexGlue paper's evaluation.
The speedup includes quantized GEMMs, SageAttention, and CuTe fusion; it does not
isolate the benefit of torchao quantization alone.

## Models and environment

| Variant | Weights and attention | Early exit | Measurements |
|---|---|---|---|
| `float_full` | Original float weights, FP16 inference, PyTorch SDPA | Disabled | GT metrics and latency |
| `quant_full` | Frozen static FP8 linears, SageAttention 2, CuTe preprocessing and matching head | Disabled | GT metrics and latency |
| `quant_early` | Same quantized checkpoint and backends, plus CuTe early-score fusion | Enabled | GT metrics, latency, and exit decisions |
| Float early-exit follow-up | Same original float weights and SDPA as `float_full` | Enabled | Exit decisions and output consistency; no new latency or aggregate GT-metric benchmark |

The quantized pipeline uses INT8 Q/K and FP8 P/V attention computation with
SageAttention. The positional projection remains floating point. This is mixed
precision inference, not an all-INT8 or all-FP8 model. All four variants use
`torch.compile(mode="reduce-overhead")`; the quantized options are
`cute_preprocess="qkv"`, `head_backend="cute-precise"`, and, for early exit,
`early_exit_backend="cute"`.

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D |
| Python environment | `/root/autodl-tmp/conda_env/sfm` |
| PyTorch / CUDA build | `2.11.0+cu128` / CUDA 12.8 |
| NumPy | `2.5.2` |
| OpenCV | `5.0.0` |
| CuTe DSL | `4.8.1` |
| PyTorch CPU threads | 4 |
| Float32 matmul precision / matmul TF32 | `highest` / disabled |
| cuDNN TF32 | Enabled |
| Detector | RaCo, up to 4,096 keypoints, detection resize 1,024 |
| Dense descriptor | Original floating DINOv3 descriptor, resize 512 × 512 |
| Transformer depth | 9 blocks |
| Minimum covisibility | 0.9 |
| Match confidence threshold | 0.1 |
| Maximum matching distance | 3 px |
| Early-stop decay | `None` for full depth; `0.5` for early exit |
| Pair batch size | 1 |

All variants share exactly the same cached detector/descriptor outputs. Weights,
activation scales, and thresholds were frozen throughout evaluation; there was
no recalibration or training on these benchmark pairs during this run. Core
packages and model weights were unchanged.

## Datasets and metric denominators

| Dataset | Pairs passed through each model | Primary GT evaluation |
|---|---:|---|
| MegaDepth-1500 | 1,500 | Relative pose AUC over all 1,500 pairs |
| ScanNet-1500 | 1,500 | Relative pose AUC over all 1,500 pairs |
| HardMatch | 1,000 | Fundamental-matrix mAA over 904 eligible pairs |

## Full-depth accuracy

All values are percentages. Changes are **percentage points (pp)**, quantized
minus float. The baseline uses original weights with FP16 inference, not FP32
inference. These comparisons use GT rather than treating float predictions as GT.

| Dataset | Metric | Float full | Quantized full | Change (pp) |
|---|---|---:|---:|---:|
| MegaDepth-1500 | AUC@5° | 42.85 | 43.22 | +0.37 |
| MegaDepth-1500 | AUC@10° | 61.24 | 61.11 | −0.14 |
| MegaDepth-1500 | AUC@20° | 75.91 | 75.58 | −0.34 |
| ScanNet-1500 | AUC@5° | 31.79 | 31.78 | −0.01 |
| ScanNet-1500 | AUC@10° | 54.46 | 54.38 | −0.08 |
| ScanNet-1500 | AUC@20° | 71.99 | 72.05 | +0.07 |
| HardMatch | mAA@5px | 29.91 | 29.74 | −0.18 |
| HardMatch | mAA@10px | 40.47 | 40.06 | −0.41 |
| HardMatch | mAA@10px, failed F = zero | 39.68 | 39.26 | −0.42 |
| HardMatch | mAA@10px, official assets only (903 eligible pairs) | 40.43 | 40.02 | −0.41 |

Sampson precision at 3 px was 96.166% → 96.165% on MegaDepth and
48.666% → 48.675% on ScanNet.

## Matching speed and early-exit accuracy

Primary latencies are mean CUDA-event matcher durations in milliseconds per
pair, excluding IO and CPU output processing.

| Dataset | Float full | Quantized full | Full-depth speedup | Quantized early exit |
|---|---:|---:|---:|---:|
| MegaDepth-1500 | 23.92 | 14.47 | 1.65× | 15.32 |
| ScanNet-1500 | 24.00 | 14.50 | 1.66× | 15.47 |
| HardMatch | 24.02 | 14.51 | 1.66× | 14.27 |
| All 4,000 pairs | 23.98 | 14.49 | 1.65× | 15.11 |

Accuracy with early exit remains:

| Dataset | Metric | Quantized full | Quantized early exit |
|---|---|---:|---:|
| MegaDepth-1500 | AUC@5° / 10° / 20° | 43.22 / 61.11 / 75.58 | 43.08 / 60.94 / 75.39 |
| ScanNet-1500 | AUC@5° / 10° / 20° | 31.78 / 54.38 / 72.05 | 31.78 / 54.38 / 72.05 |
| HardMatch | mAA@10px | 40.06 | 40.06 |

Final covisibility rejection counts, which are distinct from early-exit counts:

| Dataset | Float full | Quantized full | Quantized early exit |
|---|---:|---:|---:|
| MegaDepth-1500 | 13 | 14 | 17 |
| ScanNet-1500 | 8 | 10 | 10 |
| HardMatch | 405 | 409 | 410 |

## Early-exit decisions: original float versus quantized

The initial float baseline had early exit disabled. A separate follow-up enabled
it on the original float model and evaluated all 4,000 pairs using the same
features and thresholds. Every float pair that reached block 9 produced exactly
the same match indices and final covisibility score as its saved float-full
prediction.

An early exit means `stop_layer < 9`. At block `k` (1–8), the rejection threshold
is `0.9 * 0.5 ** (9 - k)`. An actual early exit returns empty matches and skips
the later matching-head computation. The final covisibility threshold remains
0.9 for pairs that reach block 9.

| Dataset | Float exits | Quantized exits | Trigger agreement | Exact exit-layer agreement |
|---|---:|---:|---:|---:|
| MegaDepth-1500 | 8 / 1,500 (0.53%) | 8 / 1,500 (0.53%) | 100% | 99.93% |
| ScanNet-1500 | 0 / 1,500 | 0 / 1,500 | 100% | 100% |
| HardMatch | 265 / 1,000 (26.5%) | 280 / 1,000 (28.0%) | 97.7% | 95.8% |
| All pairs | 273 / 4,000 | 288 / 4,000 | 99.425% | 98.925% |

MegaDepth has exactly the same eight exiting pairs; one exits at block 6 after
quantization instead of block 7. HardMatch has 261 pairs that exit in both
models, 19 that exit only in the quantized model, and four that exit only in the
float model. Among the 261 common exits, 242 exit at the same block. Of the 19
quantized-only exits, 17 occur at block 8, one at block 7, and one at block 5.

HardMatch's full exit-layer distribution is:

| Executed blocks | Float pairs | Quantized pairs |
|---|---:|---:|
| 1 | 0 | 0 |
| 2 | 2 | 2 |
| 3 | 14 | 12 |
| 4 | 16 | 20 |
| 5 | 30 | 29 |
| 6 | 36 | 38 |
| 7 | 58 | 57 |
| 8 | 109 | 122 |
| 9 (no early exit) | 735 | 720 |

Mean executed blocks are 8.976 → 8.9753 on MegaDepth, 9 → 9 on ScanNet, and
8.369 → 8.348 on HardMatch. Most additional HardMatch exits are late, so the
increase in exit frequency saves relatively little computation. These are
measured block counts, not estimates of end-to-end speedup.
