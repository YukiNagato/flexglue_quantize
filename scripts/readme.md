## calibration and evaluation tools install

Use a source checkout to run `scripts/`; they are not included in the installed
wheel. Install the additional dependencies for calibration, image feature
extraction and dataset preparation:

```bash
python -m pip install torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-build-isolation ".[ptq]" "torchao==0.17.0" "torchvision==0.26.0"
```

The `ptq` extra includes torchao, torchvision, Pillow, h5py and huggingface-hub.
The commands pin the torchao/torchvision versions used with PyTorch 2.11.0.
Loading an already exported FP8 checkpoint does not require torchao.

If all dependencies are already managed in an existing environment, install only
the project with `python -m pip install --no-deps --no-build-isolation -e .`.
This leaves installed dependencies unchanged, but is not a complete setup for a
fresh environment.

## Calibration and export

Run tools from a source checkout, either as modules or direct scripts:

```bash
# Existing source indexes/metadata live in dataset_sources/.
# Repeat download commands for all eight dataset names before merging manifests.
python -m scripts.prepare_flexglue_samples mapfree --output data/flexglue_ptq_100
python -m scripts.prepare_multidata_validation mapfree --output data/flexglue_eval_multidata_100

# Audit/merge the eight 100-pair calibration subsets into a shared manifest.
python -m scripts.validate_flexglue_samples --data data/flexglue_ptq_100

# Observe all heads, including intermediate early-exit scores; no early rejection.
python -m scripts.ptq_flexglue --data data/flexglue_ptq_100 \
  --weights weights/flexglue.pth --detector-weights weights/raco.pth \
  --validation-data data/flexglue_eval_100 --output results/calibration

# Export the validated combination; posenc.Wr remains float by default.
python -m scripts.ptq --calibration results/calibration \
  --weights weights/flexglue.pth --output weights/model_fp8.pt
```

Export on CUDA to reproduce the existing CUDA-generated checkpoint exactly.
Recomputing quantization on CPU can introduce rounding differences in scales
and FP8 weights; loading the exported checkpoint does not recompute these values.

The calibration manifest must contain `images.json`, `pairs.jsonl` and
`manifest.json`, using the existing scripts' schema. `--validation-data` is an
optional disjointness check, not data used to fit scales. Downloaders retain the
existing dataset-specific selection logic and source metadata requirement;
`FLEXGLUE_DATA_SOURCES` can point to an external metadata/cache directory.
ScanNet++ calibration needs `--scannetpp-source /path/to/scene/dslr`.
ETH evaluation preparation takes `--source /path/to/eth/mvs_dslr`.
Dataset indexes, images, credentials, and downloaded archives are excluded from
the Python package and Git. This reorganization does not publish a dataset bundle.

## Tests

The complete pair-level evaluation datasets are prepared separately from PTQ data:

```bash
python scripts/download_benchmarks.py
```

This downloads the official MegaDepth-1500, ScanNet-1500 and HardMatch archives to
`data/benchmarks/`, verifies ZIP CRCs, decodes the images/depth maps, and writes
`pairs.jsonl`, `images.json` and `manifest.json` for each dataset. Original source
annotations are retained. The final evaluation selection is recorded in
`configs/benchmark_data.json`: 1500 MegaDepth pairs, 1500 ScanNet pairs, and **all
1000 HardMatch pairs** (900 original test + 100 original validation), with no
subset sampling. HardMatch source split labels remain available. Its upstream
fundamental-matrix eligibility flags are recorded per pair without removing any
pairs from the all-pair manifest. This downloader does not install dependencies
or run model evaluation.

The current HardMatch archive omits image assets for one annotated pair. The
downloader recovers that pair from the Wikimedia URLs in its original metadata,
resizes to a 1024-pixel longest side and scales the original correspondences.
These files are marked `recovered_from_original_sources`, with hashes and the
exact transformation in `recovery.json`; they are not claimed to be byte-identical
to official assets. All 1000 pairs remain in the final manifest. The audit also
records missing GT and fundamental-matrix eligibility, so metric denominators
must be reported separately from the number of pairs passed through inference.

After installing the optional tools and preparing the required weights, run the
complete frozen-weight comparison:

```bash
python -m scripts.evaluate_benchmarks
```

This evaluates original float weights (FP16 inference), packed FP8 with the
default CuTe backends, and that same quantized model with early rejection.
All variants use identical cached RaCo/DINO features and compiled inference.
Results and per-pair predictions go to `results/streaming_benchmarks_20261118/` by
default; `--output` selects another run directory. The script can resume completed
stages in the same directory; use a new directory after changing code or settings.
Each model warms up once on 32 uniformly selected pairs (two rounds), then
processes all 4000 pairs in manifest order with one matcher call per pair.
Primary timing uses CUDA events around matching, after input transfer and before
output transfer. CPU↔GPU transfers, CPU output processing, disk cache loading,
feature extraction, startup warmup, and CPU geometry are excluded. Full stream
wall times remain diagnostic only; `matcher_comparison.json` is the primary
speed summary. No per-pair warmup or repeated timing calls are used.
Pose AUC uses the fixed OpenCV RANSAC protocol
recorded in each report; this is a controlled quantization comparison, not an
exact reproduction of the paper's complete evaluation setup. HardMatch reports
the upstream F metric and an additional version assigning zero to failed F fits.

```bash
python -m unittest discover -s tests -v
python -m scripts.validate_package \
  --checkpoint results/eval100_ptq_best_posenc_float/model_fp8.pt \
  --pairs results/standalone_courtyard \
  --reference results/early_fusion_20261117/interleaved_v2/predictions.pt \
  --output results/package_validation/report.json
```

Weights, original calibration ranges, rejection thresholds, and the numerical
kernel algorithms are retained. Compiled binaries and runtime caches are generated
on the user's machine and are not release artifacts.
