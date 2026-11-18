"""FlexGlue-compatible inference interface for packed, calibrated FP8 weights."""
from pathlib import Path

import torch
from torch import nn

from .core import FlexGlue as BaseFlexGlue
from .linear import FP8Linear


class FlexGlue(BaseFlexGlue):
    """Load a packed PTQ checkpoint and retain FlexGlue's matching API.

    Inputs and outputs of ``forward``, ``match`` and ``match_dense`` are unchanged.
    ``early_stop_decay=None`` runs all configured layers. A decay in [0, 1)
    enables early rejection and requires positive ``min_covisibility``.
    Compile explicitly with ``model.compile()`` after moving to the GPU.
    This class performs inference only; calibration/export live in scripts.
    """

    def __init__(self, *, weights, attention_backend="cute", cute_preprocess=None,
                 head_backend="cute-precise", early_exit_backend="cute", device=None,
                 descriptor_weights=None, **conf):
        if cute_preprocess is None:
            cute_preprocess = "qkv" if attention_backend in ("cute", "warp") else "none"
        if early_exit_backend not in ("none", "compiled", "cute"):
            raise ValueError(f"Unknown early-exit backend: {early_exit_backend}")
        if unknown := conf.keys() - self.default_conf.keys():
            raise KeyError(f"Unknown conf: {sorted(unknown)}")
        packed = torch.load(Path(weights), map_location="cpu", weights_only=True)
        if not isinstance(packed, dict) or packed.get("format") != "flexglue-best-ptq-v1":
            raise ValueError("Expected a flexglue-best-ptq-v1 checkpoint; export it with scripts.ptq.")
        config = dict(packed["conf"])
        # Early exit is an explicit run-time choice, never silently inherited.
        config["early_stop_decay"] = None
        for key in ("input_dim", "descriptor_dim", "search_dim", "num_heads", "n_layers"):
            if key in conf and conf[key] != config[key]:
                raise ValueError(f"{key} must match checkpoint ({config[key]}). Use per-call n_layers to truncate.")
        config.update(conf)
        describe, refine = config.get("describe", False), config.get("refine", False)
        has_descriptor = any(k.startswith("descriptor.") for k in packed["state_dict"])
        has_refiner = any(k.startswith("refiner.") for k in packed["state_dict"])
        if has_descriptor or has_refiner:
            raise ValueError("This checkpoint format supports matcher weights only; attach descriptor/refiner separately.")
        if describe and descriptor_weights is None:
            raise ValueError("describe=True requires descriptor_weights (the original float checkpoint).")
        super().__init__(**{**config, "weights": None, "describe": False, "refine": False})
        targets = {r["name"] for r in packed["ptq"]["linear_coverage"] if r["quantized"]}
        linears = {n: m for n, m in self.named_modules() if isinstance(m, nn.Linear)}
        if targets - linears.keys():
            raise ValueError(f"Unknown quantized modules: {sorted(targets - linears.keys())}")
        for name in sorted(targets):
            parent, _, child = name.rpartition(".")
            setattr(self.get_submodule(parent), child, FP8Linear(linears[name], torch.ones(())))
        self.load_state_dict(packed["state_dict"], strict=True)
        self.ptq_info = packed["ptq"]
        self.checkpoint_path = str(Path(weights).resolve())
        if describe:
            from .descriptor import DINOv3Descriptor
            self.descriptor = DINOv3Descriptor(weights=descriptor_weights)
        if refine:
            from .refiner import RoMav2Refiner
            self.refiner = RoMav2Refiner(self.conf.refine_scales)
        self.conf.describe, self.conf.refine = describe, refine
        self.resolve_conf({})
        if device is not None:
            self.to(device)
        from .attention import install as install_attention
        from .head import install as install_head
        install_attention(self, attention_backend, cute_preprocess)
        install_head(self, head_backend)
        self.attention_backend = attention_backend
        self.cute_preprocess = cute_preprocess
        self.head_backend = head_backend
        self.early_exit_backend = early_exit_backend
        if early_exit_backend != "none":
            from .early_exit import install
            install(self, early_exit_backend)
        # In eager mode CuTe receives parameters such as the head temperature
        # directly. DLPack rejects requires_grad tensors even under no_grad.
        # This package is inference-only; freeze the optional float modules too.
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_pretrained(cls, checkpoint, **kwargs):
        """Load an already downloaded checkpoint (no implicit network access)."""
        return cls(weights=checkpoint, **kwargs)


def load_best_ptq(checkpoint, device="cuda", compile=True, early_stop_decay=None,
                  cute_preprocess=None, head_backend="cute-precise", early_exit_backend="cute",
                  **conf):
    """Compatibility loader; new code may construct FlexGlue(weights=...) directly."""
    model = FlexGlue(weights=checkpoint, device=device, early_stop_decay=early_stop_decay,
                     cute_preprocess=cute_preprocess, head_backend=head_backend,
                     early_exit_backend=early_exit_backend, **conf)
    if compile:
        model.compile(mode="reduce-overhead")
    return model
