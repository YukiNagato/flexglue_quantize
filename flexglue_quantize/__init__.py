"""Editable FlexGlue FP8 inference, independent of torch_sfm and pip flexglue."""
from .descriptor import DINOv3Descriptor
from .model import FlexGlue, load_best_ptq

__version__ = "0.1.0"
__all__ = ["FlexGlue", "DINOv3Descriptor", "load_best_ptq"]
