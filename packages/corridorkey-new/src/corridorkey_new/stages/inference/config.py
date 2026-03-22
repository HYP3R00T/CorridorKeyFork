"""Inference stage — configuration contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

OptimizationMode = Literal["auto", "speed", "lowvram"]

# VRAM threshold below which lowvram mode is selected automatically.
_VRAM_LOWVRAM_THRESHOLD_GB = 12.0

# Tiled refiner defaults (lowvram mode).
REFINER_TILE_SIZE = 512
REFINER_TILE_OVERLAP = 128

# VRAM-adaptive img_size thresholds.
# Below 6 GB: 1024 (4 refiner tiles, fits comfortably in 4 GB)
# Below 12 GB: 1536 (good quality, fits in 6-11 GB)
# 12 GB+: 2048 (native training resolution, full quality)
_VRAM_IMG_SIZE_TIERS: list[tuple[float, int]] = [
    (6.0, 1024),
    (12.0, 1536),
]
_VRAM_IMG_SIZE_DEFAULT = 2048


def adaptive_img_size(vram_gb: float) -> int:
    """Return the recommended img_size for the given VRAM amount.

    Args:
        vram_gb: Total GPU VRAM in GB.

    Returns:
        img_size to use: 1024 for <6 GB, 1536 for 6-12 GB, 2048 for 12+ GB.
    """
    for threshold, size in _VRAM_IMG_SIZE_TIERS:
        if vram_gb < threshold:
            return size
    return _VRAM_IMG_SIZE_DEFAULT


@dataclass
class InferenceConfig:
    """Configuration for the inference stage.

    Attributes:
        checkpoint_path: Path to the .pth model checkpoint file.
        device: PyTorch device string ("cuda", "cuda:0", "mps", "cpu").
        img_size: Square resolution the model runs at. Must match the
            resolution the checkpoint was trained at (default 2048).
        use_refiner: Whether to enable the CNN refiner module.
        mixed_precision: Run the forward pass under autocast.
            Ignored on CPU (autocast is a no-op there).
        model_precision: Weight dtype. float32 is safe everywhere;
            float16/bfloat16 saves VRAM on CUDA but may reduce accuracy.
        optimization_mode: Refiner execution strategy.
            "auto"    — probe VRAM; < 12 GB → lowvram, else → speed.
            "speed"   — full-frame refiner pass.
            "lowvram" — tiled refiner (512×512, 128px overlap).
        refiner_scale: Multiplier applied to the CNN refiner's delta output.
            1.0 applies full refinement. 0.0 skips the refiner output entirely.
            Reducing toward 0.0 speeds up processing at the cost of edge quality.
    """

    checkpoint_path: Path
    device: str = "cpu"
    img_size: int = 2048
    use_refiner: bool = True
    mixed_precision: bool = True
    model_precision: torch.dtype = field(default=torch.float32)
    optimization_mode: OptimizationMode = "auto"
    refiner_scale: float = 1.0
