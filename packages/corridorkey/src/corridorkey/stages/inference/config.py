"""Inference stage — configuration contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch

RefinerMode = Literal["auto", "full_frame", "tiled"]
BackendChoice = Literal["auto", "torch", "mlx"]

# Valid img_size values. 0 means auto-select based on VRAM.
VALID_IMG_SIZES = (0, 512, 1024, 1536, 2048)

# VRAM threshold below which tiled refiner mode is selected automatically.
_VRAM_TILED_THRESHOLD_GB = 12.0

# VRAM threshold below which empty_cache() is called after each frame to
# prevent OOM on the next frame. Separate from the tiling threshold.
_VRAM_FREE_CACHE_THRESHOLD_GB = 6.0

# Tiled refiner defaults.
REFINER_TILE_SIZE = 512
REFINER_TILE_OVERLAP = 128

# VRAM-adaptive img_size thresholds.
# Below 6 GB:  1024 (fits comfortably in 4 GB)
# Below 12 GB: 1536 (good quality, fits in 6–11 GB)
# 12 GB+:      2048 (native training resolution, full quality)
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
        img_size: 1024 for <6 GB, 1536 for 6–12 GB, 2048 for 12+ GB.
    """
    for threshold, size in _VRAM_IMG_SIZE_TIERS:
        if vram_gb < threshold:
            return size
    return _VRAM_IMG_SIZE_DEFAULT


@dataclass
class InferenceConfig:
    """Runtime configuration for the inference stage.

    Attributes:
        checkpoint_path: Path to the .pth model checkpoint file.
        device: PyTorch device string ("cuda", "cuda:0", "mps", "cpu").
        img_size: Square resolution the model runs at. Must be one of
            0, 512, 1024, 1536, or 2048. 0 means auto-select based on VRAM
            (resolved by ``pipeline.to_inference_config()`` before the model
            is loaded — do not pass 0 directly to ``load_model``).
            2048 is the native training resolution and produces the best output.
            Smaller values reduce VRAM usage at the cost of output quality.
        use_refiner: Whether to enable the CNN refiner module. The refiner
            corrects transformer macroblocking artifacts at subject edges.
            Disabling it is faster but produces visibly coarser alpha mattes.
        mixed_precision: Run the forward pass under autocast (fp16/bf16).
            Ignored on CPU. Reduces VRAM usage with minimal quality impact.
        model_precision: Weight dtype for the model forward pass.
            float32 is safe everywhere. float16/bfloat16 saves VRAM on CUDA
            but may reduce numerical stability.
        refiner_mode: Controls how the CNN refiner executes.
            "auto"       — probe VRAM; <12 GB → tiled, else → full_frame.
            "full_frame" — run the refiner on the full image at once.
                           Best performance on GPUs with 12+ GB VRAM.
            "tiled"      — run the refiner in 512×512 overlapping tiles.
                           Keeps peak VRAM flat. Identical output quality
                           to full_frame. Required on low-VRAM GPUs.
        refiner_scale: Multiplier applied to the CNN refiner's delta output.
            1.0 applies full refinement. 0.0 disables the refiner output
            entirely (equivalent to use_refiner=False but without skipping
            the forward pass). Values between 0 and 1 blend between no
            refinement and full refinement.
        backend: Which inference backend to use.
            "auto"  — Apple Silicon + corridorkey-mlx installed → mlx,
                      else → torch.
            "torch" — always use PyTorch (CUDA / ROCm / MPS / CPU).
            "mlx"   — always use MLX (Apple Silicon only, optional package).
            Can also be overridden via the CORRIDORKEY_BACKEND env var.
    """

    checkpoint_path: Path
    device: str = "cpu"
    img_size: int = 2048
    use_refiner: bool = True
    mixed_precision: bool = True
    model_precision: torch.dtype = field(default=torch.float32)
    refiner_mode: RefinerMode = "auto"
    refiner_scale: float = 1.0
    backend: BackendChoice = "auto"

    def __post_init__(self) -> None:
        if self.img_size not in VALID_IMG_SIZES:
            raise ValueError(
                f"img_size must be one of {VALID_IMG_SIZES}, got {self.img_size}. Use 0 for auto-select based on VRAM."
            )
        # mixed_precision is a no-op when model_precision is already float16 —
        # the weights are already in half precision so autocast adds no benefit.
        if self.model_precision == torch.float16:
            self.mixed_precision = False
