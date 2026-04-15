"""Unit tests for corridorkey.stages.inference.config."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from corridorkey.stages.inference.config import (
    _VRAM_TILED_THRESHOLD_GB,
    REFINER_TILE_OVERLAP,
    REFINER_TILE_SIZE,
    VALID_IMG_SIZES,
    InferenceConfig,
)


class TestInferenceConfigDefaults:
    def test_device_default(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth")
        assert cfg.device == "cpu"

    def test_img_size_default(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth")
        assert cfg.img_size == 2048

    def test_use_refiner_default(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth")
        assert cfg.use_refiner is True

    def test_mixed_precision_default(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth")
        assert cfg.mixed_precision is True

    def test_model_precision_default(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth")
        assert cfg.model_precision == torch.float32

    def test_refiner_mode_default(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth")
        assert cfg.refiner_mode == "auto"


class TestInferenceConfigOverrides:
    def test_device_override(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth", device="cuda")
        assert cfg.device == "cuda"

    def test_img_size_override(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth", img_size=512)
        assert cfg.img_size == 512

    def test_refiner_mode_tiled(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth", refiner_mode="tiled")
        assert cfg.refiner_mode == "tiled"

    def test_refiner_mode_full_frame(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth", refiner_mode="full_frame")
        assert cfg.refiner_mode == "full_frame"

    def test_checkpoint_path_stored(self, tmp_path: Path):
        p = tmp_path / "weights.pth"
        cfg = InferenceConfig(checkpoint_path=p)
        assert cfg.checkpoint_path == p


class TestInferenceConfigValidation:
    @pytest.mark.parametrize("size", [0, 512, 1024, 1536, 2048])
    def test_valid_img_sizes_accepted(self, tmp_path: Path, size: int):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth", img_size=size)
        assert cfg.img_size == size

    @pytest.mark.parametrize("size", [256, 768, 1000, 1920, 4096, -1])
    def test_invalid_img_size_raises(self, tmp_path: Path, size: int):
        with pytest.raises(ValueError, match="img_size must be one of"):
            InferenceConfig(checkpoint_path=tmp_path / "m.pth", img_size=size)

    def test_valid_img_sizes_constant(self):
        assert set(VALID_IMG_SIZES) == {0, 512, 1024, 1536, 2048}

    def test_mixed_precision_cleared_when_model_precision_float16(self, tmp_path: Path):
        """mixed_precision=True is a no-op with float16 weights — should be silently cleared."""
        cfg = InferenceConfig(
            checkpoint_path=tmp_path / "m.pth",
            mixed_precision=True,
            model_precision=torch.float16,
        )
        assert cfg.mixed_precision is False

    def test_mixed_precision_kept_when_model_precision_bfloat16(self, tmp_path: Path):
        """mixed_precision=True with bfloat16 is valid — autocast still applies."""
        cfg = InferenceConfig(
            checkpoint_path=tmp_path / "m.pth",
            mixed_precision=True,
            model_precision=torch.bfloat16,
        )
        assert cfg.mixed_precision is True

    def test_mixed_precision_kept_when_model_precision_float32(self, tmp_path: Path):
        cfg = InferenceConfig(
            checkpoint_path=tmp_path / "m.pth",
            mixed_precision=True,
            model_precision=torch.float32,
        )
        assert cfg.mixed_precision is True


class TestModuleConstants:
    def test_vram_threshold_positive(self):
        assert _VRAM_TILED_THRESHOLD_GB > 0

    def test_tile_size_positive(self):
        assert REFINER_TILE_SIZE > 0

    def test_tile_overlap_less_than_tile_size(self):
        assert REFINER_TILE_OVERLAP < REFINER_TILE_SIZE

    def test_tile_overlap_positive(self):
        assert REFINER_TILE_OVERLAP > 0


class TestAdaptiveImgSize:
    def test_returns_default_for_high_vram(self):
        from corridorkey.stages.inference.config import _VRAM_IMG_SIZE_DEFAULT, adaptive_img_size

        assert adaptive_img_size(16.0) == _VRAM_IMG_SIZE_DEFAULT

    def test_returns_1024_for_low_vram(self):
        from corridorkey.stages.inference.config import adaptive_img_size

        assert adaptive_img_size(4.0) == 1024

    def test_returns_1536_for_mid_vram(self):
        from corridorkey.stages.inference.config import adaptive_img_size

        assert adaptive_img_size(8.0) == 1536

    def test_boundary_exactly_6gb_returns_1536(self):
        """At exactly 6.0 GB, the first tier threshold is not met — returns 1536."""
        from corridorkey.stages.inference.config import adaptive_img_size

        assert adaptive_img_size(6.0) == 1536

    def test_boundary_exactly_12gb_returns_default(self):
        """At exactly 12.0 GB, neither tier threshold is met — returns the default 2048."""
        from corridorkey.stages.inference.config import _VRAM_IMG_SIZE_DEFAULT, adaptive_img_size

        assert adaptive_img_size(12.0) == _VRAM_IMG_SIZE_DEFAULT

    def test_zero_vram_returns_1024(self):
        """0.0 GB (unknown VRAM) falls below the first threshold — returns 1024."""
        from corridorkey.stages.inference.config import adaptive_img_size

        assert adaptive_img_size(0.0) == 1024
