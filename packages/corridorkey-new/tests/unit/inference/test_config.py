"""Unit tests for corridorkey_new.stages.inference.config."""

from __future__ import annotations

from pathlib import Path

import torch
from corridorkey_new.stages.inference.config import (
    _VRAM_LOWVRAM_THRESHOLD_GB,
    REFINER_TILE_OVERLAP,
    REFINER_TILE_SIZE,
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

    def test_optimization_mode_default(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth")
        assert cfg.optimization_mode == "auto"


class TestInferenceConfigOverrides:
    def test_device_override(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth", device="cuda")
        assert cfg.device == "cuda"

    def test_img_size_override(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth", img_size=512)
        assert cfg.img_size == 512

    def test_optimization_mode_lowvram(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth", optimization_mode="lowvram")
        assert cfg.optimization_mode == "lowvram"

    def test_optimization_mode_speed(self, tmp_path: Path):
        cfg = InferenceConfig(checkpoint_path=tmp_path / "m.pth", optimization_mode="speed")
        assert cfg.optimization_mode == "speed"

    def test_checkpoint_path_stored(self, tmp_path: Path):
        p = tmp_path / "weights.pth"
        cfg = InferenceConfig(checkpoint_path=p)
        assert cfg.checkpoint_path == p


class TestModuleConstants:
    def test_vram_threshold_positive(self):
        assert _VRAM_LOWVRAM_THRESHOLD_GB > 0

    def test_tile_size_positive(self):
        assert REFINER_TILE_SIZE > 0

    def test_tile_overlap_less_than_tile_size(self):
        assert REFINER_TILE_OVERLAP < REFINER_TILE_SIZE

    def test_tile_overlap_positive(self):
        assert REFINER_TILE_OVERLAP > 0
