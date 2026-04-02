"""Additional pipeline config tests — covering uncovered branches in pipeline.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from corridorkey.infra.config.inference import InferenceSettings
from corridorkey.infra.config.pipeline import CorridorKeyConfig
from pydantic import ValidationError


class TestDeviceValidatorNonStringInput:
    def test_non_string_device_raises(self):
        """The `not isinstance(v, str)` branch — passing a non-string device."""
        with pytest.raises((ValidationError, ValueError)):
            CorridorKeyConfig(device=42)  # type: ignore[arg-type]

    def test_none_device_raises(self):
        with pytest.raises((ValidationError, ValueError)):
            CorridorKeyConfig(device=None)  # type: ignore[arg-type]


class TestToInferenceConfigPrecisionAuto:
    """Cover the auto precision resolution branches in to_inference_config."""

    def test_auto_precision_cpu_resolves_to_float32(self, tmp_path: Path):
        import torch

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            inference=InferenceSettings(checkpoint_path=p, model_precision="auto"),
        )
        ic = cfg.to_inference_config(device="cpu")
        assert ic.model_precision == torch.float32

    def test_auto_precision_cuda_ampere_resolves_to_bfloat16(self, tmp_path: Path):
        """Ampere+ GPU (major >= 8) should resolve to bfloat16."""
        import torch

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cuda",
            inference=InferenceSettings(checkpoint_path=p, model_precision="auto"),
        )

        mock_props = type("Props", (), {"major": 8, "name": "A100"})()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            ic = cfg.to_inference_config(device="cuda")
        assert ic.model_precision == torch.bfloat16

    def test_auto_precision_cuda_pre_ampere_resolves_to_float16(self, tmp_path: Path):
        """Pre-Ampere GPU (major < 8) should resolve to float16."""
        import torch

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cuda",
            inference=InferenceSettings(checkpoint_path=p, model_precision="auto"),
        )

        mock_props = type("Props", (), {"major": 7, "name": "V100"})()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            ic = cfg.to_inference_config(device="cuda")
        assert ic.model_precision == torch.float16

    def test_auto_precision_cuda_unavailable_resolves_to_float32(self, tmp_path: Path):
        """CUDA device string but torch.cuda.is_available() == False → float32 fallback."""
        import torch

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cuda",
            inference=InferenceSettings(checkpoint_path=p, model_precision="auto"),
        )
        with patch("torch.cuda.is_available", return_value=False):
            ic = cfg.to_inference_config(device="cuda")
        assert ic.model_precision == torch.float32


class TestToInferenceConfigRefinerModeAuto:
    """Cover the refiner_mode='auto' resolution branches."""

    def test_auto_refiner_mode_low_vram_resolves_to_tiled(self, tmp_path: Path):
        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            inference=InferenceSettings(checkpoint_path=p, refiner_mode="auto"),
        )
        with patch("corridorkey.stages.inference.orchestrator._probe_vram_gb", return_value=8.0):
            _, resolved = cfg.to_inference_config(device="cuda", _return_resolved_refiner_mode=True)
        assert resolved == "tiled"

    def test_auto_refiner_mode_high_vram_resolves_to_full_frame(self, tmp_path: Path):
        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            inference=InferenceSettings(checkpoint_path=p, refiner_mode="auto"),
        )
        with patch("corridorkey.stages.inference.orchestrator._probe_vram_gb", return_value=24.0):
            _, resolved = cfg.to_inference_config(device="cuda", _return_resolved_refiner_mode=True)
        assert resolved == "full_frame"

    def test_auto_refiner_mode_zero_vram_resolves_to_full_frame(self, tmp_path: Path):
        """0.0 VRAM (unknown) → full_frame (don't force tiled on unknown hardware)."""
        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            inference=InferenceSettings(checkpoint_path=p, refiner_mode="auto"),
        )
        with patch("corridorkey.stages.inference.orchestrator._probe_vram_gb", return_value=0.0):
            _, resolved = cfg.to_inference_config(device="cuda", _return_resolved_refiner_mode=True)
        assert resolved == "full_frame"
