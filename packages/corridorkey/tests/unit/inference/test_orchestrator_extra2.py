"""Cover remaining orchestrator lines: refiner=None scale path, pynvml probe, MPS branch."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from corridorkey.stages.inference.config import InferenceConfig
from corridorkey.stages.inference.orchestrator import (
    _probe_vram_gb,
    _should_tile_refiner,
    run_inference,
)
from corridorkey.stages.preprocessor.contracts import FrameMeta, PreprocessedFrame


def _make_config(tmp_path: Path, **kwargs) -> InferenceConfig:
    defaults = {"checkpoint_path": tmp_path / "m.pth", "device": "cpu"}
    defaults.update(kwargs)
    return InferenceConfig(**defaults)  # ty:ignore[invalid-argument-type]


def _make_frame() -> PreprocessedFrame:
    meta = FrameMeta(frame_index=0, original_h=32, original_w=32)
    return PreprocessedFrame(tensor=torch.zeros(1, 4, 32, 32), meta=meta)


def _make_model_output() -> dict:
    return {"alpha": torch.zeros(1, 1, 32, 32), "fg": torch.zeros(1, 3, 32, 32)}


class TestRefinerScaleHookNoRefinerAttr:
    """Line 84: the `if refiner is not None` guard — model has no .refiner attribute."""

    def test_scale_hook_skipped_when_model_has_no_refiner(self, tmp_path: Path):
        """When model has no .refiner, the scale hook block is a no-op."""
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="full_frame", refiner_scale=0.5)
        frame = _make_frame()
        model = MagicMock(spec=[])  # no attributes at all
        model.return_value = _make_model_output()
        # Should not raise even though model has no .refiner
        result = run_inference(frame, model, cfg)
        assert result is not None


class TestProbeVramGbPynvml:
    """Lines 203-210: the pynvml success path in _probe_vram_gb."""

    def test_pynvml_success_path_returns_gb(self):
        """When pynvml succeeds, returns total memory in GB."""
        mock_mem = MagicMock()
        mock_mem.total = 8 * 1024**3  # 8 GB

        with (
            patch("pynvml.nvmlInit"),
            patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=MagicMock()),
            patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_mem),
        ):
            result = _probe_vram_gb("cuda")

        assert result == pytest.approx(8.0, abs=0.1)

    def test_pynvml_failure_falls_back_to_torch(self):
        """When pynvml raises, falls back to torch.cuda.get_device_properties."""
        mock_props = MagicMock()
        mock_props.total_memory = 6 * 1024**3  # 6 GB

        with (
            patch("pynvml.nvmlInit", side_effect=Exception("no nvml")),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            result = _probe_vram_gb("cuda")

        assert result == pytest.approx(6.0, abs=0.1)

    def test_cuda_index_parsed_correctly(self):
        """cuda:1 should probe device index 1."""
        mock_mem = MagicMock()
        mock_mem.total = 10 * 1024**3

        captured_index = []
        with (
            patch("pynvml.nvmlInit"),
            patch("pynvml.nvmlDeviceGetHandleByIndex", side_effect=lambda i: captured_index.append(i) or MagicMock()),
            patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_mem),
        ):
            _probe_vram_gb("cuda:1")

        assert captured_index == [1]


class TestShouldTileRefinerMPS:
    """Line 164: the MPS branch in _should_tile_refiner."""

    def test_mps_device_always_tiles(self, tmp_path: Path):
        """MPS device must always use tiled refiner (no resolved_refiner_mode)."""
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="auto", device="mps")
        # Patch torch.device to return type="mps"
        with patch("corridorkey.stages.inference.orchestrator.torch") as mock_torch:
            mock_torch.device.return_value = MagicMock(type="mps")
            result = _should_tile_refiner(cfg, resolved_refiner_mode=None)
        assert result is True
