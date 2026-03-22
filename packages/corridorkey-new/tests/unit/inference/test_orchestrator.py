"""Unit tests for corridorkey_new.stages.inference.orchestrator.

All tests mock the model — no checkpoint or GPU required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from corridorkey_new.stages.inference.config import InferenceConfig
from corridorkey_new.stages.inference.contracts import InferenceResult
from corridorkey_new.stages.inference.orchestrator import (
    _probe_vram_gb,
    _should_tile_refiner,
    run_inference,
)
from corridorkey_new.stages.preprocessor.contracts import FrameMeta, PreprocessedFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, **kwargs) -> InferenceConfig:
    return InferenceConfig(checkpoint_path=tmp_path / "m.pth", device="cpu", **kwargs)


def _make_frame(h: int = 32, w: int = 32) -> PreprocessedFrame:
    meta = FrameMeta(frame_index=0, original_h=h, original_w=w)
    tensor = torch.zeros(1, 4, h, w)
    return PreprocessedFrame(tensor=tensor, meta=meta)


def _make_model_output(h: int = 32, w: int = 32) -> dict:
    return {
        "alpha": torch.zeros(1, 1, h, w),
        "fg": torch.zeros(1, 3, h, w),
    }


# ---------------------------------------------------------------------------
# _should_tile_refiner
# ---------------------------------------------------------------------------


class TestShouldTileRefiner:
    def test_false_when_refiner_disabled(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=False)
        assert _should_tile_refiner(cfg) is False

    def test_true_when_lowvram(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="tiled")
        assert _should_tile_refiner(cfg) is True

    def test_false_when_speed(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="full_frame")
        assert _should_tile_refiner(cfg) is False

    def test_auto_low_vram_returns_true(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="auto")
        with patch("corridorkey_new.stages.inference.orchestrator._probe_vram_gb", return_value=8.0):
            assert _should_tile_refiner(cfg) is True

    def test_auto_high_vram_returns_false(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="auto")
        with patch("corridorkey_new.stages.inference.orchestrator._probe_vram_gb", return_value=24.0):
            assert _should_tile_refiner(cfg) is False

    def test_auto_zero_vram_returns_false(self, tmp_path: Path):
        """0.0 means VRAM unknown — should not trigger tiled mode."""
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="auto")
        with patch("corridorkey_new.stages.inference.orchestrator._probe_vram_gb", return_value=0.0):
            assert _should_tile_refiner(cfg) is False


# ---------------------------------------------------------------------------
# _probe_vram_gb
# ---------------------------------------------------------------------------


class TestProbeVramGb:
    def test_returns_float(self):
        result = _probe_vram_gb("cpu")
        assert isinstance(result, float)

    def test_returns_zero_on_cpu(self):
        # On CPU with no NVIDIA GPU, pynvml and torch.cuda both fail → 0.0
        result = _probe_vram_gb("cpu")
        assert result >= 0.0


# ---------------------------------------------------------------------------
# run_inference
# ---------------------------------------------------------------------------


class TestRunInference:
    def test_returns_inference_result(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=False)
        frame = _make_frame()
        model = MagicMock(return_value=_make_model_output())

        result = run_inference(frame, model, cfg)

        assert isinstance(result, InferenceResult)

    def test_meta_preserved(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=False)
        frame = _make_frame()
        model = MagicMock(return_value=_make_model_output())

        result = run_inference(frame, model, cfg)

        assert result.meta is frame.meta

    def test_alpha_shape(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=False)
        frame = _make_frame(32, 32)
        model = MagicMock(return_value=_make_model_output(32, 32))

        result = run_inference(frame, model, cfg)

        assert result.alpha.shape == (1, 1, 32, 32)

    def test_fg_shape(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=False)
        frame = _make_frame(32, 32)
        model = MagicMock(return_value=_make_model_output(32, 32))

        result = run_inference(frame, model, cfg)

        assert result.fg.shape == (1, 3, 32, 32)

    def test_model_called_with_tensor(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=False)
        frame = _make_frame()
        model = MagicMock(return_value=_make_model_output())

        run_inference(frame, model, cfg)

        model.assert_called_once_with(frame.tensor)

    def test_no_refiner_hook_when_refiner_disabled(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=False)
        frame = _make_frame()
        model = MagicMock(return_value=_make_model_output())
        model.refiner = MagicMock()

        run_inference(frame, model, cfg)

        # register_forward_hook should NOT have been called
        model.refiner.register_forward_hook.assert_not_called()

    def test_refiner_hook_registered_in_lowvram(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="tiled")
        frame = _make_frame()
        model = MagicMock(return_value=_make_model_output())
        handle = MagicMock()
        model.refiner.register_forward_hook.return_value = handle

        run_inference(frame, model, cfg)

        model.refiner.register_forward_hook.assert_called_once()
        handle.remove.assert_called_once()
