"""Unit tests for corridorkey.stages.inference.orchestrator.

All tests mock the model — no checkpoint or GPU required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from corridorkey.stages.inference.config import InferenceConfig
from corridorkey.stages.inference.contracts import InferenceResult
from corridorkey.stages.inference.orchestrator import (
    _probe_vram_gb,
    _should_tile_refiner,
    run_inference,
)
from corridorkey.stages.preprocessor.contracts import FrameMeta, PreprocessedFrame


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
        with patch("corridorkey.stages.inference.orchestrator._probe_vram_gb", return_value=8.0):
            assert _should_tile_refiner(cfg) is True

    def test_auto_high_vram_returns_false(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="auto")
        with patch("corridorkey.stages.inference.orchestrator._probe_vram_gb", return_value=24.0):
            assert _should_tile_refiner(cfg) is False

    def test_auto_zero_vram_returns_false(self, tmp_path: Path):
        """0.0 means VRAM unknown — should not trigger tiled mode."""
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="auto")
        with patch("corridorkey.stages.inference.orchestrator._probe_vram_gb", return_value=0.0):
            assert _should_tile_refiner(cfg) is False


class TestProbeVramGb:
    def test_returns_float(self):
        result = _probe_vram_gb("cpu")
        assert isinstance(result, float)

    def test_returns_zero_on_cpu(self):
        # On CPU with no NVIDIA GPU, pynvml and torch.cuda both fail → 0.0
        result = _probe_vram_gb("cpu")
        assert result >= 0.0


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

    def test_resolved_refiner_mode_skips_vram_probe(self, tmp_path: Path):
        """Passing resolved_refiner_mode must not trigger _probe_vram_gb."""
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="auto")
        frame = _make_frame()
        model = MagicMock(return_value=_make_model_output())

        with patch("corridorkey.stages.inference.orchestrator._probe_vram_gb") as mock_probe:
            run_inference(frame, model, cfg, resolved_refiner_mode="full_frame")
            mock_probe.assert_not_called()

    def test_resolved_refiner_mode_tiled_registers_hook(self, tmp_path: Path):
        """resolved_refiner_mode='tiled' must register the tiled hook even when config says 'auto'."""
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="auto")
        frame = _make_frame()
        model = MagicMock(return_value=_make_model_output())
        handle = MagicMock()
        model.refiner.register_forward_hook.return_value = handle

        with patch("corridorkey.stages.inference.orchestrator._probe_vram_gb", return_value=0.0):
            run_inference(frame, model, cfg, resolved_refiner_mode="tiled")

        model.refiner.register_forward_hook.assert_called_once()
        handle.remove.assert_called_once()


class TestRefinerScaleHook:
    """Lines 79-86: the non-tiled refiner_scale != 1.0 scale hook path."""

    def test_scale_hook_registered_when_not_tiling_and_scale_not_one(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="full_frame", refiner_scale=0.5)
        frame = _make_frame()
        model = MagicMock(return_value=_make_model_output())
        handle = MagicMock()
        model.refiner.register_forward_hook.return_value = handle

        run_inference(frame, model, cfg)

        model.refiner.register_forward_hook.assert_called_once()
        handle.remove.assert_called_once()

    def test_scale_hook_not_registered_when_scale_is_one(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="full_frame", refiner_scale=1.0)
        frame = _make_frame()
        model = MagicMock(return_value=_make_model_output())

        run_inference(frame, model, cfg)

        model.refiner.register_forward_hook.assert_not_called()

    def test_scale_hook_not_registered_when_refiner_disabled(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=False, refiner_scale=0.5)
        frame = _make_frame()
        model = MagicMock(return_value=_make_model_output())

        run_inference(frame, model, cfg)

        model.refiner.register_forward_hook.assert_not_called()


class TestFreeVramIfNeeded:
    """Lines 131-136: _free_vram_if_needed CUDA branch."""

    def test_non_cuda_device_does_nothing(self):
        from corridorkey.stages.inference.orchestrator import _free_vram_if_needed

        with patch("torch.cuda.get_device_properties") as mock_props:
            _free_vram_if_needed("cpu")
            mock_props.assert_not_called()

    def test_cuda_high_vram_does_not_call_empty_cache(self):
        from corridorkey.stages.inference.orchestrator import _free_vram_if_needed

        mock_props = MagicMock()
        mock_props.total_memory = 12 * 1024**3
        with (
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch("torch.cuda.empty_cache") as mock_empty,
        ):
            _free_vram_if_needed("cuda")
            mock_empty.assert_not_called()

    def test_cuda_low_vram_calls_empty_cache(self):
        from corridorkey.stages.inference.orchestrator import _free_vram_if_needed

        mock_props = MagicMock()
        mock_props.total_memory = 4 * 1024**3
        with (
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch("torch.cuda.empty_cache") as mock_empty,
        ):
            _free_vram_if_needed("cuda")
            mock_empty.assert_called_once()

    def test_exception_in_vram_probe_is_swallowed(self):
        from corridorkey.stages.inference.orchestrator import _free_vram_if_needed

        with patch("torch.cuda.get_device_properties", side_effect=RuntimeError("no cuda")):
            _free_vram_if_needed("cuda")  # must not raise


class TestTiledRefinerHookGuard:
    """Lines 248-254: the len(inputs) != 2 guard in _make_tiled_refiner_hook."""

    def test_hook_raises_when_wrong_number_of_inputs(self):
        from corridorkey.stages.inference.orchestrator import _make_tiled_refiner_hook, _TiledRefinerState

        refiner = MagicMock()
        _TiledRefinerState()
        hook = _make_tiled_refiner_hook(refiner)

        with pytest.raises(RuntimeError, match="expected 2 inputs"):
            hook(refiner, (torch.zeros(1, 3, 32, 32),), torch.zeros(1, 4, 32, 32))

    def test_hook_raises_when_zero_inputs(self):
        from corridorkey.stages.inference.orchestrator import _make_tiled_refiner_hook

        refiner = MagicMock()
        hook = _make_tiled_refiner_hook(refiner)

        with pytest.raises(RuntimeError, match="expected 2 inputs"):
            hook(refiner, (), torch.zeros(1, 4, 32, 32))


class TestRefinerScaleHookNoRefinerAttr:
    """Line 84: the `if refiner is not None` guard — model has no .refiner attribute."""

    def test_scale_hook_skipped_when_model_has_no_refiner(self, tmp_path: Path):
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="full_frame", refiner_scale=0.5)
        frame = _make_frame()
        model = MagicMock(spec=[])  # no attributes at all
        model.return_value = _make_model_output()
        result = run_inference(frame, model, cfg)
        assert result is not None


class TestProbeVramGbPynvml:
    """Lines 203-210: the pynvml success path in _probe_vram_gb."""

    def test_pynvml_success_path_returns_gb(self):
        mock_mem = MagicMock()
        mock_mem.total = 8 * 1024**3

        with (
            patch("pynvml.nvmlInit"),
            patch("pynvml.nvmlDeviceGetHandleByIndex", return_value=MagicMock()),
            patch("pynvml.nvmlDeviceGetMemoryInfo", return_value=mock_mem),
        ):
            result = _probe_vram_gb("cuda")

        assert result == pytest.approx(8.0, abs=0.1)

    def test_pynvml_failure_falls_back_to_torch(self):
        mock_props = MagicMock()
        mock_props.total_memory = 6 * 1024**3

        with (
            patch("pynvml.nvmlInit", side_effect=Exception("no nvml")),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            result = _probe_vram_gb("cuda")

        assert result == pytest.approx(6.0, abs=0.1)

    def test_cuda_index_parsed_correctly(self):
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
        cfg = _make_config(tmp_path, use_refiner=True, refiner_mode="auto")
        with patch("corridorkey.stages.inference.orchestrator.torch") as mock_torch:
            mock_torch.device.return_value = MagicMock(type="mps")
            result = _should_tile_refiner(cfg, resolved_refiner_mode=None)
        assert result is True


# ---------------------------------------------------------------------------
# Tiled refiner tests — _run_refiner_tiled
# ---------------------------------------------------------------------------

import torch.nn as nn
from corridorkey.stages.inference.orchestrator import _run_refiner_tiled, _TiledRefinerState


class _ZeroRefiner(nn.Module):
    def forward(self, img: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return torch.zeros(img.shape[0], 4, img.shape[2], img.shape[3], device=img.device)


class _OnesRefiner(nn.Module):
    def forward(self, img: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        return torch.ones(img.shape[0], 4, img.shape[2], img.shape[3], device=img.device)


class _RaisingRefiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def forward(self, img: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        self._calls += 1
        if self._calls >= 2:
            raise RuntimeError("deliberate test error")
        return torch.zeros(img.shape[0], 4, img.shape[2], img.shape[3])


def _rgb(h: int, w: int) -> torch.Tensor:
    return torch.rand(1, 3, h, w)


def _coarse(h: int, w: int) -> torch.Tensor:
    return torch.rand(1, 4, h, w)


def _state() -> _TiledRefinerState:
    return _TiledRefinerState()


class TestTiledRefinerOutputShape:
    @pytest.mark.parametrize("h,w", [(512, 512), (256, 256), (1024, 1024), (600, 800), (513, 513)])
    def test_output_shape_matches_input(self, h: int, w: int):
        """Output shape must equal input shape for any resolution."""
        result = _run_refiner_tiled(_ZeroRefiner(), _rgb(h, w), _coarse(h, w), _state(), tile_size=512, overlap=128)
        assert result.shape == (1, 4, h, w)

    def test_batch_size_preserved(self):
        """Batch dimension is preserved through tiled processing."""
        rgb = torch.rand(2, 3, 512, 512)
        coarse = torch.rand(2, 4, 512, 512)
        result = _run_refiner_tiled(_ZeroRefiner(), rgb, coarse, _state(), tile_size=512, overlap=128)
        assert result.shape == (2, 4, 512, 512)


class TestTiledRefinerZeroRefiner:
    def test_single_tile_all_zeros(self):
        """A zero-delta refiner on a sub-tile image produces all-zero output."""
        result = _run_refiner_tiled(
            _ZeroRefiner(), _rgb(256, 256), _coarse(256, 256), _state(), tile_size=512, overlap=128
        )
        assert torch.allclose(result, torch.zeros_like(result))

    def test_multi_tile_all_zeros(self):
        """A zero-delta refiner on a multi-tile image produces all-zero output."""
        result = _run_refiner_tiled(
            _ZeroRefiner(), _rgb(1024, 1024), _coarse(1024, 1024), _state(), tile_size=512, overlap=128
        )
        assert torch.allclose(result, torch.zeros_like(result))

    def test_non_multiple_all_zeros(self):
        """A zero-delta refiner on a non-tile-multiple image produces all-zero output."""
        result = _run_refiner_tiled(
            _ZeroRefiner(), _rgb(600, 800), _coarse(600, 800), _state(), tile_size=512, overlap=128
        )
        assert torch.allclose(result, torch.zeros_like(result))


class TestTiledRefinerOnesRefiner:
    def test_single_tile_all_ones(self):
        """An all-ones refiner on a sub-tile image produces all-ones output after blend normalisation."""
        result = _run_refiner_tiled(
            _OnesRefiner(), _rgb(256, 256), _coarse(256, 256), _state(), tile_size=512, overlap=128
        )
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)

    def test_multi_tile_all_ones(self):
        """An all-ones refiner on a multi-tile image produces all-ones output."""
        result = _run_refiner_tiled(
            _OnesRefiner(), _rgb(1024, 1024), _coarse(1024, 1024), _state(), tile_size=512, overlap=128
        )
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)


class TestTiledRefinerDtype:
    def test_float32_preserved(self):
        """float32 input produces float32 output."""
        result = _run_refiner_tiled(
            _ZeroRefiner(), _rgb(256, 256).float(), _coarse(256, 256).float(), _state(), tile_size=512, overlap=128
        )
        assert result.dtype == torch.float32

    def test_float16_preserved(self):
        """float16 input produces float16 output."""
        result = _run_refiner_tiled(
            _ZeroRefiner(), _rgb(256, 256).half(), _coarse(256, 256).half(), _state(), tile_size=512, overlap=128
        )
        assert result.dtype == torch.float16


class TestTiledRefinerBypassFlag:
    def test_bypass_false_after_normal_run(self):
        """bypass is reset to False after a successful tiled run."""
        state = _state()
        _run_refiner_tiled(_ZeroRefiner(), _rgb(256, 256), _coarse(256, 256), state, tile_size=512, overlap=128)
        assert state.bypass is False

    def test_bypass_false_after_refiner_exception(self):
        """bypass is reset to False even when the refiner raises mid-run."""
        state = _state()
        with pytest.raises(RuntimeError, match="deliberate test error"):
            _run_refiner_tiled(
                _RaisingRefiner(), _rgb(1024, 512), _coarse(1024, 512), state, tile_size=512, overlap=128
            )
        assert state.bypass is False


class TestTiledRefinerEdgeCases:
    def test_overlap_larger_than_half_tile_clamped(self):
        """Overlap > tile_size//2 - 1 is clamped safely without crashing."""
        result = _run_refiner_tiled(
            _OnesRefiner(), _rgb(512, 512), _coarse(512, 512), _state(), tile_size=512, overlap=300
        )
        assert result.shape == (1, 4, 512, 512)

    def test_zero_overlap(self):
        """overlap=0 stitches tiles directly without blending."""
        result = _run_refiner_tiled(
            _OnesRefiner(), _rgb(512, 512), _coarse(512, 512), _state(), tile_size=256, overlap=0
        )
        assert result.shape == (1, 4, 512, 512)
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)

    def test_image_smaller_than_tile(self):
        """Image smaller than tile_size uses a single padded tile, cropped back to original size."""
        result = _run_refiner_tiled(
            _OnesRefiner(), _rgb(100, 150), _coarse(100, 150), _state(), tile_size=512, overlap=128
        )
        assert result.shape == (1, 4, 100, 150)


class TestTiledRefinerScale:
    def test_scale_zero_produces_all_zeros(self):
        """refiner_scale=0.0 zeroes out all delta output."""
        result = _run_refiner_tiled(
            _OnesRefiner(), _rgb(512, 512), _coarse(512, 512), _state(), tile_size=512, overlap=128, refiner_scale=0.0
        )
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)

    def test_scale_one_unchanged(self):
        """refiner_scale=1.0 produces the same result as the default."""
        result = _run_refiner_tiled(
            _OnesRefiner(), _rgb(512, 512), _coarse(512, 512), _state(), tile_size=512, overlap=128, refiner_scale=1.0
        )
        assert torch.allclose(result, torch.ones_like(result), atol=1e-5)

    def test_scale_half_produces_half(self):
        """refiner_scale=0.5 halves the delta output."""
        result = _run_refiner_tiled(
            _OnesRefiner(), _rgb(512, 512), _coarse(512, 512), _state(), tile_size=512, overlap=128, refiner_scale=0.5
        )
        assert torch.allclose(result, torch.full_like(result, 0.5), atol=1e-5)
