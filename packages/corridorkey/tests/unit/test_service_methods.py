"""Unit tests for CorridorKeyService methods that don't require a GPU.

Covers: default_inference_params, default_output_config, detect_device,
get_vram_info, is_engine_loaded, scan_clips, get_clips_by_state,
load_engine/unload_engine lifecycle, and helper converters.

No GPU, no model files, no real frame data needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from corridorkey.clip_state import ClipEntry, ClipState
from corridorkey.config import CorridorKeyConfig
from corridorkey.contracts import InferenceParams, OutputConfig
from corridorkey.service import (
    CorridorKeyService,
    inference_params_to_postprocess,
    output_config_to_write_config,
)


def _config(
    checkpoint_dir: str = "/fake/ckpt",
    device: str = "cpu",
    optimization_mode: str = "auto",
    precision: str = "fp32",
    despill_strength: float = 0.8,
    auto_despeckle: bool = True,
    despeckle_size: int = 300,
    refiner_scale: float = 0.9,
    source_passthrough: bool = True,
    edge_erode_px: int = 5,
    edge_blur_px: int = 11,
    input_is_linear: bool = True,
    fg_format: str = "png",
    matte_format: str = "png",
    comp_format: str = "png",
    processed_format: str = "png",
    exr_compression: str = "zip",
) -> CorridorKeyConfig:
    return CorridorKeyConfig(
        checkpoint_dir=Path(checkpoint_dir),
        device=device,
        optimization_mode=optimization_mode,
        precision=precision,
        despill_strength=despill_strength,
        auto_despeckle=auto_despeckle,
        despeckle_size=despeckle_size,
        refiner_scale=refiner_scale,
        source_passthrough=source_passthrough,
        edge_erode_px=edge_erode_px,
        edge_blur_px=edge_blur_px,
        input_is_linear=input_is_linear,
        fg_format=fg_format,
        matte_format=matte_format,
        comp_format=comp_format,
        processed_format=processed_format,
        exr_compression=exr_compression,
    )


class TestInferenceParamsToPostprocess:
    """inference_params_to_postprocess - field mapping."""

    def test_all_fields_mapped(self):
        params = InferenceParams(
            despill_strength=0.5,
            auto_despeckle=False,
            despeckle_size=200,
            source_passthrough=True,
            edge_erode_px=4,
            edge_blur_px=9,
        )
        pp = inference_params_to_postprocess(params)
        assert pp.despill_strength == 0.5
        assert pp.auto_despeckle is False
        assert pp.despeckle_size == 200
        assert pp.source_passthrough is True
        assert pp.edge_erode_px == 4
        assert pp.edge_blur_px == 9
        assert pp.fg_is_straight is True


class TestOutputConfigToWriteConfig:
    """output_config_to_write_config - field mapping + dirs injection."""

    def test_dirs_injected(self):
        cfg = OutputConfig(exr_compression="pxr24")
        dirs = {"fg": "/out/fg", "matte": "/out/matte", "comp": "/out/comp", "processed": "/out/proc"}
        wc = output_config_to_write_config(cfg, dirs)
        assert wc.dirs == dirs
        assert wc.exr_compression == "pxr24"

    def test_formats_preserved(self):
        cfg = OutputConfig(fg_format="png", matte_format="exr", comp_format="png", processed_format="exr")
        wc = output_config_to_write_config(cfg, {})
        assert wc.fg_format == "png"
        assert wc.matte_format == "exr"
        assert wc.processed_format == "exr"


class TestDefaultParams:
    """default_inference_params and default_output_config seed from config."""

    def test_inference_params_from_config(self):
        config = _config()
        service = CorridorKeyService(config)
        params = service.default_inference_params()
        assert params.despill_strength == 0.8
        assert params.despeckle_size == 300
        assert params.refiner_scale == 0.9
        assert params.source_passthrough is True
        assert params.edge_erode_px == 5
        assert params.edge_blur_px == 11
        assert params.input_is_linear is True

    def test_output_config_from_config(self):
        config = _config()
        service = CorridorKeyService(config)
        cfg = service.default_output_config()
        assert cfg.fg_format == "png"
        assert cfg.exr_compression == "zip"


class TestDetectDevice:
    """detect_device - resolves and stores the compute device."""

    def test_returns_resolved_device(self):
        service = CorridorKeyService(_config(device="cpu"))
        with patch("corridorkey.service.device_utils") as mock_du:
            mock_du.resolve_device.return_value = "cpu"
            result = service.detect_device("cpu")
        assert result == "cpu"

    def test_stores_device(self):
        service = CorridorKeyService(_config(device="cpu"))
        with patch("corridorkey.service.device_utils") as mock_du:
            mock_du.resolve_device.return_value = "cpu"
            service.detect_device("cpu")
        assert service._device == "cpu"


class TestGetVramInfo:
    """get_vram_info - returns empty dict when CUDA unavailable."""

    def test_returns_empty_when_cuda_unavailable(self):
        service = CorridorKeyService(_config())
        with (
            patch("corridorkey.service.device_utils"),
            patch("torch.cuda.is_available", return_value=False),
        ):
            result = service.get_vram_info()
        assert result == {}

    def test_returns_dict_when_cuda_available(self):
        service = CorridorKeyService(_config())
        mock_props = MagicMock()
        mock_props.total_mem = 8 * 1024**3
        mock_props.name = "Test GPU"
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch("torch.cuda.memory_reserved", return_value=1 * 1024**3),
            patch("torch.cuda.memory_allocated", return_value=512 * 1024**2),
        ):
            result = service.get_vram_info()
        assert "total" in result
        assert "name" in result
        assert result["name"] == "Test GPU"


class TestIsEngineLoaded:
    """is_engine_loaded - reflects engine load state."""

    def test_false_before_load(self):
        service = CorridorKeyService(_config())
        assert service.is_engine_loaded() is False

    def test_true_after_manual_set(self):
        service = CorridorKeyService(_config())
        service._engine = MagicMock()
        service._engine_loaded = True
        assert service.is_engine_loaded() is True

    def test_false_after_engine_none(self):
        service = CorridorKeyService(_config())
        service._engine = None
        service._engine_loaded = True  # inconsistent state
        assert service.is_engine_loaded() is False


class TestScanClips:
    """scan_clips - delegates to scan_clips_dir."""

    def test_delegates_to_scan_clips_dir(self):
        service = CorridorKeyService(_config())
        mock_clips = [MagicMock()]
        with patch("corridorkey.service.scan_clips_dir", return_value=mock_clips) as mock_scan:
            result = service.scan_clips("/some/dir")
        mock_scan.assert_called_once_with("/some/dir", allow_standalone_videos=True)
        assert result == mock_clips


class TestGetClipsByState:
    """get_clips_by_state - filters clips by state."""

    def _clips(self) -> list[ClipEntry]:
        return [
            ClipEntry(name="a", root_path="/a", state=ClipState.READY),
            ClipEntry(name="b", root_path="/b", state=ClipState.COMPLETE),
            ClipEntry(name="c", root_path="/c", state=ClipState.READY),
        ]

    def test_filters_ready(self):
        service = CorridorKeyService(_config())
        clips = self._clips()
        result = service.get_clips_by_state(clips, ClipState.READY)
        assert len(result) == 2
        assert all(c.state == ClipState.READY for c in result)

    def test_filters_complete(self):
        service = CorridorKeyService(_config())
        clips = self._clips()
        result = service.get_clips_by_state(clips, ClipState.COMPLETE)
        assert len(result) == 1
        assert result[0].name == "b"

    def test_empty_when_no_match(self):
        service = CorridorKeyService(_config())
        clips = self._clips()
        result = service.get_clips_by_state(clips, ClipState.ERROR)
        assert result == []


class TestUnloadEngine:
    """unload_engine - clears engine reference and calls cache clear."""

    def test_engine_cleared_after_unload(self):
        service = CorridorKeyService(_config())
        service._engine = MagicMock()
        service._engine_loaded = True
        with patch("corridorkey.service.device_utils") as mock_du:
            mock_du.clear_device_cache = MagicMock()
            service.unload_engine()
        assert service._engine is None
        assert service._engine_loaded is False

    def test_unload_calls_clear_cache(self):
        service = CorridorKeyService(_config())
        service._engine = MagicMock()
        service._engine_loaded = True
        with patch("corridorkey.service.device_utils") as mock_du:
            mock_du.clear_device_cache = MagicMock()
            service.unload_engine()
        mock_du.clear_device_cache.assert_called_once()

    def test_unload_when_no_engine_does_not_crash(self):
        service = CorridorKeyService(_config())
        with patch("corridorkey.service.device_utils") as mock_du:
            mock_du.clear_device_cache = MagicMock()
            service.unload_engine()  # must not raise


class TestJobQueue:
    """job_queue property - lazy initialisation."""

    def test_lazy_init(self):
        service = CorridorKeyService(_config())
        assert service._job_queue is None
        q = service.job_queue
        assert q is not None
        assert service._job_queue is q

    def test_same_instance_returned(self):
        service = CorridorKeyService(_config())
        q1 = service.job_queue
        q2 = service.job_queue
        assert q1 is q2
