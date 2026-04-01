"""Unit tests for the unified Runner class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
from corridorkey.events import PipelineEvents
from corridorkey.runtime.runner import PipelineConfig, Runner
from corridorkey.stages.inference import InferenceConfig, InferenceResult
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.preprocessor import FrameMeta, PreprocessConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(tmp_path: Path, frame_count: int = 2) -> ClipManifest:
    frames_dir = tmp_path / "Frames"
    frames_dir.mkdir()
    alpha_dir = tmp_path / "AlphaFrames"
    alpha_dir.mkdir()
    output_dir = tmp_path / "Output"
    output_dir.mkdir()
    for i in range(frame_count):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"frame_{i:06d}.png"), img)
        cv2.imwrite(str(alpha_dir / f"frame_{i:06d}.png"), np.zeros((32, 32), dtype=np.uint8))
    return ClipManifest(
        clip_name="test",
        clip_root=tmp_path,
        frames_dir=frames_dir,
        alpha_frames_dir=alpha_dir,
        output_dir=output_dir,
        needs_alpha=False,
        frame_count=frame_count,
        frame_range=(0, frame_count),
        is_linear=False,
    )


def _make_inference_config(tmp_path: Path) -> InferenceConfig:
    return InferenceConfig(
        checkpoint_path=tmp_path / "model.pth",
        device="cpu",
        img_size=512,
    )


def _make_fake_result(frame_index: int = 0) -> InferenceResult:
    meta = FrameMeta(frame_index=frame_index, original_h=32, original_w=32)
    return InferenceResult(
        alpha=torch.zeros(1, 1, 32, 32),
        fg=torch.zeros(1, 3, 32, 32),
        meta=meta,
    )


def _make_pipeline_config(tmp_path: Path, devices: list[str] | None = None) -> PipelineConfig:
    return PipelineConfig(
        preprocess=PreprocessConfig(img_size=512, device="cpu"),
        inference=_make_inference_config(tmp_path),
        devices=devices or [],
    )


# ---------------------------------------------------------------------------
# Runner dispatches to PipelineRunner for 0 or 1 device
# ---------------------------------------------------------------------------


class TestRunnerSingleGPUDispatch:
    def test_empty_devices_uses_pipeline_runner(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        cfg = _make_pipeline_config(tmp_path, devices=[])

        with (
            patch("corridorkey.runtime.runner.PipelineRunner.run") as mock_run,
            patch("corridorkey.runtime.runner.MultiGPURunner.run") as mock_multi,
        ):
            Runner(manifest, cfg).run()

        mock_run.assert_called_once()
        mock_multi.assert_not_called()

    def test_single_device_uses_pipeline_runner(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        cfg = _make_pipeline_config(tmp_path, devices=["cpu"])

        with (
            patch("corridorkey.runtime.runner.PipelineRunner.run") as mock_run,
            patch("corridorkey.runtime.runner.MultiGPURunner.run") as mock_multi,
        ):
            Runner(manifest, cfg).run()

        mock_run.assert_called_once()
        mock_multi.assert_not_called()


# ---------------------------------------------------------------------------
# Runner dispatches to MultiGPURunner for 2+ devices
# ---------------------------------------------------------------------------


class TestRunnerMultiGPUDispatch:
    def test_two_devices_uses_multi_gpu_runner(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        cfg = _make_pipeline_config(tmp_path, devices=["cpu", "cpu"])

        with (
            patch("corridorkey.runtime.runner.PipelineRunner.run") as mock_single,
            patch("corridorkey.runtime.runner.MultiGPURunner.run") as mock_multi,
        ):
            Runner(manifest, cfg).run()

        mock_multi.assert_called_once()
        mock_single.assert_not_called()

    def test_three_devices_uses_multi_gpu_runner(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        cfg = _make_pipeline_config(tmp_path, devices=["cpu", "cpu", "cpu"])

        with (
            patch("corridorkey.runtime.runner.PipelineRunner.run") as mock_single,
            patch("corridorkey.runtime.runner.MultiGPURunner.run") as mock_multi,
        ):
            Runner(manifest, cfg).run()

        mock_multi.assert_called_once()
        mock_single.assert_not_called()

    def test_multi_gpu_requires_inference_config(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
            inference=None,  # not set
            devices=["cpu", "cpu"],
        )
        with pytest.raises(ValueError, match="inference must be set"):
            Runner(manifest, cfg).run()

    def test_multi_gpu_config_receives_correct_fields(self, tmp_path: Path):
        """Runner must forward queue depths, write config, and events to MultiGPUConfig."""
        from corridorkey.stages.writer.contracts import WriteConfig

        manifest = _make_manifest(tmp_path)
        write_cfg = WriteConfig(output_dir=tmp_path)
        events = PipelineEvents()
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
            inference=_make_inference_config(tmp_path),
            devices=["cpu", "cpu"],
            write=write_cfg,
            input_queue_depth=5,
            output_queue_depth=6,
            events=events,
        )

        captured = {}

        def fake_multi_gpu_runner_init(self, manifest, config):
            captured["config"] = config

        with (
            patch.object(
                __import__("corridorkey.runtime.runner", fromlist=["MultiGPURunner"]).MultiGPURunner,
                "__init__",
                fake_multi_gpu_runner_init,
            ),
            patch("corridorkey.runtime.runner.MultiGPURunner.run"),
        ):
            Runner(manifest, cfg).run()

        mc = captured["config"]
        assert mc.write is write_cfg
        assert mc.input_queue_depth == 5
        assert mc.output_queue_depth == 6
        assert mc.events is events


# ---------------------------------------------------------------------------
# Events override
# ---------------------------------------------------------------------------


class TestRunnerEventsOverride:
    def test_events_kwarg_overrides_config_events(self, tmp_path: Path):
        """Events passed to Runner() take precedence over config.events."""
        manifest = _make_manifest(tmp_path)

        config_events = PipelineEvents()
        kwarg_events = PipelineEvents()

        cfg = _make_pipeline_config(tmp_path)
        cfg.events = config_events

        captured = {}

        def fake_pipeline_runner_init(self, manifest, config):
            captured["events"] = config.events

        with (
            patch.object(
                __import__("corridorkey.runtime.runner", fromlist=["PipelineRunner"]).PipelineRunner,
                "__init__",
                fake_pipeline_runner_init,
            ),
            patch("corridorkey.runtime.runner.PipelineRunner.run"),
        ):
            Runner(manifest, cfg, events=kwarg_events).run()

        assert captured["events"] is kwarg_events

    def test_config_events_used_when_no_kwarg(self, tmp_path: Path):
        """When no events kwarg is given, config.events is used."""
        manifest = _make_manifest(tmp_path)

        config_events = PipelineEvents()
        cfg = _make_pipeline_config(tmp_path)
        cfg.events = config_events

        captured = {}

        def fake_pipeline_runner_init(self, manifest, config):
            captured["events"] = config.events

        with (
            patch.object(
                __import__("corridorkey.runtime.runner", fromlist=["PipelineRunner"]).PipelineRunner,
                "__init__",
                fake_pipeline_runner_init,
            ),
            patch("corridorkey.runtime.runner.PipelineRunner.run"),
        ):
            Runner(manifest, cfg).run()

        assert captured["events"] is config_events

    def test_none_events_kwarg_does_not_override(self, tmp_path: Path):
        """Passing events=None explicitly should not override config.events."""
        manifest = _make_manifest(tmp_path)

        config_events = PipelineEvents()
        cfg = _make_pipeline_config(tmp_path)
        cfg.events = config_events

        captured = {}

        def fake_pipeline_runner_init(self, manifest, config):
            captured["events"] = config.events

        with (
            patch.object(
                __import__("corridorkey.runtime.runner", fromlist=["PipelineRunner"]).PipelineRunner,
                "__init__",
                fake_pipeline_runner_init,
            ),
            patch("corridorkey.runtime.runner.PipelineRunner.run"),
        ):
            Runner(manifest, cfg, events=None).run()

        assert captured["events"] is config_events


# ---------------------------------------------------------------------------
# PipelineConfig.devices field
# ---------------------------------------------------------------------------


class TestPipelineConfigDevices:
    def test_devices_defaults_to_empty_list(self, tmp_path: Path):
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
            inference=_make_inference_config(tmp_path),
        )
        assert cfg.devices == []

    def test_devices_field_stored(self, tmp_path: Path):
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
            inference=_make_inference_config(tmp_path),
            devices=["cuda:0", "cuda:1"],
        )
        assert cfg.devices == ["cuda:0", "cuda:1"]


# ---------------------------------------------------------------------------
# End-to-end: Runner with mocked workers (single GPU)
# ---------------------------------------------------------------------------


class TestRunnerEndToEnd:
    def test_single_gpu_processes_all_frames(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=3)
        cfg = _make_pipeline_config(tmp_path)

        # load_model checks file existence before the mock intercepts it
        (tmp_path / "model.pth").touch()
        fake_model = MagicMock()

        def fake_run_inference(frame, model, config, **kwargs):
            return _make_fake_result(frame.meta.frame_index)

        with (
            patch("corridorkey.stages.inference.load_model", return_value=fake_model),
            patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=fake_run_inference),
            patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey.runtime.worker.write_frame"),
        ):
            Runner(manifest, cfg).run()
