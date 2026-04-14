"""Unit tests for internal frame-loop threading primitives.

Tests PipelineConfig, _InferenceWorker, _override_device,
and run_clip — the internal assembly line used by the Engine.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
from corridorkey.errors import JobCancelledError
from corridorkey.runtime.runner import PipelineConfig, _InferenceWorker, run_clip
from corridorkey.stages.inference import InferenceConfig, InferenceResult
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.preprocessor import PreprocessConfig, PreprocessedFrame
from corridorkey.stages.preprocessor.contracts import FrameMeta


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


def _make_inference_config(tmp_path: Path, device: str = "cpu") -> InferenceConfig:
    return InferenceConfig(checkpoint_path=tmp_path / "model.pth", device=device, img_size=512)


def _make_fake_frame(idx: int = 0) -> PreprocessedFrame:
    meta = FrameMeta(frame_index=idx, original_h=32, original_w=32)
    return PreprocessedFrame(tensor=torch.zeros(1, 4, 32, 32), meta=meta)


def _make_fake_result(frame_index: int = 0) -> InferenceResult:
    meta = FrameMeta(frame_index=frame_index, original_h=32, original_w=32)
    return InferenceResult(alpha=torch.zeros(1, 1, 32, 32), fg=torch.zeros(1, 3, 32, 32), meta=meta)


def _make_pipeline_config(tmp_path: Path, devices: list[str] | None = None) -> PipelineConfig:
    return PipelineConfig(
        preprocess=PreprocessConfig(img_size=512, device="cpu"),
        inference=_make_inference_config(tmp_path),
        devices=devices or [],
    )


# _InferenceWorker


class TestInferenceWorker:
    def test_processes_frames_and_sends_stop_when_last(self, tmp_path: Path):
        from corridorkey.runtime.queue import STOP, BoundedQueue
        from corridorkey.runtime.runner import _FrameWork, _InferenceWork

        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        frame = _make_fake_frame(0)
        result = _make_fake_result(0)
        manifest = _make_manifest(tmp_path)
        in_q.put(_FrameWork(frame=frame, manifest=manifest))
        in_q.put_stop()

        counter = [1]
        lock = threading.Lock()
        with patch("corridorkey.stages.inference.orchestrator.run_inference", return_value=result):
            t = _InferenceWorker(
                preprocess_queue=in_q,
                inference_queue=out_q,
                model=MagicMock(),
                config=_make_inference_config(tmp_path),
                remaining=counter,
                remaining_lock=lock,
            ).start()
            t.join(timeout=5)

        item = out_q.get()
        assert isinstance(item, _InferenceWork)
        _, _, meta = item.transfer.resolve()
        assert meta is result.meta
        assert out_q.get() is STOP

    def test_non_last_worker_does_not_send_stop(self, tmp_path: Path):
        from corridorkey.runtime.queue import BoundedQueue

        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        counter = [2]
        lock = threading.Lock()
        t = _InferenceWorker(
            preprocess_queue=in_q,
            inference_queue=out_q,
            model=MagicMock(),
            config=_make_inference_config(tmp_path),
            remaining=counter,
            remaining_lock=lock,
        ).start()
        t.join(timeout=5)

        assert len(out_q) == 0

    def test_two_workers_process_all_frames_exactly_once(self, tmp_path: Path):
        from corridorkey.runtime.queue import STOP, BoundedQueue
        from corridorkey.runtime.runner import _FrameWork, _InferenceWork

        in_q: BoundedQueue = BoundedQueue(20)
        out_q: BoundedQueue = BoundedQueue(20)
        manifest = _make_manifest(tmp_path)

        n_frames = 6
        frames = [_make_fake_frame(i) for i in range(n_frames)]
        results = [_make_fake_result(i) for i in range(n_frames)]
        for f in frames:
            in_q.put(_FrameWork(frame=f, manifest=manifest))
        in_q.put_stop()

        remaining = [2]
        lock = threading.Lock()
        call_map = {id(f): r for f, r in zip(frames, results, strict=True)}

        def side_effect(frame, model, config, **kwargs):
            return call_map[id(frame)]

        with patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=side_effect):
            for _ in range(2):
                _InferenceWorker(
                    preprocess_queue=in_q,
                    inference_queue=out_q,
                    model=MagicMock(),
                    config=_make_inference_config(tmp_path),
                    remaining=remaining,
                    remaining_lock=lock,
                ).start()

        received = []
        while True:
            item = out_q.get()
            if item is STOP:
                break
            assert isinstance(item, _InferenceWork)
            _, _, meta = item.transfer.resolve()
            received.append(meta)

        assert len(received) == n_frames
        assert {m.frame_index for m in received} == set(range(n_frames))


# run_clip — validation


class TestRunClipValidation:
    def test_missing_inference_raises(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
            inference=None,
        )
        with pytest.raises(ValueError, match="inference is not set"):
            run_clip(manifest, cfg)


# run_clip — end-to-end (mocked model + inference)


class TestRunClipEndToEnd:
    def _run(self, tmp_path, frame_count, devices):
        manifest = _make_manifest(tmp_path, frame_count=frame_count)
        cfg = _make_pipeline_config(tmp_path, devices=devices)

        def fake_run_inference(frame, model, config, **kwargs):
            return _make_fake_result(frame.meta.frame_index)

        with (
            patch("corridorkey.stages.inference.loader.load_model", return_value=MagicMock()),
            patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=fake_run_inference),
            patch("corridorkey.runtime.runner.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey.runtime.runner.write_frame"),
        ):
            run_clip(manifest, cfg)

    def test_no_devices_processes_all_frames(self, tmp_path: Path):
        self._run(tmp_path, frame_count=3, devices=[])

    def test_single_device_processes_all_frames(self, tmp_path: Path):
        self._run(tmp_path, frame_count=3, devices=["cpu"])

    def test_two_devices_process_all_frames(self, tmp_path: Path):
        self._run(tmp_path, frame_count=4, devices=["cpu", "cpu"])

    def test_model_load_failure_raises(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=1)
        cfg = _make_pipeline_config(tmp_path, devices=["cpu", "cpu"])

        from corridorkey.errors import ModelError

        with (
            patch("corridorkey.stages.inference.loader.load_model", side_effect=RuntimeError("no checkpoint")),
            pytest.raises(ModelError, match="Failed to load model"),
        ):
            run_clip(manifest, cfg)


# run_clip — pre-loaded model device validation


class TestRunClipPreloadedModel:
    def _make_model_on(self, device: torch.device) -> MagicMock:
        param = MagicMock()
        param.device = device
        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([param]))
        return model

    def _run_with_preloaded(self, tmp_path: Path, model_device: torch.device, config_device: str) -> None:
        manifest = _make_manifest(tmp_path, frame_count=1)
        model = self._make_model_on(model_device)
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
            inference=_make_inference_config(tmp_path, device=config_device),
            model=model,
        )

        def fake_run_inference(frame, mdl, config, **kwargs):
            return _make_fake_result(frame.meta.frame_index)

        with (
            patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=fake_run_inference),
            patch("corridorkey.runtime.runner.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey.runtime.runner.write_frame"),
        ):
            run_clip(manifest, cfg)

    def test_exact_device_match_accepted(self, tmp_path: Path):
        self._run_with_preloaded(tmp_path, torch.device("cpu"), "cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bare_cuda_matches_cuda_0(self, tmp_path: Path):
        self._run_with_preloaded(tmp_path, torch.device("cuda:0"), "cuda")

    def test_device_mismatch_raises(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=1)
        model = self._make_model_on(torch.device("cpu"))
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
            inference=_make_inference_config(tmp_path, device="cuda:0"),
            model=model,
        )
        with pytest.raises(ValueError, match="config.model is on"):
            run_clip(manifest, cfg)

    def test_multi_device_with_preloaded_model_raises(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=1)
        model = self._make_model_on(torch.device("cpu"))
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
            inference=_make_inference_config(tmp_path, device="cpu"),
            devices=["cpu", "cpu"],
            model=model,
        )
        with pytest.raises(ValueError, match="cannot be shared"):
            run_clip(manifest, cfg)


# PipelineConfig defaults


class TestPipelineConfigDefaults:
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


# run_clip — cancellation


class TestRunClipCancellation:
    def _run_with_cancel(self, tmp_path: Path, frame_count: int = 10, cancel_after_frames: int = 2):

        manifest = _make_manifest(tmp_path, frame_count=frame_count)
        cfg = _make_pipeline_config(tmp_path)
        cancel_event = threading.Event()

        written: list[int] = []
        error_holder: list[Exception] = []

        def counting_write(frame, config):
            written.append(frame.frame_index)
            if len(written) >= cancel_after_frames:
                cancel_event.set()

        def fake_run_inference(frame, model, config, **kwargs):
            return _make_fake_result(frame.meta.frame_index)

        def run_target():
            try:
                with (
                    patch("corridorkey.stages.inference.loader.load_model", return_value=MagicMock()),
                    patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=fake_run_inference),
                    patch("corridorkey.runtime.runner.postprocess_frame", return_value=MagicMock()),
                    patch("corridorkey.runtime.runner.write_frame", side_effect=counting_write),
                ):
                    run_clip(manifest, cfg, cancel_event=cancel_event)
            except Exception as e:
                error_holder.append(e)

        t = threading.Thread(target=run_target)
        t.start()
        t.join(timeout=15)

        assert not t.is_alive(), "run_clip thread did not exit — possible deadlock"
        return error_holder[0] if error_holder else None

    def test_cancel_raises_job_cancelled_error(self, tmp_path: Path):
        err = self._run_with_cancel(tmp_path, frame_count=10, cancel_after_frames=2)
        assert isinstance(err, JobCancelledError)

    def test_cancel_does_not_deadlock(self, tmp_path: Path):
        err = self._run_with_cancel(tmp_path, frame_count=10, cancel_after_frames=1)
        assert isinstance(err, JobCancelledError)

    def test_no_cancel_completes_normally(self, tmp_path: Path):
        err = self._run_with_cancel(tmp_path, frame_count=3, cancel_after_frames=999)
        assert err is None
