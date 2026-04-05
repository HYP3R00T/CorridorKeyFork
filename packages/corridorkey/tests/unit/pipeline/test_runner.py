"""Unit tests for Runner and PipelineConfig."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
from corridorkey.events import PipelineEvents
from corridorkey.runtime.runner import PipelineConfig, Runner, _AtomicCounter, _InferenceWorker, _override_device
from corridorkey.stages.inference import InferenceConfig, InferenceResult
from corridorkey.stages.loader.contracts import LoadResult
from corridorkey.stages.preprocessor import PreprocessConfig, PreprocessedFrame
from corridorkey.stages.preprocessor.contracts import FrameMeta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manifest(tmp_path: Path, frame_count: int = 2) -> LoadResult:
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
    return LoadResult(
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


# ---------------------------------------------------------------------------
# _AtomicCounter
# ---------------------------------------------------------------------------


class TestAtomicCounter:
    def test_decrement_returns_new_value(self):
        c = _AtomicCounter(3)
        assert c.decrement() == 2
        assert c.decrement() == 1
        assert c.decrement() == 0

    def test_thread_safe_decrement(self):
        import threading

        n = 50
        c = _AtomicCounter(n)
        results: list[int] = []
        lock = threading.Lock()

        def worker():
            v = c.decrement()
            with lock:
                results.append(v)

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert sorted(results) == list(range(0, n))


# ---------------------------------------------------------------------------
# _override_device
# ---------------------------------------------------------------------------


class TestOverrideDevice:
    def test_device_is_replaced(self, tmp_path: Path):
        cfg = _make_inference_config(tmp_path)
        assert _override_device(cfg, "cuda:1").device == "cuda:1"

    def test_other_fields_unchanged(self, tmp_path: Path):
        cfg = _make_inference_config(tmp_path)
        new_cfg = _override_device(cfg, "cuda:1")
        assert new_cfg.checkpoint_path == cfg.checkpoint_path
        assert new_cfg.img_size == cfg.img_size

    def test_original_config_unchanged(self, tmp_path: Path):
        cfg = _make_inference_config(tmp_path)
        _override_device(cfg, "cuda:1")
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# _InferenceWorker
# ---------------------------------------------------------------------------


class TestInferenceWorker:
    def _make_worker(self, tmp_path, in_q, out_q, counter, worker_index=0):

        return _InferenceWorker(
            preprocess_queue=in_q,
            inference_queue=out_q,
            model=MagicMock(),
            config=_make_inference_config(tmp_path),
            active_workers=counter,
        )

    def test_processes_frames_and_sends_stop_when_last(self, tmp_path: Path):
        from corridorkey.runtime.queue import STOP, BoundedQueue

        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        frame = _make_fake_frame(0)
        result = _make_fake_result(0)
        in_q.put(frame)
        in_q.put_stop()

        counter = _AtomicCounter(1)
        with patch("corridorkey.stages.inference.orchestrator.run_inference", return_value=result):
            t = _InferenceWorker(
                preprocess_queue=in_q,
                inference_queue=out_q,
                model=MagicMock(),
                config=_make_inference_config(tmp_path),
                active_workers=counter,
            ).start()
            t.join(timeout=5)

        assert out_q.get() is result
        assert out_q.get() is STOP

    def test_non_last_worker_does_not_send_stop(self, tmp_path: Path):
        from corridorkey.runtime.queue import BoundedQueue

        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        counter = _AtomicCounter(2)  # this worker is not last
        t = _InferenceWorker(
            preprocess_queue=in_q,
            inference_queue=out_q,
            model=MagicMock(),
            config=_make_inference_config(tmp_path),
            active_workers=counter,
        ).start()
        t.join(timeout=5)

        assert len(out_q) == 0

    def test_two_workers_process_all_frames_exactly_once(self, tmp_path: Path):
        from corridorkey.runtime.queue import STOP, BoundedQueue

        in_q: BoundedQueue = BoundedQueue(20)
        out_q: BoundedQueue = BoundedQueue(20)

        n_frames = 6
        frames = [_make_fake_frame(i) for i in range(n_frames)]
        results = [_make_fake_result(i) for i in range(n_frames)]
        for f in frames:
            in_q.put(f)
        in_q.put_stop()

        counter = _AtomicCounter(2)
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
                    active_workers=counter,
                ).start()

        received = []
        while True:
            item = out_q.get()
            if item is STOP:
                break
            received.append(item)

        assert len(received) == n_frames
        assert {r.meta.frame_index for r in received} == set(range(n_frames))


# ---------------------------------------------------------------------------
# Runner — missing inference config
# ---------------------------------------------------------------------------


class TestRunnerValidation:
    def test_missing_inference_raises(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
            inference=None,
        )
        with pytest.raises(ValueError, match="inference is not set"):
            Runner(manifest, cfg).run()


# ---------------------------------------------------------------------------
# Runner — events override
# ---------------------------------------------------------------------------


class TestRunnerEventsOverride:
    def _run_with_mocks(self, tmp_path, cfg, **runner_kwargs):
        """Run Runner with all workers mocked, return the events seen by _InferenceWorker."""
        manifest = _make_manifest(tmp_path, frame_count=1)
        fake_model = MagicMock()
        captured_events: list = []

        original_init = _InferenceWorker.__init__

        def capturing_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            captured_events.append(self.events)

        def fake_run_inference(frame, model, config, **kwargs):
            return _make_fake_result(frame.meta.frame_index)

        with (
            patch.object(_InferenceWorker, "__init__", capturing_init),
            patch("corridorkey.stages.inference.loader.load_model", return_value=fake_model),
            patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=fake_run_inference),
            patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey.runtime.worker.write_frame"),
        ):
            Runner(manifest, cfg, **runner_kwargs).run()

        return captured_events

    def test_events_kwarg_overrides_config_events(self, tmp_path: Path):
        config_events = PipelineEvents()
        kwarg_events = PipelineEvents()
        cfg = _make_pipeline_config(tmp_path)
        cfg.events = config_events

        seen = self._run_with_mocks(tmp_path, cfg, events=kwarg_events)
        assert all(e is kwarg_events for e in seen)

    def test_config_events_used_when_no_kwarg(self, tmp_path: Path):
        config_events = PipelineEvents()
        cfg = _make_pipeline_config(tmp_path)
        cfg.events = config_events

        seen = self._run_with_mocks(tmp_path, cfg)
        assert all(e is config_events for e in seen)

    def test_none_events_kwarg_does_not_override(self, tmp_path: Path):
        config_events = PipelineEvents()
        cfg = _make_pipeline_config(tmp_path)
        cfg.events = config_events

        seen = self._run_with_mocks(tmp_path, cfg, events=None)
        assert all(e is config_events for e in seen)


# ---------------------------------------------------------------------------
# Runner — end-to-end (mocked model + inference)
# ---------------------------------------------------------------------------


class TestRunnerEndToEnd:
    def _run(self, tmp_path, frame_count, devices):
        manifest = _make_manifest(tmp_path, frame_count=frame_count)
        cfg = _make_pipeline_config(tmp_path, devices=devices)
        fake_model = MagicMock()

        def fake_run_inference(frame, model, config, **kwargs):
            return _make_fake_result(frame.meta.frame_index)

        with (
            patch("corridorkey.stages.inference.loader.load_model", return_value=fake_model),
            patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=fake_run_inference),
            patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey.runtime.worker.write_frame"),
        ):
            Runner(manifest, cfg).run()

    def test_no_devices_processes_all_frames(self, tmp_path: Path):
        self._run(tmp_path, frame_count=3, devices=[])

    def test_single_device_processes_all_frames(self, tmp_path: Path):
        self._run(tmp_path, frame_count=3, devices=["cpu"])

    def test_two_devices_process_all_frames(self, tmp_path: Path):
        self._run(tmp_path, frame_count=4, devices=["cpu", "cpu"])

    def test_three_devices_process_all_frames(self, tmp_path: Path):
        self._run(tmp_path, frame_count=6, devices=["cpu", "cpu", "cpu"])

    def test_model_load_failure_raises(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=1)
        cfg = _make_pipeline_config(tmp_path, devices=["cpu", "cpu"])

        with (
            patch("corridorkey.stages.inference.loader.load_model", side_effect=RuntimeError("no checkpoint")),
            pytest.raises(RuntimeError, match="Failed to load model"),
        ):
            Runner(manifest, cfg).run()


# ---------------------------------------------------------------------------
# PipelineConfig defaults
# ---------------------------------------------------------------------------


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
