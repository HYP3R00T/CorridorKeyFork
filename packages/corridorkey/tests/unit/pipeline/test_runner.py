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
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.preprocessor import PreprocessConfig, PreprocessedFrame
from corridorkey.stages.preprocessor.contracts import FrameMeta

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
# Runner — pre-loaded model (cfg.model) device validation
# ---------------------------------------------------------------------------


class TestRunnerPreloadedModel:
    """Tests for the cfg.model fast-path in _load_models."""

    def _make_model_on(self, device: torch.device) -> MagicMock:
        """Return a mock nn.Module whose first parameter lives on *device*."""
        param = MagicMock()
        param.device = device
        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([param]))
        return model

    def _run_with_preloaded(self, tmp_path: Path, model_device: torch.device, config_device: str) -> None:
        """Wire up Runner with a pre-loaded model and run it end-to-end."""
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
            patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey.runtime.worker.write_frame"),
        ):
            Runner(manifest, cfg).run()

    def test_exact_device_match_accepted(self, tmp_path: Path):
        """Model on cpu, config device cpu — should pass without error."""
        self._run_with_preloaded(tmp_path, torch.device("cpu"), "cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_bare_cuda_matches_cuda_0(self, tmp_path: Path):
        """Model on cuda:0, config device 'cuda' — must not raise (the bug fix)."""
        self._run_with_preloaded(tmp_path, torch.device("cuda:0"), "cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_0_matches_cuda_0(self, tmp_path: Path):
        """Model on cuda:0, config device 'cuda:0' — exact match should pass."""
        self._run_with_preloaded(tmp_path, torch.device("cuda:0"), "cuda:0")

    def test_device_mismatch_raises(self, tmp_path: Path):
        """Model on cpu, config device cuda:0 — must raise ValueError."""
        manifest = _make_manifest(tmp_path, frame_count=1)
        model = self._make_model_on(torch.device("cpu"))
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
            inference=_make_inference_config(tmp_path, device="cuda:0"),
            model=model,
        )
        with pytest.raises(ValueError, match="cfg.model is on"):
            Runner(manifest, cfg).run()

    def test_multi_device_with_preloaded_model_raises(self, tmp_path: Path):
        """cfg.model cannot be shared across multiple devices."""
        manifest = _make_manifest(tmp_path, frame_count=1)
        model = self._make_model_on(torch.device("cpu"))
        cfg = PipelineConfig(
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
            inference=_make_inference_config(tmp_path, device="cpu"),
            devices=["cpu", "cpu"],
            model=model,
        )
        with pytest.raises(ValueError, match="cfg.model cannot be shared"):
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


# ---------------------------------------------------------------------------
# Runner — cancellation
# ---------------------------------------------------------------------------


class TestRunnerCancellation:
    """Tests for Runner.cancel() — verifies JobCancelledError is raised and
    the pipeline shuts down cleanly without deadlocking."""

    def _make_config(self, tmp_path: Path) -> PipelineConfig:
        return _make_pipeline_config(tmp_path, devices=[])

    def _run_with_cancel(
        self,
        tmp_path: Path,
        frame_count: int = 10,
        cancel_after_frames: int = 2,
    ):
        """Run the pipeline and cancel after ``cancel_after_frames`` are written.

        Returns the JobCancelledError raised by run(), or None if it completed.
        """
        import threading

        manifest = _make_manifest(tmp_path, frame_count=frame_count)
        cfg = self._make_config(tmp_path)
        runner = Runner(manifest, cfg)

        written: list[int] = []
        error_holder: list[Exception] = []

        original_write = __import__("corridorkey.runtime.worker", fromlist=["write_frame"]).write_frame

        def counting_write(frame, config):
            written.append(frame.frame_index)
            if len(written) >= cancel_after_frames:
                runner.cancel()

        def fake_run_inference(frame, model, config, **kwargs):
            return _make_fake_result(frame.meta.frame_index)

        def run_target():
            try:
                with (
                    patch("corridorkey.stages.inference.loader.load_model", return_value=MagicMock()),
                    patch(
                        "corridorkey.stages.inference.orchestrator.run_inference",
                        side_effect=fake_run_inference,
                    ),
                    patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()),
                    patch("corridorkey.runtime.worker.write_frame", side_effect=counting_write),
                ):
                    runner.run()
            except Exception as e:
                error_holder.append(e)

        t = threading.Thread(target=run_target)
        t.start()
        t.join(timeout=15)

        assert not t.is_alive(), "Runner thread did not exit — possible deadlock"
        return error_holder[0] if error_holder else None

    def test_cancel_raises_job_cancelled_error(self, tmp_path: Path):
        from corridorkey.errors import JobCancelledError

        err = self._run_with_cancel(tmp_path, frame_count=10, cancel_after_frames=2)
        assert isinstance(err, JobCancelledError)

    def test_cancelled_error_contains_clip_name(self, tmp_path: Path):
        from corridorkey.errors import JobCancelledError

        err = self._run_with_cancel(tmp_path, frame_count=10, cancel_after_frames=2)
        assert isinstance(err, JobCancelledError)
        assert err.clip_name == "test"

    def test_cancel_does_not_deadlock(self, tmp_path: Path):
        """Cancelling on the first frame must not leave any thread blocked."""
        err = self._run_with_cancel(tmp_path, frame_count=10, cancel_after_frames=1)
        from corridorkey.errors import JobCancelledError

        assert isinstance(err, JobCancelledError)

    def test_cancel_before_run_raises_immediately(self, tmp_path: Path):
        """Calling cancel() before run() should cause run() to raise JobCancelledError."""
        from corridorkey.errors import JobCancelledError

        manifest = _make_manifest(tmp_path, frame_count=3)
        cfg = self._make_config(tmp_path)
        runner = Runner(manifest, cfg)
        runner.cancel()  # cancel before run

        def fake_run_inference(frame, model, config, **kwargs):
            return _make_fake_result(frame.meta.frame_index)

        with (
            patch("corridorkey.stages.inference.loader.load_model", return_value=MagicMock()),
            patch(
                "corridorkey.stages.inference.orchestrator.run_inference",
                side_effect=fake_run_inference,
            ),
            patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey.runtime.worker.write_frame"),
            pytest.raises(JobCancelledError),
        ):
            runner.run()

    def test_no_cancel_completes_normally(self, tmp_path: Path):
        """Without cancellation, run() should complete without raising."""
        err = self._run_with_cancel(tmp_path, frame_count=3, cancel_after_frames=999)
        assert err is None

    def test_cancel_is_idempotent(self, tmp_path: Path):
        """Calling cancel() multiple times should not raise or cause issues."""
        manifest = _make_manifest(tmp_path, frame_count=1)
        cfg = self._make_config(tmp_path)
        runner = Runner(manifest, cfg)
        runner.cancel()
        runner.cancel()  # second call must not raise
        runner.cancel()


# ---------------------------------------------------------------------------
# Runner — clip-level events
# ---------------------------------------------------------------------------


class TestRunnerClipEvents:
    """Verifies on_clip_complete and on_clip_error fire at the right times."""

    def _base_patches(self, frame_count: int = 3):
        """Return the standard mock stack used by most tests in this class."""
        from unittest.mock import MagicMock, patch

        def fake_run_inference(frame, model, config, **kwargs):
            return _make_fake_result(frame.meta.frame_index)

        return [
            patch("corridorkey.stages.inference.loader.load_model", return_value=MagicMock()),
            patch(
                "corridorkey.stages.inference.orchestrator.run_inference",
                side_effect=fake_run_inference,
            ),
            patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey.runtime.worker.write_frame"),
        ]

    def test_clip_complete_fires_on_success(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=3)
        cfg = _make_pipeline_config(tmp_path)

        completed: list[tuple[str, int]] = []
        cfg.events = PipelineEvents(on_clip_complete=lambda name, n: completed.append((name, n)))

        patches = self._base_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            Runner(manifest, cfg).run()

        assert completed == [("test", 3)]

    def test_clip_complete_carries_frame_count(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=5)
        cfg = _make_pipeline_config(tmp_path)

        frame_counts: list[int] = []
        cfg.events = PipelineEvents(on_clip_complete=lambda name, n: frame_counts.append(n))

        patches = self._base_patches(frame_count=5)
        with patches[0], patches[1], patches[2], patches[3]:
            Runner(manifest, cfg).run()

        assert frame_counts == [5]

    def test_clip_complete_not_fired_on_cancel(self, tmp_path: Path):
        from corridorkey.errors import JobCancelledError

        manifest = _make_manifest(tmp_path, frame_count=10)
        cfg = _make_pipeline_config(tmp_path)

        completed: list[str] = []
        cfg.events = PipelineEvents(on_clip_complete=lambda name, n: completed.append(name))

        runner = Runner(manifest, cfg)

        def fake_write(frame, config):
            runner.cancel()

        def fake_run_inference(frame, model, config, **kwargs):
            return _make_fake_result(frame.meta.frame_index)

        with (
            patch("corridorkey.stages.inference.loader.load_model", return_value=MagicMock()),
            patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=fake_run_inference),
            patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey.runtime.worker.write_frame", side_effect=fake_write),
            pytest.raises(JobCancelledError),
        ):
            runner.run()

        assert completed == []

    def test_clip_error_fires_on_cancel(self, tmp_path: Path):
        from corridorkey.errors import JobCancelledError

        manifest = _make_manifest(tmp_path, frame_count=10)
        cfg = _make_pipeline_config(tmp_path)

        errors: list[tuple[str, Exception]] = []
        cfg.events = PipelineEvents(on_clip_error=lambda name, err: errors.append((name, err)))

        runner = Runner(manifest, cfg)

        def fake_write(frame, config):
            runner.cancel()

        def fake_run_inference(frame, model, config, **kwargs):
            return _make_fake_result(frame.meta.frame_index)

        with (
            patch("corridorkey.stages.inference.loader.load_model", return_value=MagicMock()),
            patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=fake_run_inference),
            patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey.runtime.worker.write_frame", side_effect=fake_write),
            pytest.raises(JobCancelledError),
        ):
            runner.run()

        assert len(errors) == 1
        name, err = errors[0]
        assert name == "test"
        assert isinstance(err, JobCancelledError)

    def test_clip_error_fires_before_exception_propagates(self, tmp_path: Path):
        """on_clip_error must be called before JobCancelledError reaches the caller."""
        from corridorkey.errors import JobCancelledError

        manifest = _make_manifest(tmp_path, frame_count=10)
        cfg = _make_pipeline_config(tmp_path)

        sequence: list[str] = []
        cfg.events = PipelineEvents(on_clip_error=lambda name, err: sequence.append("event"))

        runner = Runner(manifest, cfg)

        def fake_write(frame, config):
            runner.cancel()

        def fake_run_inference(frame, model, config, **kwargs):
            return _make_fake_result(frame.meta.frame_index)

        try:
            with (
                patch("corridorkey.stages.inference.loader.load_model", return_value=MagicMock()),
                patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=fake_run_inference),
                patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()),
                patch("corridorkey.runtime.worker.write_frame", side_effect=fake_write),
            ):
                runner.run()
        except JobCancelledError:
            sequence.append("exception")

        assert sequence == ["event", "exception"]

    def test_clip_error_not_fired_on_success(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=3)
        cfg = _make_pipeline_config(tmp_path)

        errors: list[str] = []
        cfg.events = PipelineEvents(on_clip_error=lambda name, err: errors.append(name))

        patches = self._base_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            Runner(manifest, cfg).run()

        assert errors == []

    def test_both_callbacks_can_be_set_simultaneously(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=2)
        cfg = _make_pipeline_config(tmp_path)

        completed: list[str] = []
        errors: list[str] = []
        cfg.events = PipelineEvents(
            on_clip_complete=lambda name, n: completed.append(name),
            on_clip_error=lambda name, err: errors.append(name),
        )

        patches = self._base_patches()
        with patches[0], patches[1], patches[2], patches[3]:
            Runner(manifest, cfg).run()

        assert completed == ["test"]
        assert errors == []
