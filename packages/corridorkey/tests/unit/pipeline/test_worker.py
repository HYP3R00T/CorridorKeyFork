"""Unit tests for corridorkey.runtime.worker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch
from corridorkey.events import PipelineEvents
from corridorkey.runtime.queue import STOP, BoundedQueue
from corridorkey.runtime.runner import _AtomicCounter, _InferenceWorker
from corridorkey.runtime.worker import PostWriteWorker, PreprocessWorker
from corridorkey.stages.inference import InferenceConfig, InferenceResult
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.preprocessor import FrameReadError, PreprocessConfig, PreprocessedFrame
from corridorkey.stages.preprocessor.contracts import FrameMeta

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_png(path: Path, h: int = 32, w: int = 32, channels: int = 3) -> None:
    img = np.zeros((h, w), dtype=np.uint8) if channels == 1 else np.zeros((h, w, channels), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_manifest(tmp_path: Path, frame_count: int = 2) -> ClipManifest:
    frames_dir = tmp_path / "Frames"
    frames_dir.mkdir()
    alpha_dir = tmp_path / "AlphaFrames"
    alpha_dir.mkdir()
    output_dir = tmp_path / "Output"
    output_dir.mkdir()

    for i in range(frame_count):
        _write_png(frames_dir / f"frame_{i:06d}.png", channels=3)
        _write_png(alpha_dir / f"frame_{i:06d}.png", channels=1)

    return ClipManifest(
        clip_name="test_clip",
        clip_root=tmp_path,
        frames_dir=frames_dir,
        alpha_frames_dir=alpha_dir,
        output_dir=output_dir,
        needs_alpha=False,
        frame_count=frame_count,
        frame_range=(0, frame_count),
        is_linear=False,
    )


# ---------------------------------------------------------------------------
# PreprocessWorker
# ---------------------------------------------------------------------------


class TestPreprocessWorker:
    def test_queues_all_frames(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=3)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        worker = PreprocessWorker(manifest=manifest, config=config, preprocess_queue=q)
        t = worker.start()
        t.join(timeout=10)

        items = []
        while True:
            item = q.get()
            if item is STOP:
                break
            items.append(item)

        assert len(items) == 3
        assert all(isinstance(f, PreprocessedFrame) for f in items)

    def test_frame_indices_are_sequential(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=3)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        worker = PreprocessWorker(manifest=manifest, config=config, preprocess_queue=q)
        t = worker.start()
        t.join(timeout=10)

        indices = []
        while True:
            item = q.get()
            if item is STOP:
                break
            assert isinstance(item, PreprocessedFrame)
            indices.append(item.meta.frame_index)

        assert indices == [0, 1, 2]

    def test_stop_sent_after_all_frames(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=2)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        worker = PreprocessWorker(manifest=manifest, config=config, preprocess_queue=q)
        t = worker.start()
        t.join(timeout=10)

        # Drain all frames
        for _ in range(manifest.frame_count):
            q.get()

        # Next item must be STOP
        assert q.get() is STOP

    def test_frame_read_error_skips_frame(self, tmp_path: Path):
        """FrameReadError on one frame should skip it, not abort the worker."""
        manifest = _make_manifest(tmp_path, frame_count=3)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        call_count = 0
        original_preprocess = __import__(
            "corridorkey.stages.preprocessor", fromlist=["preprocess_frame"]
        ).preprocess_frame

        def patched_preprocess(m, i, c, **kwargs):
            nonlocal call_count
            call_count += 1
            if i == 1:
                raise FrameReadError("simulated read error")
            return original_preprocess(m, i, c, **kwargs)

        with patch("corridorkey.runtime.worker.preprocess_frame", side_effect=patched_preprocess):
            worker = PreprocessWorker(manifest=manifest, config=config, preprocess_queue=q)
            t = worker.start()
            t.join(timeout=10)

        items = []
        while True:
            item = q.get()
            if item is STOP:
                break
            items.append(item)

        # Frame 1 was skipped — only frames 0 and 2 should be queued
        assert len(items) == 2
        assert items[0].meta.frame_index == 0
        assert items[1].meta.frame_index == 2

    def test_stop_sent_even_on_all_errors(self, tmp_path: Path):
        """STOP must be sent even if every frame raises FrameReadError."""
        manifest = _make_manifest(tmp_path, frame_count=2)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        with patch(
            "corridorkey.runtime.worker.preprocess_frame",
            side_effect=FrameReadError("all broken"),
        ):
            worker = PreprocessWorker(manifest=manifest, config=config, preprocess_queue=q)
            t = worker.start()
            t.join(timeout=10)

        assert q.get() is STOP

    def test_start_returns_thread(self, tmp_path: Path):
        import threading

        manifest = _make_manifest(tmp_path, frame_count=1)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        worker = PreprocessWorker(manifest=manifest, config=config, preprocess_queue=q)
        t = worker.start()
        assert isinstance(t, threading.Thread)
        t.join(timeout=10)

    def test_tensor_shape_in_queue(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=1)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        worker = PreprocessWorker(manifest=manifest, config=config, preprocess_queue=q)
        t = worker.start()
        t.join(timeout=10)

        frame = q.get()
        assert frame is not STOP
        assert isinstance(frame, PreprocessedFrame)
        assert frame.tensor.shape == (1, 4, 32, 32)


# ---------------------------------------------------------------------------
# InferenceWorker (stub — pass-through behaviour)
# ---------------------------------------------------------------------------


class TestInferenceWorker:
    """_InferenceWorker tests mock run_inference — no checkpoint required."""

    def _make_config(self, tmp_path: Path) -> InferenceConfig:
        return InferenceConfig(checkpoint_path=tmp_path / "model.pth", device="cpu")

    def _make_fake_frame(self) -> PreprocessedFrame:
        meta = FrameMeta(frame_index=0, original_h=64, original_w=64)
        tensor = torch.zeros(1, 4, 32, 32)
        return PreprocessedFrame(tensor=tensor, meta=meta)

    def _make_fake_result(self, frame: PreprocessedFrame) -> InferenceResult:
        return InferenceResult(
            alpha=torch.zeros(1, 1, 32, 32),
            fg=torch.zeros(1, 3, 32, 32),
            meta=frame.meta,
        )

    def _worker(self, tmp_path, in_q, out_q, **kwargs):
        return _InferenceWorker(
            preprocess_queue=in_q,
            inference_queue=out_q,
            model=MagicMock(),
            config=self._make_config(tmp_path),
            active_workers=_AtomicCounter(1),
            **kwargs,
        )

    def test_passes_items_through(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)

        frame = self._make_fake_frame()
        result = self._make_fake_result(frame)
        in_q.put(frame)
        in_q.put_stop()

        with patch("corridorkey.stages.inference.orchestrator.run_inference", return_value=result):
            t = self._worker(tmp_path, in_q, out_q).start()
            t.join(timeout=5)

        assert out_q.get() is result
        assert out_q.get() is STOP

    def test_stop_propagates_downstream(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        t = self._worker(tmp_path, in_q, out_q).start()
        t.join(timeout=5)

        assert out_q.get() is STOP

    def test_multiple_frames_passed_through(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)

        frames = [self._make_fake_frame() for _ in range(4)]
        results = [self._make_fake_result(f) for f in frames]
        for f in frames:
            in_q.put(f)
        in_q.put_stop()

        with patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=results):
            t = self._worker(tmp_path, in_q, out_q).start()
            t.join(timeout=5)

        received = []
        while True:
            item = out_q.get()
            if item is STOP:
                break
            received.append(item)

        assert received == results

    def test_inference_error_skips_frame(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)

        frame1 = self._make_fake_frame()
        frame2 = self._make_fake_frame()
        result2 = self._make_fake_result(frame2)
        in_q.put(frame1)
        in_q.put(frame2)
        in_q.put_stop()

        def side_effect(frame, model, config, **kwargs):
            if frame is frame1:
                raise RuntimeError("simulated inference failure")
            return result2

        with patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=side_effect):
            t = self._worker(tmp_path, in_q, out_q).start()
            t.join(timeout=5)

        assert out_q.get() is result2
        assert out_q.get() is STOP

    def test_stop_sent_even_on_all_errors(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)

        in_q.put(self._make_fake_frame())
        in_q.put_stop()

        with patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=RuntimeError("boom")):
            t = self._worker(tmp_path, in_q, out_q).start()
            t.join(timeout=5)

        assert out_q.get() is STOP


# ---------------------------------------------------------------------------
# PostWriteWorker — mocks postprocess_frame and write_frame
# ---------------------------------------------------------------------------


class TestPostWriteWorker:
    def _make_fake_result(self) -> InferenceResult:
        meta = FrameMeta(frame_index=0, original_h=64, original_w=64)
        return InferenceResult(
            alpha=torch.zeros(1, 1, 32, 32),
            fg=torch.zeros(1, 3, 32, 32),
            meta=meta,
        )

    def test_drains_queue_and_exits(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)

        for _ in range(3):
            in_q.put(self._make_fake_result())
        in_q.put_stop()

        with (
            patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey.runtime.worker.write_frame"),
        ):
            worker = PostWriteWorker(inference_queue=in_q, output_dir=tmp_path)
            t = worker.start()
            t.join(timeout=5)

        assert not t.is_alive()

    def test_exits_immediately_on_stop(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        worker = PostWriteWorker(inference_queue=in_q, output_dir=tmp_path)
        t = worker.start()
        t.join(timeout=5)

        assert not t.is_alive()

    def test_start_returns_thread(self, tmp_path: Path):
        import threading

        in_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        worker = PostWriteWorker(inference_queue=in_q, output_dir=tmp_path)
        t = worker.start()
        assert isinstance(t, threading.Thread)
        t.join(timeout=5)

    def test_postprocess_and_write_called_per_frame(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        for _ in range(3):
            in_q.put(self._make_fake_result())
        in_q.put_stop()

        with (
            patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()) as mock_pp,
            patch("corridorkey.runtime.worker.write_frame") as mock_wf,
        ):
            worker = PostWriteWorker(inference_queue=in_q, output_dir=tmp_path)
            t = worker.start()
            t.join(timeout=5)

        assert mock_pp.call_count == 3
        assert mock_wf.call_count == 3

    def test_error_skips_frame_does_not_abort(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        in_q.put(self._make_fake_result())
        in_q.put(self._make_fake_result())
        in_q.put_stop()

        with (
            patch("corridorkey.runtime.worker.postprocess_frame", side_effect=RuntimeError("boom")),
            patch("corridorkey.runtime.worker.write_frame"),
        ):
            worker = PostWriteWorker(inference_queue=in_q, output_dir=tmp_path)
            t = worker.start()
            t.join(timeout=5)

        assert not t.is_alive()


# ---------------------------------------------------------------------------
# Worker event callbacks
# ---------------------------------------------------------------------------


class TestPreprocessWorkerEvents:
    def test_stage_start_fires(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=2)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        starts: list[tuple[str, int]] = []
        events = PipelineEvents(on_stage_start=lambda s, t: starts.append((s, t)))
        worker = PreprocessWorker(manifest=manifest, config=config, preprocess_queue=q, events=events)
        t = worker.start()
        t.join(timeout=10)
        # drain
        while q.get() is not STOP:
            pass

        assert any(s == "preprocess" for s, _ in starts)

    def test_stage_done_fires(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=1)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        dones: list[str] = []
        events = PipelineEvents(on_stage_done=lambda s: dones.append(s))
        worker = PreprocessWorker(manifest=manifest, config=config, preprocess_queue=q, events=events)
        t = worker.start()
        t.join(timeout=10)
        while q.get() is not STOP:
            pass

        assert "preprocess" in dones

    def test_preprocess_queued_fires_per_frame(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=3)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        queued: list[int] = []
        events = PipelineEvents(on_preprocess_queued=lambda i: queued.append(i))
        worker = PreprocessWorker(manifest=manifest, config=config, preprocess_queue=q, events=events)
        t = worker.start()
        t.join(timeout=10)
        while q.get() is not STOP:
            pass

        assert queued == [0, 1, 2]

    def test_frame_error_fires_on_read_error(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=2)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        errors: list[tuple[str, int]] = []
        events = PipelineEvents(on_frame_error=lambda s, i, e: errors.append((s, i)))

        with patch(
            "corridorkey.runtime.worker.preprocess_frame",
            side_effect=FrameReadError("bad"),
        ):
            worker = PreprocessWorker(manifest=manifest, config=config, preprocess_queue=q, events=events)
            t = worker.start()
            t.join(timeout=10)

        assert len(errors) == 2
        assert all(s == "preprocess" for s, _ in errors)

    def test_queue_depth_fires_with_postwrite_queue(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=2)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)
        pw_q: BoundedQueue = BoundedQueue(10)

        depths: list[tuple[int, int]] = []
        events = PipelineEvents(on_queue_depth=lambda p, w: depths.append((p, w)))
        worker = PreprocessWorker(
            manifest=manifest, config=config, preprocess_queue=q, inference_queue=pw_q, events=events
        )
        t = worker.start()
        t.join(timeout=10)
        while q.get() is not STOP:
            pass

        assert len(depths) == 2  # one per frame


class TestInferenceWorkerEvents:
    def _make_config(self, tmp_path: Path) -> InferenceConfig:
        return InferenceConfig(checkpoint_path=tmp_path / "model.pth", device="cpu")

    def _make_fake_frame(self, idx: int = 0) -> PreprocessedFrame:
        import torch
        from corridorkey.stages.preprocessor.contracts import FrameMeta

        meta = FrameMeta(frame_index=idx, original_h=32, original_w=32)
        return PreprocessedFrame(tensor=torch.zeros(1, 4, 32, 32), meta=meta)

    def _make_fake_result(self, frame: PreprocessedFrame) -> InferenceResult:
        import torch

        return InferenceResult(
            alpha=torch.zeros(1, 1, 32, 32),
            fg=torch.zeros(1, 3, 32, 32),
            meta=frame.meta,
        )

    def _worker(self, tmp_path, in_q, out_q, **kwargs):
        return _InferenceWorker(
            preprocess_queue=in_q,
            inference_queue=out_q,
            model=MagicMock(),
            config=self._make_config(tmp_path),
            active_workers=_AtomicCounter(1),
            **kwargs,
        )

    def test_stage_start_fires(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        starts: list[str] = []
        events = PipelineEvents(on_stage_start=lambda s, t: starts.append(s))
        t = self._worker(tmp_path, in_q, out_q, events=events).start()
        t.join(timeout=5)

        assert any("inference" in s for s in starts)

    def test_stage_done_fires(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        dones: list[str] = []
        events = PipelineEvents(on_stage_done=lambda s: dones.append(s))
        t = self._worker(tmp_path, in_q, out_q, events=events).start()
        t.join(timeout=5)

        assert any("inference" in s for s in dones)

    def test_inference_start_fires_per_frame(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        frame = self._make_fake_frame(3)
        result = self._make_fake_result(frame)
        in_q.put(frame)
        in_q.put_stop()

        started: list[int] = []
        events = PipelineEvents(on_inference_start=lambda i: started.append(i))
        with patch("corridorkey.stages.inference.orchestrator.run_inference", return_value=result):
            t = self._worker(tmp_path, in_q, out_q, events=events).start()
            t.join(timeout=5)

        assert started == [3]

    def test_inference_queued_fires_per_frame(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        frame = self._make_fake_frame(7)
        result = self._make_fake_result(frame)
        in_q.put(frame)
        in_q.put_stop()

        queued: list[int] = []
        events = PipelineEvents(on_inference_queued=lambda i: queued.append(i))
        with patch("corridorkey.stages.inference.orchestrator.run_inference", return_value=result):
            t = self._worker(tmp_path, in_q, out_q, events=events).start()
            t.join(timeout=5)

        assert queued == [7]

    def test_frame_error_fires_on_inference_exception(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        frame = self._make_fake_frame(2)
        in_q.put(frame)
        in_q.put_stop()

        errors: list[tuple[str, int]] = []
        events = PipelineEvents(on_frame_error=lambda s, i, e: errors.append((s, i)))
        with patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=RuntimeError("boom")):
            t = self._worker(tmp_path, in_q, out_q, events=events).start()
            t.join(timeout=5)

        assert len(errors) == 1
        assert errors[0][1] == 2


class TestPostWriteWorkerEvents:
    def _make_fake_result(self, idx: int = 0) -> InferenceResult:
        import torch
        from corridorkey.stages.preprocessor.contracts import FrameMeta

        meta = FrameMeta(frame_index=idx, original_h=32, original_w=32)
        return InferenceResult(
            alpha=torch.zeros(1, 1, 32, 32),
            fg=torch.zeros(1, 3, 32, 32),
            meta=meta,
        )

    def test_stage_start_fires(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        starts: list[tuple[str, int]] = []
        events = PipelineEvents(on_stage_start=lambda s, t: starts.append((s, t)))
        worker = PostWriteWorker(inference_queue=in_q, output_dir=tmp_path, total_frames=5, events=events)
        t = worker.start()
        t.join(timeout=5)

        assert any(s == "postwrite" for s, _ in starts)
        assert any(total == 5 for _, total in starts)

    def test_stage_done_fires(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        dones: list[str] = []
        events = PipelineEvents(on_stage_done=lambda s: dones.append(s))
        worker = PostWriteWorker(inference_queue=in_q, output_dir=tmp_path, events=events)
        t = worker.start()
        t.join(timeout=5)

        assert "postwrite" in dones

    def test_frame_written_fires_per_frame(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        for i in range(3):
            in_q.put(self._make_fake_result(i))
        in_q.put_stop()

        written: list[int] = []
        events = PipelineEvents(on_frame_written=lambda i, t: written.append(i))
        with (
            patch("corridorkey.runtime.worker.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey.runtime.worker.write_frame"),
        ):
            worker = PostWriteWorker(inference_queue=in_q, output_dir=tmp_path, total_frames=3, events=events)
            t = worker.start()
            t.join(timeout=5)

        assert sorted(written) == [0, 1, 2]

    def test_frame_error_fires_on_postprocess_exception(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        in_q.put(self._make_fake_result(4))
        in_q.put_stop()

        errors: list[tuple[str, int]] = []
        events = PipelineEvents(on_frame_error=lambda s, i, e: errors.append((s, i)))
        with patch("corridorkey.runtime.worker.postprocess_frame", side_effect=RuntimeError("boom")):
            worker = PostWriteWorker(inference_queue=in_q, output_dir=tmp_path, events=events)
            t = worker.start()
            t.join(timeout=5)

        assert errors == [("postwrite", 4)]
