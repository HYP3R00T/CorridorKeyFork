"""Unit tests for corridorkey_new.pipeline.worker."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from corridorkey_new.loader.contracts import ClipManifest
from corridorkey_new.pipeline.queue import STOP, BoundedQueue
from corridorkey_new.pipeline.worker import InferenceWorker, PostWriteWorker, PreprocessWorker
from corridorkey_new.preprocessor import FrameReadError, PreprocessConfig, PreprocessedFrame

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

        worker = PreprocessWorker(manifest=manifest, config=config, output_queue=q)
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

        worker = PreprocessWorker(manifest=manifest, config=config, output_queue=q)
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

        worker = PreprocessWorker(manifest=manifest, config=config, output_queue=q)
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
        original_preprocess = __import__("corridorkey_new.preprocessor", fromlist=["preprocess_frame"]).preprocess_frame

        def patched_preprocess(m, i, c, **kwargs):
            nonlocal call_count
            call_count += 1
            if i == 1:
                raise FrameReadError("simulated read error")
            return original_preprocess(m, i, c, **kwargs)

        with patch("corridorkey_new.pipeline.worker.preprocess_frame", side_effect=patched_preprocess):
            worker = PreprocessWorker(manifest=manifest, config=config, output_queue=q)
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
            "corridorkey_new.pipeline.worker.preprocess_frame",
            side_effect=FrameReadError("all broken"),
        ):
            worker = PreprocessWorker(manifest=manifest, config=config, output_queue=q)
            t = worker.start()
            t.join(timeout=10)

        assert q.get() is STOP

    def test_start_returns_thread(self, tmp_path: Path):
        import threading

        manifest = _make_manifest(tmp_path, frame_count=1)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        worker = PreprocessWorker(manifest=manifest, config=config, output_queue=q)
        t = worker.start()
        assert isinstance(t, threading.Thread)
        t.join(timeout=10)

    def test_tensor_shape_in_queue(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=1)
        config = PreprocessConfig(img_size=32, device="cpu")
        q: BoundedQueue = BoundedQueue(10)

        worker = PreprocessWorker(manifest=manifest, config=config, output_queue=q)
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
    def _make_fake_frame(self) -> MagicMock:
        return MagicMock(spec=PreprocessedFrame)

    def test_passes_items_through(self):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)

        frame = self._make_fake_frame()
        in_q.put(frame)
        in_q.put_stop()

        worker = InferenceWorker(input_queue=in_q, output_queue=out_q, model=None)
        t = worker.start()
        t.join(timeout=5)

        assert out_q.get() is frame
        assert out_q.get() is STOP

    def test_stop_propagates_downstream(self):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)

        in_q.put_stop()

        worker = InferenceWorker(input_queue=in_q, output_queue=out_q, model=None)
        t = worker.start()
        t.join(timeout=5)

        assert out_q.get() is STOP

    def test_multiple_frames_passed_through(self):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)

        frames = [self._make_fake_frame() for _ in range(4)]
        for f in frames:
            in_q.put(f)
        in_q.put_stop()

        worker = InferenceWorker(input_queue=in_q, output_queue=out_q, model=None)
        t = worker.start()
        t.join(timeout=5)

        received = []
        while True:
            item = out_q.get()
            if item is STOP:
                break
            received.append(item)

        assert received == frames


# ---------------------------------------------------------------------------
# PostWriteWorker (stub — drains queue)
# ---------------------------------------------------------------------------


class TestPostWriteWorker:
    def _make_fake_frame(self) -> MagicMock:
        return MagicMock(spec=PreprocessedFrame)

    def test_drains_queue_and_exits(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)

        for _ in range(3):
            in_q.put(self._make_fake_frame())
        in_q.put_stop()

        worker = PostWriteWorker(input_queue=in_q, output_dir=tmp_path)
        t = worker.start()
        t.join(timeout=5)

        assert not t.is_alive()

    def test_exits_immediately_on_stop(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        worker = PostWriteWorker(input_queue=in_q, output_dir=tmp_path)
        t = worker.start()
        t.join(timeout=5)

        assert not t.is_alive()

    def test_start_returns_thread(self, tmp_path: Path):
        import threading

        in_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        worker = PostWriteWorker(input_queue=in_q, output_dir=tmp_path)
        t = worker.start()
        assert isinstance(t, threading.Thread)
        t.join(timeout=5)
