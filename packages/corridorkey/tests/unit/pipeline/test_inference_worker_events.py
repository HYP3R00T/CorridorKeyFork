"""Tests for _InferenceWorker event callbacks."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch
from corridorkey.events import PipelineEvents
from corridorkey.runtime.queue import BoundedQueue
from corridorkey.runtime.runner import _FrameWork, _InferenceWorker
from corridorkey.stages.inference import InferenceConfig, InferenceResult
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.preprocessor import PreprocessedFrame
from corridorkey.stages.preprocessor.contracts import FrameMeta


def _make_config(tmp_path: Path) -> InferenceConfig:
    return InferenceConfig(checkpoint_path=tmp_path / "model.pth", device="cpu")


def _make_frame(idx: int = 0) -> PreprocessedFrame:
    meta = FrameMeta(frame_index=idx, original_h=32, original_w=32)
    return PreprocessedFrame(tensor=torch.zeros(1, 4, 32, 32), meta=meta)


def _make_result(frame: PreprocessedFrame) -> InferenceResult:
    return InferenceResult(
        alpha=torch.zeros(1, 1, 32, 32),
        fg=torch.zeros(1, 3, 32, 32),
        meta=frame.meta,
    )


def _make_manifest(tmp_path: Path) -> ClipManifest:
    frames_dir = tmp_path / "Frames"
    frames_dir.mkdir(exist_ok=True)
    alpha_dir = tmp_path / "AlphaFrames"
    alpha_dir.mkdir(exist_ok=True)
    output_dir = tmp_path / "Output"
    output_dir.mkdir(exist_ok=True)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    cv2.imwrite(str(frames_dir / "frame_000000.png"), img)
    cv2.imwrite(str(alpha_dir / "frame_000000.png"), np.zeros((32, 32), dtype=np.uint8))
    return ClipManifest(
        clip_name="test",
        clip_root=tmp_path,
        frames_dir=frames_dir,
        alpha_frames_dir=alpha_dir,
        output_dir=output_dir,
        needs_alpha=False,
        frame_count=1,
        frame_range=(0, 1),
        is_linear=False,
    )


def _wrap(frame: PreprocessedFrame, tmp_path: Path) -> _FrameWork:
    return _FrameWork(frame=frame, manifest=_make_manifest(tmp_path))


class TestInferenceWorkerEvents:
    def test_stage_start_fires(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        starts: list[str] = []
        events = PipelineEvents(on_stage_start=lambda s, t: starts.append(s))
        worker = _InferenceWorker(
            preprocess_queue=in_q,
            inference_queue=out_q,
            model=MagicMock(),
            config=_make_config(tmp_path),
            remaining=[1],
            remaining_lock=threading.Lock(),
            events=events,
        )
        t = worker.start()
        t.join(timeout=5)

        assert any("inference" in s for s in starts)

    def test_stage_done_fires(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        dones: list[str] = []
        events = PipelineEvents(on_stage_done=lambda s: dones.append(s))
        worker = _InferenceWorker(
            preprocess_queue=in_q,
            inference_queue=out_q,
            model=MagicMock(),
            config=_make_config(tmp_path),
            remaining=[1],
            remaining_lock=threading.Lock(),
            events=events,
        )
        t = worker.start()
        t.join(timeout=5)

        assert any("inference" in s for s in dones)

    def test_inference_start_fires_per_frame(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        frame = _make_frame(5)
        result = _make_result(frame)
        in_q.put(_wrap(frame, tmp_path))
        in_q.put_stop()

        started: list[int] = []
        events = PipelineEvents(on_inference_start=lambda i: started.append(i))
        with patch("corridorkey.stages.inference.orchestrator.run_inference", return_value=result):
            worker = _InferenceWorker(
                preprocess_queue=in_q,
                inference_queue=out_q,
                model=MagicMock(),
                config=_make_config(tmp_path),
                remaining=[1],
                remaining_lock=threading.Lock(),
                events=events,
            )
            t = worker.start()
            t.join(timeout=5)

        assert started == [5]

    def test_inference_queued_fires_per_frame(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        frame = _make_frame(3)
        result = _make_result(frame)
        in_q.put(_wrap(frame, tmp_path))
        in_q.put_stop()

        queued: list[int] = []
        events = PipelineEvents(on_inference_queued=lambda i: queued.append(i))
        with patch("corridorkey.stages.inference.orchestrator.run_inference", return_value=result):
            worker = _InferenceWorker(
                preprocess_queue=in_q,
                inference_queue=out_q,
                model=MagicMock(),
                config=_make_config(tmp_path),
                remaining=[1],
                remaining_lock=threading.Lock(),
                events=events,
            )
            t = worker.start()
            t.join(timeout=5)

        assert queued == [3]

    def test_frame_error_fires_on_inference_exception(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        frame = _make_frame(7)
        in_q.put(_wrap(frame, tmp_path))
        in_q.put_stop()

        errors: list[tuple[str, int]] = []
        events = PipelineEvents(on_frame_error=lambda s, i, e: errors.append((s, i)))
        with patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=RuntimeError("boom")):
            worker = _InferenceWorker(
                preprocess_queue=in_q,
                inference_queue=out_q,
                model=MagicMock(),
                config=_make_config(tmp_path),
                remaining=[1],
                remaining_lock=threading.Lock(),
                events=events,
            )
            t = worker.start()
            t.join(timeout=5)

        assert len(errors) == 1
        stage, idx = errors[0]
        assert "inference" in stage
        assert idx == 7

    def test_queue_depth_fires_after_each_frame(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        frame = _make_frame(0)
        result = _make_result(frame)
        in_q.put(_wrap(frame, tmp_path))
        in_q.put_stop()

        depths: list[tuple[int, int]] = []
        events = PipelineEvents(on_queue_depth=lambda p, w: depths.append((p, w)))
        with patch("corridorkey.stages.inference.orchestrator.run_inference", return_value=result):
            worker = _InferenceWorker(
                preprocess_queue=in_q,
                inference_queue=out_q,
                model=MagicMock(),
                config=_make_config(tmp_path),
                remaining=[1],
                remaining_lock=threading.Lock(),
                events=events,
            )
            t = worker.start()
            t.join(timeout=5)

        assert len(depths) >= 1
