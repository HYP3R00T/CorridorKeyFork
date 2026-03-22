"""Unit tests for MultiGPURunner and related helpers."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch
from corridorkey_new.runtime.queue import STOP, BoundedQueue
from corridorkey_new.runtime.runner import (
    MultiGPUConfig,
    MultiGPURunner,
    _AtomicCounter,
    _MultiGPUInferenceWorker,
    _override_device,
)
from corridorkey_new.stages.inference import InferenceConfig, InferenceResult
from corridorkey_new.stages.loader.contracts import ClipManifest
from corridorkey_new.stages.preprocessor import FrameMeta, PreprocessConfig, PreprocessedFrame

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inference_config(tmp_path: Path) -> InferenceConfig:
    return InferenceConfig(checkpoint_path=tmp_path / "model.pth", device="cpu")


def _make_fake_frame(idx: int = 0) -> PreprocessedFrame:
    meta = FrameMeta(frame_index=idx, original_h=32, original_w=32)
    return PreprocessedFrame(tensor=torch.zeros(1, 4, 32, 32), meta=meta)


def _make_fake_result(frame: PreprocessedFrame) -> InferenceResult:
    return InferenceResult(
        alpha=torch.zeros(1, 1, 32, 32),
        fg=torch.zeros(1, 3, 32, 32),
        meta=frame.meta,
    )


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
        """Concurrent decrements should each return a unique value."""
        n = 50
        c = _AtomicCounter(n)
        results = []
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
        new_cfg = _override_device(cfg, "cuda:1")
        assert new_cfg.device == "cuda:1"

    def test_other_fields_unchanged(self, tmp_path: Path):
        cfg = _make_inference_config(tmp_path)
        new_cfg = _override_device(cfg, "cuda:1")
        assert new_cfg.checkpoint_path == cfg.checkpoint_path
        assert new_cfg.img_size == cfg.img_size
        assert new_cfg.use_refiner == cfg.use_refiner

    def test_original_config_unchanged(self, tmp_path: Path):
        cfg = _make_inference_config(tmp_path)
        _override_device(cfg, "cuda:1")
        assert cfg.device == "cpu"


# ---------------------------------------------------------------------------
# _MultiGPUInferenceWorker
# ---------------------------------------------------------------------------


class TestMultiGPUInferenceWorker:
    def test_processes_frames_and_sends_stop_when_last(self, tmp_path: Path):
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        frame = _make_fake_frame(0)
        result = _make_fake_result(frame)
        in_q.put(frame)
        in_q.put_stop()

        counter = _AtomicCounter(1)  # only one worker
        with patch("corridorkey_new.stages.inference.orchestrator.run_inference", return_value=result):
            worker = _MultiGPUInferenceWorker(
                input_queue=in_q,
                output_queue=out_q,
                model=MagicMock(),
                config=_make_inference_config(tmp_path),
                active_workers=counter,
                worker_index=0,
            )
            t = worker.start()
            t.join(timeout=5)

        assert out_q.get() is result
        assert out_q.get() is STOP  # last worker sends STOP downstream

    def test_non_last_worker_does_not_send_stop(self, tmp_path: Path):
        """When active_workers > 0 after decrement, STOP is NOT sent downstream."""
        in_q: BoundedQueue = BoundedQueue(10)
        out_q: BoundedQueue = BoundedQueue(10)
        in_q.put_stop()

        counter = _AtomicCounter(2)  # two workers — this one is not last
        worker = _MultiGPUInferenceWorker(
            input_queue=in_q,
            output_queue=out_q,
            model=MagicMock(),
            config=_make_inference_config(tmp_path),
            active_workers=counter,
            worker_index=0,
        )
        t = worker.start()
        t.join(timeout=5)

        # out_q should be empty — no STOP sent
        assert len(out_q) == 0

    def test_two_workers_share_queue_all_frames_processed(self, tmp_path: Path):
        """Two workers pulling from the same queue process all frames exactly once."""
        in_q: BoundedQueue = BoundedQueue(20)
        out_q: BoundedQueue = BoundedQueue(20)

        n_frames = 6
        frames = [_make_fake_frame(i) for i in range(n_frames)]
        results = [_make_fake_result(f) for f in frames]
        for f in frames:
            in_q.put(f)
        in_q.put_stop()

        counter = _AtomicCounter(2)
        call_map = {id(f): r for f, r in zip(frames, results, strict=True)}

        def side_effect(frame, model, config):
            return call_map[id(frame)]

        with patch("corridorkey_new.stages.inference.orchestrator.run_inference", side_effect=side_effect):
            for i in range(2):
                w = _MultiGPUInferenceWorker(
                    input_queue=in_q,
                    output_queue=out_q,
                    model=MagicMock(),
                    config=_make_inference_config(tmp_path),
                    active_workers=counter,
                    worker_index=i,
                )
                w.start()

        # Collect all results
        received = []
        while True:
            item = out_q.get()
            if item is STOP:
                break
            received.append(item)

        assert len(received) == n_frames
        assert {r.meta.frame_index for r in received} == set(range(n_frames))


# ---------------------------------------------------------------------------
# MultiGPUConfig validation
# ---------------------------------------------------------------------------


class TestMultiGPUConfig:
    def test_empty_devices_raises(self, tmp_path: Path):
        cfg = MultiGPUConfig(
            devices=[],
            inference=_make_inference_config(tmp_path),
        )
        manifest = _make_manifest(tmp_path, frame_count=1)
        with pytest.raises(ValueError, match="at least one entry"):
            MultiGPURunner(manifest=manifest, config=cfg).run()


# ---------------------------------------------------------------------------
# MultiGPURunner integration (mocked model loading + inference)
# ---------------------------------------------------------------------------


class TestMultiGPURunner:
    def test_single_device_processes_all_frames(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=3)
        inference_cfg = InferenceConfig(
            checkpoint_path=tmp_path / "model.pth",
            device="cpu",
            img_size=512,
        )
        cfg = MultiGPUConfig(
            devices=["cpu"],
            inference=inference_cfg,
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
        )

        fake_model = MagicMock()

        def fake_run_inference(frame, model, config):
            return _make_fake_result(frame)

        with (
            patch("corridorkey_new.stages.inference.loader.load_model", return_value=fake_model),
            patch("corridorkey_new.stages.inference.orchestrator.run_inference", side_effect=fake_run_inference),
            patch("corridorkey_new.runtime.worker.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey_new.runtime.worker.write_frame"),
        ):
            runner = MultiGPURunner(manifest=manifest, config=cfg)
            runner.run()  # should complete without error

    def test_two_devices_process_all_frames(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=4)
        inference_cfg = InferenceConfig(
            checkpoint_path=tmp_path / "model.pth",
            device="cpu",
            img_size=512,
        )
        cfg = MultiGPUConfig(
            devices=["cpu", "cpu"],
            inference=inference_cfg,
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
        )

        fake_model = MagicMock()

        def fake_run_inference(frame, model, config):
            return _make_fake_result(frame)

        with (
            patch("corridorkey_new.stages.inference.loader.load_model", return_value=fake_model),
            patch("corridorkey_new.stages.inference.orchestrator.run_inference", side_effect=fake_run_inference),
            patch("corridorkey_new.runtime.worker.postprocess_frame", return_value=MagicMock()),
            patch("corridorkey_new.runtime.worker.write_frame"),
        ):
            runner = MultiGPURunner(manifest=manifest, config=cfg)
            runner.run()

    def test_model_load_failure_raises(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=1)
        inference_cfg = InferenceConfig(
            checkpoint_path=tmp_path / "model.pth",
            device="cpu",
            img_size=512,
        )
        cfg = MultiGPUConfig(
            devices=["cpu", "cpu"],
            inference=inference_cfg,
            preprocess=PreprocessConfig(img_size=512, device="cpu"),
        )

        with (
            patch("corridorkey_new.stages.inference.loader.load_model", side_effect=RuntimeError("no checkpoint")),
            pytest.raises(RuntimeError, match="Failed to load model"),
        ):
            MultiGPURunner(manifest=manifest, config=cfg).run()
