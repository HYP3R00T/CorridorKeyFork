"""Unit tests for corridorkey.events — PipelineEvents fire helpers."""

from __future__ import annotations

from pathlib import Path

from corridorkey.events import PipelineEvents


class TestPipelineEventsFireHelpers:
    def test_stage_start_fires_callback(self):
        calls: list[tuple[str, int]] = []
        e = PipelineEvents(on_stage_start=lambda s, t: calls.append((s, t)))
        e.stage_start("extract", 10)
        assert calls == [("extract", 10)]

    def test_stage_done_fires_callback(self):
        calls: list[str] = []
        e = PipelineEvents(on_stage_done=lambda s: calls.append(s))
        e.stage_done("extract")
        assert calls == ["extract"]

    def test_extract_frame_fires_callback(self):
        calls: list[tuple[int, int]] = []
        e = PipelineEvents(on_extract_frame=lambda i, t: calls.append((i, t)))
        e.extract_frame(3, 100)
        assert calls == [(3, 100)]

    def test_preprocess_queued_fires_callback(self):
        calls: list[int] = []
        e = PipelineEvents(on_preprocess_queued=lambda i: calls.append(i))
        e.preprocess_queued(5)
        assert calls == [5]

    def test_inference_start_fires_callback(self):
        calls: list[int] = []
        e = PipelineEvents(on_inference_start=lambda i: calls.append(i))
        e.inference_start(2)
        assert calls == [2]

    def test_inference_queued_fires_callback(self):
        calls: list[int] = []
        e = PipelineEvents(on_inference_queued=lambda i: calls.append(i))
        e.inference_queued(7)
        assert calls == [7]

    def test_frame_written_fires_callback(self):
        calls: list[tuple[int, int]] = []
        e = PipelineEvents(on_frame_written=lambda i, t: calls.append((i, t)))
        e.frame_written(4, 50)
        assert calls == [(4, 50)]

    def test_queue_depth_fires_callback(self):
        calls: list[tuple[int, int]] = []
        e = PipelineEvents(on_queue_depth=lambda p, w: calls.append((p, w)))
        e.queue_depth(2, 3)
        assert calls == [(2, 3)]

    def test_frame_error_fires_callback(self):
        calls: list[tuple[str, int]] = []
        err = ValueError("oops")
        e = PipelineEvents(on_frame_error=lambda s, i, ex: calls.append((s, i)))
        e.frame_error("inference", 1, err)
        assert calls == [("inference", 1)]

    def test_clip_found_fires_callback(self):
        calls: list[str] = []
        e = PipelineEvents(on_clip_found=lambda name, root: calls.append(name))
        e.clip_found("my_clip", Path("/some/path"))
        assert calls == ["my_clip"]

    def test_clip_skipped_fires_callback(self):
        calls: list[str] = []
        e = PipelineEvents(on_clip_skipped=lambda reason, path: calls.append(reason))
        e.clip_skipped("no input", Path("/some/path"))
        assert calls == ["no input"]

    def test_none_callbacks_do_not_raise(self):
        e = PipelineEvents()
        e.stage_start("extract", 0)
        e.stage_done("extract")
        e.extract_frame(0, 0)
        e.preprocess_queued(0)
        e.inference_start(0)
        e.inference_queued(0)
        e.frame_written(0, 0)
        e.queue_depth(0, 0)
        e.frame_error("stage", 0, Exception())
        e.clip_found("x", Path("."))
        e.clip_skipped("reason", Path("."))


class TestPipelineEventsConstruction:
    def test_all_fields_default_to_none(self):
        e = PipelineEvents()
        assert e.on_stage_start is None
        assert e.on_stage_done is None
        assert e.on_extract_frame is None
        assert e.on_preprocess_queued is None
        assert e.on_inference_start is None
        assert e.on_inference_queued is None
        assert e.on_frame_written is None
        assert e.on_queue_depth is None
        assert e.on_frame_error is None
        assert e.on_clip_found is None
        assert e.on_clip_skipped is None

    def test_callback_stored_when_provided(self):
        def cb(idx, total):
            pass

        e = PipelineEvents(on_frame_written=cb)
        assert e.on_frame_written is cb

    def test_multiple_callbacks_stored(self):
        def cb1(s, t):
            pass

        def cb2(s):
            pass

        e = PipelineEvents(on_stage_start=cb1, on_stage_done=cb2)
        assert e.on_stage_start is cb1
        assert e.on_stage_done is cb2

    def test_callback_called_multiple_times(self):
        calls: list[int] = []
        e = PipelineEvents(on_preprocess_queued=lambda i: calls.append(i))
        e.preprocess_queued(0)
        e.preprocess_queued(1)
        e.preprocess_queued(2)
        assert calls == [0, 1, 2]
