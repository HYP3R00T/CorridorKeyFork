"""Pipeline — event callbacks.

``PipelineEvents`` is a plain dataclass of optional callables. Any stage in
the pipeline can accept a ``PipelineEvents`` instance and fire the relevant
callback. Callers that don't care about a particular event simply leave it
as ``None``.

Design goals:
  - Zero dependencies on any UI framework.
  - All callbacks are optional — omitting one is a no-op, not an error.
  - Callbacks run on the thread that fires them (preprocess / inference /
    postwrite). Keep them fast; offload heavy work to a queue if needed.
  - A single ``PipelineEvents`` instance is shared across all stages so the
    caller gets a unified view of the whole pipeline.

Callback signatures
-------------------
on_stage_start(stage, total)
    A named stage is beginning. ``total`` is the frame count (0 if unknown).
    ``stage`` is one of: "extract", "preprocess", "inference", "postwrite".

on_stage_done(stage)
    A named stage has finished all its work.

on_extract_frame(frame_index, total)
    One frame has been written to disk during video extraction.

on_preprocess_queued(frame_index)
    A preprocessed frame has been pushed onto the inference input queue.

on_inference_start(frame_index)
    The inference worker has picked up a frame and started running the model.

on_inference_queued(frame_index)
    An inference result has been pushed onto the postwrite queue.

on_frame_written(frame_index, total)
    A frame has been fully postprocessed and written to disk.

on_queue_depth(preprocess_queue, postwrite_queue)
    Called whenever a frame moves between stages. Reports the current number
    of items waiting in each inter-stage queue. Useful for showing a live
    assembly-line view (e.g. "2 waiting for inference, 1 waiting to write").

on_frame_error(stage, frame_index, error)
    A frame was skipped due to an error in ``stage``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineEvents:
    """Container for all pipeline event callbacks.

    Every field is ``None`` by default — assign only the callbacks you need.

    Example::

        events = PipelineEvents(
            on_frame_written=lambda idx, total: print(f"  frame {idx + 1}/{total}"),
        )
        PipelineRunner(manifest, config, events=events).run()
    """

    # Stage-level events
    on_stage_start: Callable[[str, int], None] | None = field(default=None)
    on_stage_done: Callable[[str], None] | None = field(default=None)

    # Frame-level events
    on_extract_frame: Callable[[int, int], None] | None = field(default=None)
    on_preprocess_queued: Callable[[int], None] | None = field(default=None)
    on_inference_start: Callable[[int], None] | None = field(default=None)
    on_inference_queued: Callable[[int], None] | None = field(default=None)
    on_frame_written: Callable[[int, int], None] | None = field(default=None)

    # Queue depth snapshot — fires after every frame transition
    on_queue_depth: Callable[[int, int], None] | None = field(default=None)
    """on_queue_depth(preprocess_queue_depth, postwrite_queue_depth)

    Both values are point-in-time snapshots; treat them as approximate since
    worker threads may be draining concurrently.
    """

    # Error events
    on_frame_error: Callable[[str, int, Exception], None] | None = field(default=None)

    # Scan events
    on_clip_found: Callable[[str, Path], None] | None = field(default=None)
    """on_clip_found(clip_name, clip_root) — fired as each valid clip is discovered."""

    on_clip_skipped: Callable[[str, Path], None] | None = field(default=None)
    """on_clip_skipped(reason, path) — fired when a path is skipped during scanning."""

    # ---------------------------------------------------------------------------
    # Convenience fire helpers — check for None so call sites stay clean
    # ---------------------------------------------------------------------------

    def stage_start(self, stage: str, total: int = 0) -> None:
        if self.on_stage_start:
            self.on_stage_start(stage, total)

    def stage_done(self, stage: str) -> None:
        if self.on_stage_done:
            self.on_stage_done(stage)

    def extract_frame(self, frame_index: int, total: int) -> None:
        if self.on_extract_frame:
            self.on_extract_frame(frame_index, total)

    def preprocess_queued(self, frame_index: int) -> None:
        if self.on_preprocess_queued:
            self.on_preprocess_queued(frame_index)

    def inference_start(self, frame_index: int) -> None:
        if self.on_inference_start:
            self.on_inference_start(frame_index)

    def inference_queued(self, frame_index: int) -> None:
        if self.on_inference_queued:
            self.on_inference_queued(frame_index)

    def frame_written(self, frame_index: int, total: int) -> None:
        if self.on_frame_written:
            self.on_frame_written(frame_index, total)

    def queue_depth(self, preprocess_q: int, postwrite_q: int) -> None:
        if self.on_queue_depth:
            self.on_queue_depth(preprocess_q, postwrite_q)

    def frame_error(self, stage: str, frame_index: int, error: Exception) -> None:
        if self.on_frame_error:
            self.on_frame_error(stage, frame_index, error)

    def clip_found(self, clip_name: str, clip_root: Path) -> None:
        if self.on_clip_found:
            self.on_clip_found(clip_name, clip_root)

    def clip_skipped(self, reason: str, path: Path) -> None:
        if self.on_clip_skipped:
            self.on_clip_skipped(reason, path)
