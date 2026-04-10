"""Pipeline — worker threads.

Each worker runs in its own thread, pulling from an input queue and pushing
to an output queue. Workers exit cleanly when they receive the STOP sentinel
or when the shared ``cancel_event`` is set.

Workers:
    PreprocessWorker   — reads frames from disk, preprocesses, pushes tensors
    PostWriteWorker    — pulls InferenceResult, postprocesses, writes to disk

The inference worker lives in runner.py as ``_InferenceWorker`` because it
needs the shared ``_AtomicCounter`` for coordinated shutdown across N devices.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

from corridorkey.events import PipelineEvents
from corridorkey.runtime.queue import STOP, BoundedQueue
from corridorkey.stages.inference import InferenceResult
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.loader.validator import list_frames
from corridorkey.stages.postprocessor import PostprocessConfig, postprocess_frame
from corridorkey.stages.preprocessor import FrameReadError, PreprocessConfig, preprocess_frame
from corridorkey.stages.writer import WriteConfig, write_frame

logger = logging.getLogger(__name__)


@dataclass
class PreprocessWorker:
    """Reads and preprocesses frames, pushing PreprocessedFrame onto the queue.

    Attributes:
        manifest: ClipManifest for the clip being processed.
        config: Preprocessing configuration (img_size, device, strategy).
        preprocess_queue: Queue to push PreprocessedFrame objects onto.
        inference_queue: Inference queue reference — used only for queue depth
            snapshots fired via events.
        events: Optional pipeline event callbacks.
        cancel_event: Optional event that signals the worker to stop early.
            When set, the worker exits after the current frame and sends STOP
            downstream so the rest of the pipeline drains cleanly.
    """

    manifest: ClipManifest
    config: PreprocessConfig
    preprocess_queue: BoundedQueue
    inference_queue: BoundedQueue | None = None
    events: PipelineEvents | None = None
    cancel_event: threading.Event | None = None

    def run(self) -> None:
        """Entry point for the preprocess thread."""
        total = self.manifest.frame_count
        if self.events:
            self.events.stage_start("preprocess", total)

        image_files = list_frames(self.manifest.frames_dir)
        alpha_files = list_frames(self.manifest.alpha_frames_dir)  # type: ignore[arg-type]

        try:
            for i in range(*self.manifest.frame_range):
                if self.cancel_event and self.cancel_event.is_set():
                    logger.debug("preprocess_worker: cancelled at frame %d", i)
                    break
                try:
                    frame = preprocess_frame(
                        self.manifest,
                        i,
                        self.config,
                        image_files=image_files,
                        alpha_files=alpha_files,
                    )
                    # Use put_unless_cancelled so a full queue doesn't block
                    # indefinitely when the user cancels mid-run.
                    if self.cancel_event is not None:
                        enqueued = self.preprocess_queue.put_unless_cancelled(frame, self.cancel_event)
                        if not enqueued:
                            logger.debug("preprocess_worker: cancelled while waiting to enqueue frame %d", i)
                            break
                    else:
                        self.preprocess_queue.put(frame)
                    logger.debug("preprocess_worker: queued frame %d", i)
                    if self.events:
                        self.events.preprocess_queued(i)
                        self.events.queue_depth(
                            len(self.preprocess_queue),
                            len(self.inference_queue) if self.inference_queue else 0,
                        )
                except FrameReadError as e:
                    logger.error("preprocess_worker: skipping frame %d — %s", i, e)
                    if self.events:
                        self.events.frame_error("preprocess", i, e)
        finally:
            self.preprocess_queue.put_stop()
            logger.debug("preprocess_worker: sent STOP")
            if self.events:
                self.events.stage_done("preprocess")

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name="preprocess-worker", daemon=True)
        t.start()
        return t


@dataclass
class PostWriteWorker:
    """Pulls InferenceResult, postprocesses, and writes frames to disk.

    Attributes:
        inference_queue: Queue to pull InferenceResult objects from.
        output_dir: Directory to write output frames into.
        postprocess_config: Postprocessing options (despill, despeckle, checkerboard).
        write_config: Writer options (formats, enabled outputs).
            If None, a default WriteConfig pointing at output_dir is used.
        total_frames: Total frame count passed to frame_written events.
        events: Optional pipeline event callbacks.
        cancel_event: Optional event that signals the worker to stop early.
            When set, the worker exits after the current frame and sends STOP
            downstream so the rest of the pipeline drains cleanly.
    """

    inference_queue: BoundedQueue
    output_dir: Path
    postprocess_config: PostprocessConfig = field(default_factory=PostprocessConfig)
    write_config: WriteConfig | None = None
    total_frames: int = 0
    events: PipelineEvents | None = None
    cancel_event: threading.Event | None = None

    def run(self) -> None:
        """Entry point for the postprocess+write thread."""
        if self.events:
            self.events.stage_start("postwrite", self.total_frames)
        write_cfg = self.write_config or WriteConfig(output_dir=self.output_dir)
        while True:
            if self.cancel_event and self.cancel_event.is_set():
                logger.debug("postwrite_worker: cancelled, draining queue")
                self.inference_queue.put_stop()
                break
            item = self.inference_queue.get()
            if item is STOP:
                self.inference_queue.put_stop()
                break
            assert isinstance(item, InferenceResult)
            try:
                processed = postprocess_frame(item, self.postprocess_config, output_dir=self.output_dir)
                write_frame(processed, write_cfg)
                logger.debug("postwrite_worker: wrote frame %d", item.meta.frame_index)
                if self.events:
                    self.events.frame_written(item.meta.frame_index, self.total_frames)
            except Exception as e:
                logger.error("postwrite_worker: skipping frame %d — %s", item.meta.frame_index, e)
                if self.events:
                    self.events.frame_error("postwrite", item.meta.frame_index, e)

        logger.debug("postwrite_worker: done")
        if self.events:
            self.events.stage_done("postwrite")

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name="postwrite-worker", daemon=True)
        t.start()
        return t
