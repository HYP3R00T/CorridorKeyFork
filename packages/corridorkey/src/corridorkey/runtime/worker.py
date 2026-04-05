"""Pipeline — worker threads.

Each worker runs in its own thread, pulling from an input queue and pushing
to an output queue. Workers exit cleanly when they receive the STOP sentinel.

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
from corridorkey.stages.loader.contracts import LoadResult
from corridorkey.stages.loader.validator import get_frame_files
from corridorkey.stages.postprocessor import PostprocessConfig, postprocess_frame
from corridorkey.stages.preprocessor import FrameReadError, PreprocessConfig, preprocess_frame
from corridorkey.stages.writer import WriteConfig, write_frame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocess worker
# ---------------------------------------------------------------------------


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
    """

    manifest: LoadResult
    config: PreprocessConfig
    preprocess_queue: BoundedQueue
    inference_queue: BoundedQueue | None = None
    events: PipelineEvents | None = None

    def run(self) -> None:
        """Entry point for the preprocess thread."""
        total = self.manifest.frame_count
        if self.events:
            self.events.stage_start("preprocess", total)

        image_files = get_frame_files(self.manifest.frames_dir)
        alpha_files = get_frame_files(self.manifest.alpha_frames_dir)  # type: ignore[arg-type]

        try:
            for i in range(*self.manifest.frame_range):
                try:
                    frame = preprocess_frame(
                        self.manifest,
                        i,
                        self.config,
                        image_files=image_files,
                        alpha_files=alpha_files,
                    )
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


# ---------------------------------------------------------------------------
# Postprocess + write worker
# ---------------------------------------------------------------------------


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
    """

    inference_queue: BoundedQueue
    output_dir: Path
    postprocess_config: PostprocessConfig = field(default_factory=PostprocessConfig)
    write_config: WriteConfig | None = None
    total_frames: int = 0
    events: PipelineEvents | None = None

    def run(self) -> None:
        """Entry point for the postprocess+write thread."""
        if self.events:
            self.events.stage_start("postwrite", self.total_frames)
        write_cfg = self.write_config or WriteConfig(output_dir=self.output_dir)
        while True:
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
