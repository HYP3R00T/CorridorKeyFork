"""Pipeline — worker threads.

Each worker runs in its own thread, pulling from an input queue and pushing
to an output queue. Workers exit cleanly when they receive the STOP sentinel.

Workers:
    PreprocessWorker   — reads frames from disk, preprocesses, pushes tensors
    InferenceWorker    — pulls tensors, runs model, pushes InferenceResult
    PostWriteWorker    — pulls InferenceResult, postprocesses, writes to disk
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch.nn as nn

from corridorkey_new.inference import InferenceConfig, InferenceResult, run_inference
from corridorkey_new.loader.contracts import ClipManifest
from corridorkey_new.loader.validator import get_frame_files
from corridorkey_new.pipeline.queue import STOP, BoundedQueue
from corridorkey_new.postprocessor import PostprocessConfig, postprocess_frame
from corridorkey_new.preprocessor import FrameReadError, PreprocessConfig, PreprocessedFrame, preprocess_frame
from corridorkey_new.writer import WriteConfig, write_frame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Preprocess worker
# ---------------------------------------------------------------------------


@dataclass
class PreprocessWorker:
    """Reads and preprocesses frames, pushing PreprocessedFrame onto the queue.

    Runs in its own thread. Iterates over the clip's frame_range, preprocesses
    each frame, and puts the result on ``output_queue``. Puts STOP when done
    or if a fatal error occurs.

    Attributes:
        manifest: ClipManifest for the clip being processed.
        config: Preprocessing configuration (img_size, device, strategy).
        output_queue: Queue to push PreprocessedFrame objects onto.
    """

    manifest: ClipManifest
    config: PreprocessConfig
    output_queue: BoundedQueue

    def run(self) -> None:
        """Entry point for the preprocess thread."""
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
                    self.output_queue.put(frame)
                    logger.debug("preprocess_worker: queued frame %d", i)
                except FrameReadError as e:
                    # Non-fatal — log and skip the frame.
                    logger.error("preprocess_worker: skipping frame %d — %s", i, e)
        finally:
            self.output_queue.put_stop()
            logger.debug("preprocess_worker: sent STOP")

    def start(self) -> threading.Thread:
        """Start the worker in a daemon thread and return it."""
        t = threading.Thread(target=self.run, name="preprocess-worker", daemon=True)
        t.start()
        return t


# ---------------------------------------------------------------------------
# Inference worker (stub)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Inference worker
# ---------------------------------------------------------------------------


@dataclass
class InferenceWorker:
    """Pulls PreprocessedFrame, runs model inference, pushes InferenceResult.

    Attributes:
        input_queue: Queue to pull PreprocessedFrame objects from.
        output_queue: Queue to push InferenceResult objects onto.
        model: Loaded GreenFormer in eval mode.
        config: Inference configuration (device, precision, optimization mode).
    """

    input_queue: BoundedQueue
    output_queue: BoundedQueue
    model: nn.Module
    config: InferenceConfig

    def run(self) -> None:
        """Entry point for the inference thread."""
        try:
            while True:
                item = self.input_queue.get()
                if item is STOP:
                    self.input_queue.put_stop()
                    break
                assert isinstance(item, PreprocessedFrame)
                try:
                    result = run_inference(item, self.model, self.config)
                    self.output_queue.put(result)
                    logger.debug("inference_worker: queued frame %d", item.meta.frame_index)
                except Exception as e:
                    logger.error("inference_worker: skipping frame %d — %s", item.meta.frame_index, e)
        finally:
            self.output_queue.put_stop()
            logger.debug("inference_worker: sent STOP")

    def start(self) -> threading.Thread:
        """Start the worker in a daemon thread and return it."""
        t = threading.Thread(target=self.run, name="inference-worker", daemon=True)
        t.start()
        return t


# ---------------------------------------------------------------------------
# Postprocess + write worker (stub)
# ---------------------------------------------------------------------------


@dataclass
class PostWriteWorker:
    """Pulls InferenceResult, postprocesses, and writes frames to disk.

    Attributes:
        input_queue: Queue to pull InferenceResult objects from.
        output_dir: Directory to write output frames into.
        postprocess_config: Postprocessing options (despill, despeckle, checkerboard).
        write_config: Writer options (formats, enabled outputs).
            If None, a default WriteConfig pointing at output_dir is used.
    """

    input_queue: BoundedQueue
    output_dir: Path
    postprocess_config: PostprocessConfig = field(default_factory=PostprocessConfig)
    write_config: WriteConfig | None = None

    def run(self) -> None:
        """Entry point for the postprocess+write thread."""
        write_cfg = self.write_config or WriteConfig(output_dir=self.output_dir)
        while True:
            item = self.input_queue.get()
            if item is STOP:
                self.input_queue.put_stop()
                break
            assert isinstance(item, InferenceResult)
            try:
                processed = postprocess_frame(item, self.postprocess_config)
                write_frame(processed, write_cfg)
                logger.debug("postwrite_worker: wrote frame %d", item.meta.frame_index)
            except Exception as e:
                logger.error("postwrite_worker: skipping frame %d — %s", item.meta.frame_index, e)

        logger.debug("postwrite_worker: done")

    def start(self) -> threading.Thread:
        """Start the worker in a daemon thread and return it."""
        t = threading.Thread(target=self.run, name="postwrite-worker", daemon=True)
        t.start()
        return t
