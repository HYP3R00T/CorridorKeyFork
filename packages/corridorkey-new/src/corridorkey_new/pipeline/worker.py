"""Pipeline — worker threads.

Each worker runs in its own thread, pulling from an input queue and pushing
to an output queue. Workers exit cleanly when they receive the STOP sentinel.

Workers:
    PreprocessWorker   — reads frames from disk, preprocesses, pushes tensors
    InferenceWorker    — pulls tensors, runs model, pushes raw outputs   [stub]
    PostWriteWorker    — pulls raw outputs, postprocesses, writes to disk [stub]

Stubs are in place so the pipeline wiring is complete. They will be filled in
when inference and postprocessing stages are built.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corridorkey_new.loader.contracts import ClipManifest
from corridorkey_new.loader.validator import get_frame_files
from corridorkey_new.pipeline.queue import STOP, BoundedQueue
from corridorkey_new.preprocessor import FrameReadError, PreprocessConfig, preprocess_frame

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


@dataclass
class InferenceWorker:
    """Pulls PreprocessedFrame, runs model inference, pushes raw outputs.

    Stub — will be implemented when the inference stage is built.

    Attributes:
        input_queue: Queue to pull PreprocessedFrame objects from.
        output_queue: Queue to push inference outputs onto.
        model: The inference model (type TBD when inference stage is built).
    """

    input_queue: BoundedQueue
    output_queue: BoundedQueue
    model: Any  # replaced with the real model type when inference is built

    def run(self) -> None:
        """Entry point for the inference thread."""
        try:
            while True:
                item = self.input_queue.get()
                if item is STOP:
                    self.input_queue.put_stop()  # propagate to next consumer
                    break
                # TODO: run inference on item, push result to output_queue
                logger.debug("inference_worker: received frame (stub — not yet implemented)")
                self.output_queue.put(item)  # pass-through until implemented
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
    """Pulls inference outputs, postprocesses, and writes frames to disk.

    Stub — will be implemented when postprocessing and writing stages are built.

    Attributes:
        input_queue: Queue to pull inference outputs from.
        output_dir: Directory to write output frames into.
    """

    input_queue: BoundedQueue
    output_dir: Path

    def run(self) -> None:
        """Entry point for the postprocess+write thread."""
        while True:
            item = self.input_queue.get()
            if item is STOP:
                self.input_queue.put_stop()  # propagate sentinel
                break
            # TODO: postprocess item, write outputs to output_dir
            logger.debug("postwrite_worker: received frame (stub — not yet implemented)")

        logger.debug("postwrite_worker: done")

    def start(self) -> threading.Thread:
        """Start the worker in a daemon thread and return it."""
        t = threading.Thread(target=self.run, name="postwrite-worker", daemon=True)
        t.start()
        return t
