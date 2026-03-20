"""Pipeline — runner.

PipelineRunner wires the queues and workers together and runs the full
pipeline for a single clip. This is what main.py (and eventually the CLI/GUI)
calls instead of a manual frame loop.

Assembly line:

    PreprocessWorker
        → input_queue (bounded)
            → InferenceWorker
                → output_queue (bounded)
                    → PostWriteWorker

All workers run concurrently in daemon threads. The runner blocks until all
workers have finished and all queues are drained.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

from corridorkey_new.loader.contracts import ClipManifest
from corridorkey_new.pipeline.queue import BoundedQueue
from corridorkey_new.pipeline.worker import InferenceWorker, PostWriteWorker, PreprocessWorker
from corridorkey_new.preprocessor import PreprocessConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline runner.

    Attributes:
        preprocess: Preprocessing stage config (img_size, device, strategy).
        input_queue_depth: Max preprocessed frames waiting for inference.
            Keep small — each frame is ~64MB on GPU at 2048 resolution.
        output_queue_depth: Max inference outputs waiting for postprocessing.
    """

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    input_queue_depth: int = 2
    output_queue_depth: int = 2


class PipelineRunner:
    """Runs the full pipeline for a single clip.

    Instantiate once per clip, call ``run()``, then discard.

    Args:
        manifest: ClipManifest from the loader stage. Must have needs_alpha=False.
        config: Pipeline configuration.
    """

    def __init__(self, manifest: ClipManifest, config: PipelineConfig) -> None:
        self._manifest = manifest
        self._config = config

    def run(self) -> None:
        """Run the pipeline for the clip. Blocks until all frames are done."""
        cfg = self._config
        manifest = self._manifest

        logger.info(
            "pipeline_runner: starting clip='%s' frames=%d device=%s",
            manifest.clip_name,
            manifest.frame_count,
            cfg.preprocess.device,
        )

        input_queue: BoundedQueue = BoundedQueue(cfg.input_queue_depth)
        output_queue: BoundedQueue = BoundedQueue(cfg.output_queue_depth)

        preprocess_worker = PreprocessWorker(
            manifest=manifest,
            config=cfg.preprocess,
            output_queue=input_queue,
        )
        inference_worker = InferenceWorker(
            input_queue=input_queue,
            output_queue=output_queue,
            model=None,  # replaced when inference stage is built
        )
        postwrite_worker = PostWriteWorker(
            input_queue=output_queue,
            output_dir=manifest.output_dir,
        )

        threads: list[threading.Thread] = [
            preprocess_worker.start(),
            inference_worker.start(),
            postwrite_worker.start(),
        ]

        for t in threads:
            t.join()

        logger.info("pipeline_runner: clip='%s' complete", manifest.clip_name)
