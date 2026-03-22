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

import torch.nn as nn

from corridorkey_new.events import PipelineEvents
from corridorkey_new.runtime.queue import BoundedQueue
from corridorkey_new.runtime.worker import InferenceWorker, PostWriteWorker, PreprocessWorker
from corridorkey_new.stages.inference import InferenceConfig
from corridorkey_new.stages.loader.contracts import ClipManifest
from corridorkey_new.stages.postprocessor import PostprocessConfig
from corridorkey_new.stages.preprocessor import PreprocessConfig
from corridorkey_new.stages.writer import WriteConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline runner.

    Attributes:
        preprocess: Preprocessing stage config (img_size, device, strategy).
        inference: Inference stage config (checkpoint, device, precision).
            None means inference is skipped (preprocess-only mode).
        postprocess: Postprocessing stage config (despill, despeckle, checkerboard).
        write: Writer stage config (formats, enabled outputs).
            None means a default WriteConfig is derived from the manifest output_dir.
        input_queue_depth: Max preprocessed frames waiting for inference.
            Keep small — each frame is ~64MB on GPU at 2048 resolution.
        output_queue_depth: Max inference outputs waiting for postprocessing.
    """

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    inference: InferenceConfig | None = None
    model: nn.Module | None = None
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    write: WriteConfig | None = None
    input_queue_depth: int = 2
    output_queue_depth: int = 2
    events: PipelineEvents | None = None
    """Optional event callbacks for all pipeline stages. Pass a PipelineEvents
    instance to receive per-frame and per-stage progress notifications."""


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
            postwrite_queue=output_queue,
            events=cfg.events,
        )
        if cfg.inference is None:
            raise ValueError(
                "PipelineConfig.inference is not set. "
                "Build an InferenceConfig and pass it to PipelineConfig, "
                "or call load_model() and pass the model to PipelineRunner."
            )
        inference_cfg: InferenceConfig = cfg.inference

        if cfg.model is None:
            from corridorkey_new.stages.inference import load_model

            logger.info("pipeline_runner: loading model from %s", inference_cfg.checkpoint_path)
            loaded_model = load_model(inference_cfg)
        else:
            loaded_model = cfg.model

        inference_worker = InferenceWorker(
            input_queue=input_queue,
            output_queue=output_queue,
            model=loaded_model,
            config=inference_cfg,
            events=cfg.events,
        )
        postwrite_worker = PostWriteWorker(
            input_queue=output_queue,
            output_dir=manifest.output_dir,
            postprocess_config=cfg.postprocess,
            write_config=cfg.write,
            total_frames=manifest.frame_count,
            events=cfg.events,
        )

        threads: list[threading.Thread] = [
            preprocess_worker.start(),
            inference_worker.start(),
            postwrite_worker.start(),
        ]

        for t in threads:
            t.join()

        logger.info("pipeline_runner: clip='%s' complete", manifest.clip_name)
