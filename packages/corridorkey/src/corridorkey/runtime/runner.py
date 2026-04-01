"""Pipeline — runner.

PipelineRunner wires the queues and workers together and runs the full
pipeline for a single clip. This is what the CLI/GUI calls instead of a
manual frame loop.

Single-GPU assembly line:

    PreprocessWorker
        -> input_queue (bounded)
            -> InferenceWorker
                -> output_queue (bounded)
                    -> PostWriteWorker

Multi-GPU assembly line (MultiGPURunner):

    PreprocessWorker
        -> input_queue (bounded, shared)
            -> InferenceWorker[cuda:0]  -+
            -> InferenceWorker[cuda:1]  -+-> output_queue (bounded, shared)
            -> InferenceWorker[cuda:N]  -+       -> PostWriteWorker

All workers run concurrently in daemon threads. The runner blocks until all
workers have finished and all queues are drained.

Public entry point
------------------
Use ``Runner`` for all new code. It accepts a ``PipelineConfig`` and
dispatches to single-GPU or multi-GPU execution automatically based on the
``devices`` field. ``PipelineRunner`` and ``MultiGPURunner`` remain available
for callers that need direct control.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import torch.nn as nn

from corridorkey.events import PipelineEvents
from corridorkey.runtime.queue import BoundedQueue
from corridorkey.runtime.worker import InferenceWorker, PostWriteWorker, PreprocessWorker
from corridorkey.stages.inference import InferenceConfig
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.postprocessor import PostprocessConfig
from corridorkey.stages.preprocessor import PreprocessConfig
from corridorkey.stages.writer import WriteConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline runner.

    Pass to :class:`Runner` (recommended) or directly to
    :class:`PipelineRunner` / :class:`MultiGPURunner`.

    Attributes:
        preprocess: Preprocessing stage config (img_size, device, strategy).
        inference: Inference stage config (checkpoint, device, precision).
            None means inference is skipped (preprocess-only mode).
        postprocess: Postprocessing stage config (despill, despeckle, checkerboard).
        write: Writer stage config (formats, enabled outputs).
            None means a default WriteConfig is derived from the manifest output_dir.
        devices: Device strings for multi-GPU dispatch. When set to more than
            one entry, ``Runner`` uses ``MultiGPURunner`` automatically.
            Leave empty (default) for single-GPU or CPU execution.
        events: Optional event callbacks for progress reporting. Assign only
            the callbacks you need — all others are silently ignored.
        input_queue_depth: Max preprocessed frames waiting for inference.
            Keep small — each frame is ~64 MB on GPU at 2048 resolution.
        output_queue_depth: Max inference outputs waiting for postprocessing.
        resolved_refiner_mode: Pre-resolved refiner mode ("full_frame" or "tiled").
            Populated by ``to_pipeline_config()`` — do not set manually.
        model: Pre-loaded model (``nn.Module``). If None, the runner loads it
            from the checkpoint path at run time.
    """

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    inference: InferenceConfig | None = None
    model: nn.Module | None = None
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    write: WriteConfig | None = None
    devices: list[str] = field(default_factory=list)
    events: PipelineEvents | None = None
    input_queue_depth: int = 2
    output_queue_depth: int = 2
    resolved_refiner_mode: str | None = None


class PipelineRunner:
    """Runs the full pipeline for a single clip on a single device.

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

        # Resolve refiner_mode once — use the pre-resolved value from PipelineConfig
        # when available (set by to_pipeline_config, avoids a second VRAM probe).
        # Fall back to probing only when a model is loaded on-demand here.
        if cfg.resolved_refiner_mode is not None:
            resolved_refiner_mode = cfg.resolved_refiner_mode
        else:
            from corridorkey.stages.inference.orchestrator import _should_tile_refiner

            resolved_refiner_mode = "tiled" if _should_tile_refiner(inference_cfg) else "full_frame"

        if cfg.model is None:
            from corridorkey.stages.inference import load_model

            logger.info(
                "pipeline_runner: loading model from %s (refiner_mode=%s)",
                inference_cfg.checkpoint_path,
                resolved_refiner_mode,
            )
            loaded_model = load_model(inference_cfg, resolved_refiner_mode=resolved_refiner_mode)
        else:
            loaded_model = cfg.model

        inference_worker = InferenceWorker(
            input_queue=input_queue,
            output_queue=output_queue,
            model=loaded_model,
            config=inference_cfg,
            resolved_refiner_mode=resolved_refiner_mode,
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


class Runner:
    """Unified pipeline runner. Dispatches to single-GPU or multi-GPU execution
    automatically based on ``config.devices``.

    This is the recommended entry point for all interfaces. Use
    :class:`PipelineRunner` or :class:`MultiGPURunner` directly only when you
    need explicit control over the dispatch strategy.

    Rules:
        - ``config.devices`` empty or one entry  -> :class:`PipelineRunner`
        - ``config.devices`` two or more entries -> :class:`MultiGPURunner`

    Args:
        manifest: ClipManifest from the loader stage. Must have needs_alpha=False.
        config: Pipeline configuration. Build via
            ``CorridorKeyConfig.to_pipeline_config()``.
        events: Optional event callbacks. Overrides ``config.events`` when
            provided, so the same config can be reused with different handlers.
    """

    def __init__(
        self,
        manifest: ClipManifest,
        config: PipelineConfig,
        events: PipelineEvents | None = None,
    ) -> None:
        self._manifest = manifest
        self._config = config
        # events kwarg takes precedence over config.events so a config object
        # can be reused across clips with different progress handlers.
        self._events = events if events is not None else config.events

    def run(self) -> None:
        """Run the pipeline for the clip. Blocks until all frames are written."""
        cfg = self._config

        if len(cfg.devices) > 1:
            # Multi-GPU path
            if cfg.inference is None:
                raise ValueError("PipelineConfig.inference must be set for multi-GPU execution.")
            multi_cfg = MultiGPUConfig(
                devices=cfg.devices,
                inference=cfg.inference,
                preprocess=cfg.preprocess,
                postprocess=cfg.postprocess,
                write=cfg.write,
                input_queue_depth=cfg.input_queue_depth,
                output_queue_depth=cfg.output_queue_depth,
                events=self._events,
            )
            MultiGPURunner(self._manifest, multi_cfg).run()
        else:
            # Single-GPU path — inject resolved events
            single_cfg = cfg
            if self._events is not cfg.events:
                from dataclasses import replace

                single_cfg = replace(cfg, events=self._events)
            PipelineRunner(self._manifest, single_cfg).run()


@dataclass
class MultiGPUConfig:
    """Configuration for multi-GPU frame-level parallel inference.

    Each device gets its own model instance and InferenceWorker thread.
    All workers share a single input queue (preprocessed frames) and a
    single output queue (inference results), so the pipeline naturally
    load-balances — whichever GPU finishes first picks up the next frame.

    Attributes:
        devices: List of PyTorch device strings to use (e.g. ["cuda:0", "cuda:1"]).
            Must have at least one entry. Use ``resolve_devices("all")`` to
            populate this from all available CUDA GPUs.
        inference: Base InferenceConfig. The ``device`` field is overridden
            per-worker — all other fields (checkpoint, precision, etc.) are
            shared across all GPUs.
        preprocess: Preprocessing config. Runs on CPU (device-agnostic).
        postprocess: Postprocessing config.
        write: Writer config. None → default derived from manifest output_dir.
        input_queue_depth: Shared input queue depth. Scale with GPU count —
            a depth of 2×N gives each GPU a frame in flight plus one buffered.
        output_queue_depth: Shared output queue depth.
        events: Optional pipeline event callbacks.
    """

    devices: list[str]
    inference: InferenceConfig
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    write: WriteConfig | None = None
    input_queue_depth: int = 4
    output_queue_depth: int = 4
    events: PipelineEvents | None = None


class MultiGPURunner:
    """Runs the pipeline across multiple GPUs in parallel (frame-level dispatch).

    Each GPU gets its own model instance loaded in its own thread. All GPU
    workers pull from a single shared input queue and push to a single shared
    output queue. The preprocessor and postwriter are single-threaded as usual.

    Frame ordering: output frames may arrive out of order when multiple GPUs
    are running at different speeds. The PostWriteWorker writes frames as they
    arrive — if strict ordering is required, the caller should sort by
    ``InferenceResult.meta.frame_index`` after the run.

    Args:
        manifest: ClipManifest from the loader stage.
        config: MultiGPUConfig with a list of device strings.
    """

    def __init__(self, manifest: ClipManifest, config: MultiGPUConfig) -> None:
        if not config.devices:
            raise ValueError("MultiGPUConfig.devices must have at least one entry.")
        self._manifest = manifest
        self._config = config

    def run(self) -> None:
        """Run the pipeline. Blocks until all frames are written."""
        cfg = self._config
        manifest = self._manifest
        n_gpus = len(cfg.devices)

        logger.info(
            "multi_gpu_runner: starting clip='%s' frames=%d gpus=%d devices=%s",
            manifest.clip_name,
            manifest.frame_count,
            n_gpus,
            cfg.devices,
        )

        # Scale queue depth with GPU count so each GPU always has work buffered.
        input_depth = max(cfg.input_queue_depth, n_gpus * 2)
        output_depth = max(cfg.output_queue_depth, n_gpus * 2)

        input_queue: BoundedQueue = BoundedQueue(input_depth)
        output_queue: BoundedQueue = BoundedQueue(output_depth)

        # Preprocess runs on CPU — device field is informational only here.
        preprocess_worker = PreprocessWorker(
            manifest=manifest,
            config=cfg.preprocess,
            output_queue=input_queue,
            postwrite_queue=output_queue,
            events=cfg.events,
        )

        # Load one model per GPU in parallel to minimise startup time.
        models, resolved_refiner_mode = self._load_models_parallel(cfg.devices, cfg.inference)

        # One InferenceWorker per GPU — all share the same input/output queues.
        # The STOP sentinel propagation pattern: the last worker to see STOP
        # re-puts it so the next consumer (PostWriteWorker) also sees it.
        # With N workers all reading from the same queue, we need N STOPs so
        # each worker gets one. The PreprocessWorker puts exactly one STOP.
        # Solution: each InferenceWorker re-puts STOP when it sees it, but only
        # the last one should propagate downstream. We use a shared counter.
        active_workers = _AtomicCounter(n_gpus)

        inference_threads: list[threading.Thread] = []
        for i, (device, model) in enumerate(zip(cfg.devices, models, strict=True)):
            # Build a per-device config by overriding the device field.
            device_cfg = _override_device(cfg.inference, device)
            worker = _MultiGPUInferenceWorker(
                input_queue=input_queue,
                output_queue=output_queue,
                model=model,
                config=device_cfg,
                resolved_refiner_mode=resolved_refiner_mode,
                active_workers=active_workers,
                worker_index=i,
                events=cfg.events,
            )
            inference_threads.append(worker.start())

        postwrite_worker = PostWriteWorker(
            input_queue=output_queue,
            output_dir=manifest.output_dir,
            postprocess_config=cfg.postprocess,
            write_config=cfg.write,
            total_frames=manifest.frame_count,
            events=cfg.events,
        )

        all_threads = [preprocess_worker.start(), *inference_threads, postwrite_worker.start()]
        for t in all_threads:
            t.join()

        logger.info("multi_gpu_runner: clip='%s' complete", manifest.clip_name)

    def _load_models_parallel(self, devices: list[str], base_config: InferenceConfig) -> tuple[list[nn.Module], str]:
        """Load one model per device in parallel threads.

        Resolves refiner_mode once (using the first device as representative)
        before spawning threads so torch.compile decisions are made on the
        concrete mode, not "auto".

        Returns:
            Tuple of (models in device order, resolved_refiner_mode string).
        """
        from corridorkey.stages.inference.loader import load_model
        from corridorkey.stages.inference.orchestrator import _should_tile_refiner

        # Resolve once using the base config (device doesn't affect mode resolution
        # for VRAM probing — we use the first device as representative).
        resolved_refiner_mode = "tiled" if _should_tile_refiner(base_config) else "full_frame"
        logger.info("multi_gpu_runner: resolved refiner_mode=%s", resolved_refiner_mode)

        models: list[nn.Module | None] = [None] * len(devices)
        errors: list[Exception | None] = [None] * len(devices)

        def _load(idx: int, device: str) -> None:
            try:
                device_cfg = _override_device(base_config, device)
                logger.info("multi_gpu_runner: loading model on %s (worker %d)", device, idx)
                models[idx] = load_model(device_cfg, resolved_refiner_mode=resolved_refiner_mode)
                logger.info("multi_gpu_runner: model ready on %s", device)
            except Exception as e:
                errors[idx] = e
                logger.error("multi_gpu_runner: failed to load model on %s — %s", device, e)

        threads = [threading.Thread(target=_load, args=(i, d), daemon=True) for i, d in enumerate(devices)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Raise the first error encountered, if any.
        for i, err in enumerate(errors):
            if err is not None:
                raise RuntimeError(f"Failed to load model on {devices[i]}: {err}") from err

        return models, resolved_refiner_mode  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------


class _AtomicCounter:
    """Thread-safe integer counter."""

    def __init__(self, value: int) -> None:
        self._value = value
        self._lock = threading.Lock()

    def decrement(self) -> int:
        """Decrement and return the new value."""
        with self._lock:
            self._value -= 1
            return self._value


@dataclass
class _MultiGPUInferenceWorker:
    """InferenceWorker variant for multi-GPU dispatch.

    Identical to InferenceWorker except for STOP propagation: instead of
    always re-putting STOP on the input queue (for a single downstream
    consumer), it uses a shared counter. Only the last worker to finish
    puts STOP on the output queue — earlier workers just decrement the
    counter and exit.

    This prevents N duplicate STOPs from reaching PostWriteWorker.
    """

    input_queue: BoundedQueue
    output_queue: BoundedQueue
    model: nn.Module
    config: InferenceConfig
    active_workers: _AtomicCounter
    worker_index: int
    resolved_refiner_mode: str | None = None
    events: PipelineEvents | None = None

    def run(self) -> None:
        from corridorkey.runtime.queue import STOP
        from corridorkey.stages.inference.orchestrator import run_inference
        from corridorkey.stages.preprocessor import PreprocessedFrame

        if self.events:
            self.events.stage_start(f"inference[{self.config.device}]", 0)
        try:
            while True:
                item = self.input_queue.get()
                if item is STOP:
                    # Re-put so sibling workers also see STOP.
                    self.input_queue.put_stop()
                    break
                assert isinstance(item, PreprocessedFrame)
                try:
                    if self.events:
                        self.events.inference_start(item.meta.frame_index)
                    result = run_inference(
                        item, self.model, self.config, resolved_refiner_mode=self.resolved_refiner_mode
                    )
                    self.output_queue.put(result)
                    logger.debug(
                        "multi_gpu_inference[%s]: queued frame %d",
                        self.config.device,
                        item.meta.frame_index,
                    )
                    if self.events:
                        self.events.inference_queued(item.meta.frame_index)
                        self.events.queue_depth(
                            len(self.input_queue),
                            len(self.output_queue),
                        )
                except Exception as e:
                    logger.error(
                        "multi_gpu_inference[%s]: skipping frame %d — %s",
                        self.config.device,
                        item.meta.frame_index,
                        e,
                    )
                    if self.events:
                        self.events.frame_error(f"inference[{self.config.device}]", item.meta.frame_index, e)
        finally:
            # Last worker to finish sends STOP downstream.
            remaining = self.active_workers.decrement()
            if remaining == 0:
                self.output_queue.put_stop()
                logger.debug("multi_gpu_inference: last worker done, sent STOP downstream")
            if self.events:
                self.events.stage_done(f"inference[{self.config.device}]")

    def start(self) -> threading.Thread:
        t = threading.Thread(
            target=self.run,
            name=f"inference-worker-{self.config.device}",
            daemon=True,
        )
        t.start()
        return t


def _override_device(config: InferenceConfig, device: str) -> InferenceConfig:
    """Return a copy of ``config`` with ``device`` replaced."""
    from dataclasses import replace

    return replace(config, device=device)
