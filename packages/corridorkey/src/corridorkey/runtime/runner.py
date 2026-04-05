"""Pipeline — runner.

Assembly line (N devices, N >= 1):

    PreprocessWorker
        -> input_queue (bounded, shared)
            -> InferenceWorker[device:0]  -+
            -> InferenceWorker[device:1]  -+-> output_queue (bounded, shared)
            -> InferenceWorker[device:N]  -+       -> PostWriteWorker

All workers run concurrently in daemon threads. The runner blocks until all
workers have finished and all queues are drained.

When ``devices`` is empty or contains a single entry the assembly line has
one InferenceWorker — behaviour is identical to the N>1 case, no special
paths.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from corridorkey.events import PipelineEvents
from corridorkey.runtime.queue import BoundedQueue
from corridorkey.runtime.worker import PostWriteWorker, PreprocessWorker
from corridorkey.stages.inference import InferenceConfig
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.postprocessor import PostprocessConfig
from corridorkey.stages.preprocessor import PreprocessConfig
from corridorkey.stages.writer import WriteConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the pipeline runner.

    Build via :meth:`~corridorkey.infra.config.pipeline.CorridorKeyConfig.to_pipeline_config`
    rather than constructing manually.

    Attributes:
        preprocess: Preprocessing stage config (img_size, device, strategy).
        inference: Inference stage config (checkpoint, device, precision).
            None means inference is skipped (preprocess-only mode).
        postprocess: Postprocessing stage config (despill, despeckle, checkerboard).
        write: Writer stage config (formats, enabled outputs).
            None means a default WriteConfig is derived from the manifest output_dir.
        devices: Device strings for inference workers. One entry per GPU.
            Empty list or a single entry runs one worker.
            Accepted values per entry: "auto", "cuda", "cuda:N", "rocm",
            "rocm:N", "mps", "cpu". Pass ``resolve_devices("all")`` to
            populate this list with every available CUDA GPU.
            Note: "auto" resolves to a single device — it does not expand
            to multiple GPUs. Use "all" or an explicit list for multi-GPU.
        events: Optional event callbacks for progress reporting.
        input_queue_depth: Max preprocessed frames waiting for inference.
            Scaled up automatically with the number of devices.
            Each buffered frame is ~64 MB on GPU at 2048 resolution.
        output_queue_depth: Max inference outputs waiting for postprocessing.
            Scaled up automatically with the number of devices.
        resolved_refiner_mode: Pre-resolved refiner mode ("full_frame" or "tiled").
            Populated by ``to_pipeline_config()`` — do not set manually.
        model: Pre-loaded model (``nn.Module``). Shared across all workers.
            If None, each worker loads its own copy from the checkpoint path.
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


class Runner:
    """Runs the full pipeline for a single clip.

    Instantiate once per clip, call ``run()``, then discard.

    Spawns one :class:`~corridorkey.runtime.worker.PreprocessWorker`, one
    :class:`~corridorkey.runtime.worker.PostWriteWorker`, and one
    :class:`~corridorkey.runtime.worker._InferenceWorker` per device.
    With a single device the assembly line is identical to a classic
    single-GPU pipeline — no special paths.

    Args:
        manifest: ClipManifest from the loader stage. Must have needs_alpha=False.
        config: Pipeline configuration. Build via
            ``CorridorKeyConfig.to_pipeline_config()``.
        events: Optional event callbacks. Overrides ``config.events`` when
            provided, so the same config can be reused with different handlers
            per clip.
    """

    def __init__(
        self,
        manifest: ClipManifest,
        config: PipelineConfig,
        events: PipelineEvents | None = None,
    ) -> None:
        self._manifest = manifest
        self._config = config
        self._events = events if events is not None else config.events

    def run(self) -> None:
        """Run the full pipeline for the clip. Blocks until all frames are written.

        Steps
        -----
        1. Validate config — inference must be set before anything can run.
        2. Resolve devices — fall back to the single device in inference config
           if no explicit device list was provided.
        3. Size the queues — scale depth to the number of devices so each GPU
           always has at least one frame buffered, preventing starvation.
        4. Create queues and workers — one preprocessor, one inference worker per
           device, one postwriter. All share the same two queues.
        5. Load models — delegates to ``_load_models`` to put weights into VRAM.
        6. Start all threads and block — the assembly line runs concurrently;
           this method returns only when every frame has been written to disk.
        """
        cfg = self._config
        manifest = self._manifest

        # Step 1 — validate config.
        if cfg.inference is None:
            raise ValueError(
                "PipelineConfig.inference is not set. "
                "Call CorridorKeyConfig.to_pipeline_config() to build a valid config."
            )

        # Step 2 — resolve devices.
        devices = cfg.devices if cfg.devices else [cfg.inference.device]
        n = len(devices)

        logger.info(
            "runner: starting clip='%s' frames=%d devices=%s",
            manifest.clip_name,
            manifest.frame_count,
            devices,
        )

        # Step 3 — size the queues.
        preprocess_depth = max(cfg.input_queue_depth, n * 2)
        inference_depth = max(cfg.output_queue_depth, n * 2)

        # Step 4 — create queues and workers.
        preprocess_queue: BoundedQueue = BoundedQueue(preprocess_depth)
        inference_queue: BoundedQueue = BoundedQueue(inference_depth)

        preprocess_worker = PreprocessWorker(
            manifest=manifest,
            config=cfg.preprocess,
            preprocess_queue=preprocess_queue,
            inference_queue=inference_queue,
            events=self._events,
        )

        # Step 5 — load models into VRAM.
        models, resolved_refiner_mode = self._load_models(devices, cfg)

        # One shared counter ensures only the last inference worker sends STOP
        # downstream — with a single worker it decrements to 0 immediately.
        active_workers = _AtomicCounter(n)

        inference_threads: list[threading.Thread] = []
        for _i, (device, model) in enumerate(zip(devices, models, strict=True)):
            worker = _InferenceWorker(
                preprocess_queue=preprocess_queue,
                inference_queue=inference_queue,
                model=model,
                config=_override_device(cfg.inference, device),
                resolved_refiner_mode=resolved_refiner_mode,
                active_workers=active_workers,
                events=self._events,
            )
            inference_threads.append(worker.start(name=f"inference-worker-{device}"))

        postwrite_worker = PostWriteWorker(
            inference_queue=inference_queue,
            output_dir=manifest.output_dir,
            postprocess_config=cfg.postprocess,
            write_config=cfg.write,
            total_frames=manifest.frame_count,
            events=self._events,
        )

        # Step 6 — start all threads and block until done.
        all_threads = [preprocess_worker.start(), *inference_threads, postwrite_worker.start()]
        for t in all_threads:
            t.join()

        logger.info("runner: clip='%s' complete", manifest.clip_name)

    def _load_models(
        self,
        devices: list[str],
        cfg: PipelineConfig,
        timeout: float = 300.0,
    ) -> tuple[list[nn.Module], str]:
        """Return one ready-to-run model per device, plus the resolved refiner mode.

        Steps
        -----
        1. Resolve refiner mode — decided once so every model is built identically.
           "auto" is left as None so the loader probes VRAM and decides itself.
        2. Skip loading if a pre-loaded model was passed in (cfg.model) — callers
           processing multiple clips can load the model once and reuse it across
           the batch to avoid redundant disk I/O. Validate that the model is on
           the expected device before returning it.
        3. Load in parallel — one thread per device so startup time stays flat
           regardless of how many GPUs are in use.
        4. Check for hangs — any thread still alive after ``timeout`` means the GPU
           stopped responding; raise immediately rather than blocking forever.
        5. Clean up and raise on error — if any device failed, release the models
           that did load to avoid leaking VRAM, then surface the first error.

        Args:
            devices: Device strings to load onto.
            cfg: Pipeline configuration.
            timeout: Seconds to wait per loading thread before giving up.

        Returns:
            (models in device order, resolved_refiner_mode)
        """
        from corridorkey.stages.inference.loader import load_model

        assert cfg.inference is not None  # checked by caller

        # Step 1 — resolve refiner mode.
        if cfg.resolved_refiner_mode is not None:
            resolved_refiner_mode = cfg.resolved_refiner_mode
        elif cfg.inference.refiner_mode == "auto":
            resolved_refiner_mode = None  # loader will probe VRAM
        else:
            resolved_refiner_mode = cfg.inference.refiner_mode

        logger.info("runner: resolved refiner_mode=%s", resolved_refiner_mode)

        # Step 2 — return pre-loaded model if provided.
        if cfg.model is not None:
            if len(devices) > 1:
                raise ValueError(
                    "cfg.model cannot be shared across multiple devices. "
                    "Remove cfg.model and let the runner load one copy per device, "
                    "or reduce devices to a single entry."
                )
            model_device = next(cfg.model.parameters()).device
            expected = torch.device(devices[0])
            if model_device != expected:
                raise ValueError(
                    f"cfg.model is on {model_device} but the configured device is "
                    f"{expected}. Move the model to the correct device before passing it."
                )
            return [cfg.model], resolved_refiner_mode or cfg.inference.refiner_mode

        # Step 3 — load one model per device in parallel.
        models: list[nn.Module | None] = [None] * len(devices)
        errors: list[Exception | None] = [None] * len(devices)

        def _load(idx: int, device: str) -> None:
            try:
                logger.info("runner: loading model on %s (worker %d)", device, idx)
                models[idx] = load_model(
                    _override_device(cfg.inference, device),  # type: ignore[arg-type]
                    resolved_refiner_mode=resolved_refiner_mode,
                )
                logger.info("runner: model ready on %s", device)
            except Exception as e:
                errors[idx] = e
                logger.error("runner: failed to load model on %s — %s", device, e)

        threads = [threading.Thread(target=_load, args=(i, d), daemon=True) for i, d in enumerate(devices)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=timeout)

        # Step 4 — check for hangs.
        hung = [devices[i] for i, t in enumerate(threads) if t.is_alive()]
        if hung:
            raise RuntimeError(
                f"Model loading timed out after {timeout}s on device(s): {hung}. The GPU may be stalled or unavailable."
            )

        # Step 5 — clean up and raise on error.
        failed = [(i, err) for i, err in enumerate(errors) if err is not None]
        if failed:
            for i in range(len(models)):
                if models[i] is not None:
                    del models[i]
            torch.cuda.empty_cache()
            i, err = failed[0]
            raise RuntimeError(f"Failed to load model on {devices[i]}: {err}") from err

        return models, resolved_refiner_mode or cfg.inference.refiner_mode  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _AtomicCounter:
    """Thread-safe decrement counter."""

    def __init__(self, value: int) -> None:
        self._value = value
        self._lock = threading.Lock()

    def decrement(self) -> int:
        """Decrement and return the new value."""
        with self._lock:
            self._value -= 1
            return self._value


@dataclass
class _InferenceWorker:
    """Pulls PreprocessedFrame, runs inference, pushes InferenceResult.

    Coordinates shutdown via a shared ``_AtomicCounter``: each worker
    re-puts STOP on the preprocess queue so sibling workers also see it, then
    decrements the counter. Only the last worker (counter reaches 0) sends
    STOP downstream to PostWriteWorker. With a single worker this is
    equivalent to the classic single-GPU pattern.
    """

    preprocess_queue: BoundedQueue
    inference_queue: BoundedQueue
    model: nn.Module
    config: InferenceConfig
    active_workers: _AtomicCounter
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
                item = self.preprocess_queue.get()
                if item is STOP:
                    self.preprocess_queue.put_stop()  # pass STOP to sibling workers
                    break
                assert isinstance(item, PreprocessedFrame)
                try:
                    if self.events:
                        self.events.inference_start(item.meta.frame_index)
                    result = run_inference(
                        item, self.model, self.config, resolved_refiner_mode=self.resolved_refiner_mode
                    )
                    self.inference_queue.put(result)
                    logger.debug(
                        "inference[%s]: queued frame %d",
                        self.config.device,
                        item.meta.frame_index,
                    )
                    if self.events:
                        self.events.inference_queued(item.meta.frame_index)
                        self.events.queue_depth(len(self.preprocess_queue), len(self.inference_queue))
                except Exception as e:
                    logger.error(
                        "inference[%s]: skipping frame %d — %s",
                        self.config.device,
                        item.meta.frame_index,
                        e,
                    )
                    if self.events:
                        self.events.frame_error(f"inference[{self.config.device}]", item.meta.frame_index, e)
        finally:
            if self.active_workers.decrement() == 0:
                self.inference_queue.put_stop()
                logger.debug("inference: last worker done, sent STOP downstream")
            if self.events:
                self.events.stage_done(f"inference[{self.config.device}]")

    def start(self, name: str = "inference-worker") -> threading.Thread:
        t = threading.Thread(target=self.run, name=name, daemon=True)
        t.start()
        return t


def _override_device(config: InferenceConfig, device: str) -> InferenceConfig:
    """Return a copy of ``config`` with ``device`` replaced."""
    from dataclasses import replace

    return replace(config, device=device)
