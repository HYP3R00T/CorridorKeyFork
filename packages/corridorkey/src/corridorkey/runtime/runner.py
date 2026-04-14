"""Pipeline — internal frame-loop threading primitives.

Assembly line (N devices, N >= 1):

    PreprocessWorker
        -> preprocess_queue (bounded)
            -> _InferenceWorker[device:0]  -+
            -> _InferenceWorker[device:1]  -+-> inference_queue (bounded)
            -> _InferenceWorker[device:N]  -+       -> PostWriteWorker

Used internally by the Engine. Do not import from outside corridorkey.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from corridorkey.errors import JobCancelledError, ModelError
from corridorkey.events import PipelineEvents
from corridorkey.runtime.model_cache import get_default_cache
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
    """Internal configuration for the frame-loop assembly line.

    Build via :meth:`~corridorkey.infra.config.pipeline.CorridorKeyConfig.to_pipeline_config`.

    Attributes:
        preprocess: Preprocessing stage config.
        inference: Inference stage config. None skips inference.
        postprocess: Postprocessing stage config.
        write: Writer stage config. None derives a default from manifest.output_dir.
        devices: Device strings for inference workers. One entry per GPU.
        events: Optional event callbacks.
        input_queue_depth: Max preprocessed frames waiting for inference.
        output_queue_depth: Max inference outputs waiting for postprocessing.
        resolved_refiner_mode: Pre-resolved refiner mode. Set by to_pipeline_config().
        model: Pre-loaded model shared across workers. None loads from checkpoint.
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


def run_clip(
    manifest: ClipManifest,
    config: PipelineConfig,
    events: PipelineEvents | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Run the full frame-loop pipeline for a single clip.

    Spawns one PreprocessWorker, one PostWriteWorker, and one
    _InferenceWorker per device. Blocks until all frames are written.

    Args:
        manifest: ClipManifest with needs_alpha=False.
        config: Pipeline configuration.
        events: Optional event callbacks. Overrides config.events when provided.
        cancel_event: Optional threading.Event for cancellation.

    Raises:
        JobCancelledError: If cancel_event is set before all frames are processed.
        ValueError: If config.inference is None.
    """

    _events = events if events is not None else config.events
    _cancel = cancel_event or threading.Event()

    if config.inference is None:
        raise ValueError(
            "PipelineConfig.inference is not set. Call CorridorKeyConfig.to_pipeline_config() to build a valid config."
        )

    devices = config.devices if config.devices else [config.inference.device]
    n = len(devices)

    logger.info("frame_loop: clip='%s' frames=%d devices=%s", manifest.clip_name, manifest.frame_count, devices)

    preprocess_queue: BoundedQueue = BoundedQueue(max(config.input_queue_depth, n * 2))
    inference_queue: BoundedQueue = BoundedQueue(max(config.output_queue_depth, n * 2))

    preprocess_worker = PreprocessWorker(
        manifest=manifest,
        config=config.preprocess,
        preprocess_queue=preprocess_queue,
        inference_queue=inference_queue,
        events=_events,
        cancel_event=_cancel,
    )

    models, resolved_refiner_mode = _load_models(devices, config)
    # Shared counter to detect when the last inference worker finishes
    # and should send STOP downstream. Protected by a lock.
    _remaining_lock = threading.Lock()
    _remaining = [n]

    inference_threads: list[threading.Thread] = []
    for device, model in zip(devices, models, strict=True):
        worker = _InferenceWorker(
            preprocess_queue=preprocess_queue,
            inference_queue=inference_queue,
            model=model,
            config=dataclasses.replace(config.inference, device=device),
            resolved_refiner_mode=resolved_refiner_mode,
            remaining=_remaining,
            remaining_lock=_remaining_lock,
            events=_events,
            cancel_event=_cancel,
        )
        inference_threads.append(worker.start(name=f"inference-worker-{device}"))

    postwrite_worker = PostWriteWorker(
        inference_queue=inference_queue,
        output_dir=manifest.output_dir,
        postprocess_config=config.postprocess,
        write_config=config.write,
        total_frames=manifest.frame_count,
        events=_events,
        cancel_event=_cancel,
    )

    all_threads = [preprocess_worker.start(), *inference_threads, postwrite_worker.start()]
    for t in all_threads:
        t.join()

    if _cancel.is_set():
        logger.info("frame_loop: clip='%s' cancelled", manifest.clip_name)
        raise JobCancelledError(manifest.clip_name)

    logger.info("frame_loop: clip='%s' complete", manifest.clip_name)
    if _events:
        _events.clip_complete(manifest.clip_name, manifest.frame_count)


def _load_models(
    devices: list[str],
    config: PipelineConfig,
    timeout: float = 300.0,
) -> tuple[list[nn.Module], str]:
    """Load one model per device, using the process-level cache when possible.

    For single-device runs the cache is consulted first — if the config
    hash matches the cached model it is reused without reloading.
    For multi-device runs each device loads independently (no sharing).
    """
    assert config.inference is not None

    if config.resolved_refiner_mode is not None:
        resolved_refiner_mode = config.resolved_refiner_mode
    elif config.inference.refiner_mode == "auto":
        resolved_refiner_mode = None
    else:
        resolved_refiner_mode = config.inference.refiner_mode

    # Pre-loaded model passed directly — skip cache.
    if config.model is not None:
        if len(devices) > 1:
            raise ValueError(
                "config.model cannot be shared across multiple devices. "
                "Remove config.model and let the pipeline load one copy per device."
            )
        model_device = next(config.model.parameters()).device
        expected = torch.device(devices[0])

        def _resolve(d: torch.device) -> torch.device:
            if d.type == "cuda" and d.index is None:
                return torch.device("cuda", torch.cuda.current_device())
            return d

        if _resolve(model_device) != _resolve(expected):
            raise ValueError(
                f"config.model is on {model_device} but the configured device is "
                f"{expected}. Move the model to the correct device before passing it."
            )
        return [config.model], resolved_refiner_mode or config.inference.refiner_mode

    # Single device — use the process-level model cache.
    if len(devices) == 1:
        import dataclasses as _dc

        single_config = _dc.replace(config.inference, device=devices[0])
        try:
            model, effective_mode = get_default_cache().get(single_config, resolved_refiner_mode)
            return [model], effective_mode
        except Exception as e:
            raise ModelError(f"Failed to load model on {devices[0]}: {e}") from e

    # Multi-device — load one model per device in parallel (no cache).
    from corridorkey.stages.inference.loader import load_model

    models: list[nn.Module | None] = [None] * len(devices)
    errors: list[Exception | None] = [None] * len(devices)

    def _load(idx: int, device: str) -> None:
        try:
            models[idx] = load_model(
                dataclasses.replace(config.inference, device=device),  # type: ignore[arg-type]
                resolved_refiner_mode=resolved_refiner_mode,
            )
        except Exception as e:
            errors[idx] = e
            logger.error("frame_loop: failed to load model on %s — %s", device, e)

    threads = [threading.Thread(target=_load, args=(i, d), daemon=True) for i, d in enumerate(devices)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=timeout)

    hung = [devices[i] for i, t in enumerate(threads) if t.is_alive()]
    if hung:
        raise RuntimeError(f"Model loading timed out after {timeout}s on device(s): {hung}.")

    failed = [(i, err) for i, err in enumerate(errors) if err is not None]
    if failed:
        for i in range(len(models)):
            if models[i] is not None:
                del models[i]
        torch.cuda.empty_cache()
        i, err = failed[0]
        raise ModelError(f"Failed to load model on {devices[i]}: {err}") from err

    return models, resolved_refiner_mode or config.inference.refiner_mode  # type: ignore[return-value]


class _InferenceWorker:
    """Pulls PreprocessedFrame, runs inference, pushes InferenceResult."""

    def __init__(
        self,
        preprocess_queue: BoundedQueue,
        inference_queue: BoundedQueue,
        model: nn.Module,
        config: InferenceConfig,
        remaining: list[int],
        remaining_lock: threading.Lock,
        resolved_refiner_mode: str | None = None,
        events: PipelineEvents | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.preprocess_queue = preprocess_queue
        self.inference_queue = inference_queue
        self.model = model
        self.config = config
        self.remaining = remaining
        self.remaining_lock = remaining_lock
        self.resolved_refiner_mode = resolved_refiner_mode
        self.events = events
        self.cancel_event = cancel_event

    def run(self) -> None:
        from corridorkey.runtime.queue import STOP
        from corridorkey.stages.inference.orchestrator import run_inference
        from corridorkey.stages.preprocessor import PreprocessedFrame

        if self.events:
            self.events.stage_start(f"inference[{self.config.device}]", 0)

        # Defer torch.compile warmup to this thread so CUDA graph TLS is
        # bound to the inference thread, not the calling thread.
        warmup_fn = getattr(self.model, "warmup_on_current_thread", None)
        if callable(warmup_fn):
            try:
                warmup_fn()
            except Exception as e:
                logger.warning("inference[%s]: warmup failed — %s", self.config.device, e)

        try:
            while True:
                if self.cancel_event and self.cancel_event.is_set():
                    self.preprocess_queue.put_stop()
                    break
                item = self.preprocess_queue.get()
                if item is STOP:
                    self.preprocess_queue.put_stop()
                    break
                assert isinstance(item, PreprocessedFrame)
                try:
                    if self.events:
                        self.events.inference_start(item.meta.frame_index)
                    result = run_inference(
                        item, self.model, self.config, resolved_refiner_mode=self.resolved_refiner_mode
                    )
                    self.inference_queue.put(result)
                    if self.events:
                        self.events.inference_queued(item.meta.frame_index)
                        self.events.queue_depth(len(self.preprocess_queue), len(self.inference_queue))
                except Exception as e:
                    from corridorkey.errors import InferenceError, VRAMInsufficientError

                    detail = str(e)
                    if "out of memory" in detail.lower() or "cuda out of memory" in detail.lower():
                        typed: Exception = VRAMInsufficientError(detail)
                    else:
                        typed = InferenceError(item.meta.frame_index, detail)
                    logger.error(
                        "inference[%s]: skipping frame %d — %s", self.config.device, item.meta.frame_index, typed
                    )
                    if self.events:
                        self.events.frame_error(f"inference[{self.config.device}]", item.meta.frame_index, typed)
        finally:
            # Decrement the shared counter. The last worker to finish
            # sends STOP downstream to the postwrite worker.
            with self.remaining_lock:
                self.remaining[0] -= 1
                is_last = self.remaining[0] == 0
            if is_last:
                self.inference_queue.put_stop()
            if self.events:
                self.events.stage_done(f"inference[{self.config.device}]")

    def start(self, name: str = "inference-worker") -> threading.Thread:
        t = threading.Thread(target=self.run, name=name, daemon=True)
        t.start()
        return t
