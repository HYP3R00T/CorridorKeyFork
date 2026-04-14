"""Pipeline — internal frame-loop threading primitives.

Assembly line (N devices, N >= 1):

    _MultiClipPreprocessWorker
        -> preprocess_queue (bounded)
            -> _InferenceWorker[device:0]  -+
            -> _InferenceWorker[device:1]  -+-> inference_queue (bounded)
            -> _InferenceWorker[device:N]  -+       -> PostWriteWorker

Frames from all clips flow through a single shared queue pair so faster
GPUs can pull work from any clip without waiting for clip boundaries.
Each item on the queues is a ``_FrameWork`` that pairs the preprocessed
tensor with its source ``ClipManifest``, giving the postwrite worker the
output directory and write config it needs without any global state.

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
from corridorkey.runtime.queue import STOP, BoundedQueue
from corridorkey.runtime.worker import _MEM_HEADROOM, _available_ram
from corridorkey.stages.inference import InferenceConfig
from corridorkey.stages.inference.deferred import DeferredTransfer
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.postprocessor import PostprocessConfig, postprocess_frame
from corridorkey.stages.preprocessor import PreprocessConfig
from corridorkey.stages.preprocessor.contracts import PreprocessedFrame
from corridorkey.stages.writer import WriteConfig, write_frame

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


@dataclass(frozen=True)
class _FrameWork:
    """Pipeline-internal wrapper pairing a preprocessed frame with its clip manifest."""

    frame: PreprocessedFrame
    manifest: ClipManifest


class _InferenceWork:
    """Pipeline-internal: pairs a DeferredTransfer with its source ClipManifest."""

    __slots__ = ("transfer", "manifest")

    def __init__(self, transfer: DeferredTransfer, manifest: ClipManifest) -> None:
        self.transfer = transfer
        self.manifest = manifest

    @property
    def meta(self) -> object:
        return self.transfer.meta


def run_clip(
    manifest: ClipManifest,
    config: PipelineConfig,
    events: PipelineEvents | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Run the full frame-loop pipeline for a single clip.

    Convenience wrapper around :func:`run_clips` for the single-clip case.
    """
    run_clips([manifest], config, events=events, cancel_event=cancel_event)


def run_clips(
    manifests: list[ClipManifest],
    config: PipelineConfig,
    events: PipelineEvents | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Run the full frame-loop pipeline for one or more clips.

    All clips share a single preprocess queue and a single inference queue,
    so faster GPUs pull frames from any clip without waiting for clip
    boundaries. This eliminates GPU idle time between short clips in a batch.

    Spawns one _MultiClipPreprocessWorker, one PostWriteWorker, and one
    _InferenceWorker per device. Blocks until all frames from all clips
    are written.

    Args:
        manifests: One or more ClipManifests with needs_alpha=False.
        config: Pipeline configuration.
        events: Optional event callbacks. Overrides config.events when provided.
        cancel_event: Optional threading.Event for cancellation.

    Raises:
        JobCancelledError: If cancel_event is set before all frames are processed.
        ValueError: If config.inference is None or manifests is empty.
    """
    if not manifests:
        raise ValueError("run_clips: manifests list is empty.")

    _events = events if events is not None else config.events
    _cancel = cancel_event or threading.Event()

    if config.inference is None:
        raise ValueError(
            "PipelineConfig.inference is not set. Call CorridorKeyConfig.to_pipeline_config() to build a valid config."
        )

    devices = config.devices if config.devices else [config.inference.device]
    n = len(devices)
    total_frames = sum(m.frame_count for m in manifests)

    logger.info(
        "frame_loop: clips=%d total_frames=%d devices=%s",
        len(manifests),
        total_frames,
        devices,
    )

    preprocess_queue: BoundedQueue = BoundedQueue(max(config.input_queue_depth, n * 2))
    inference_queue: BoundedQueue = BoundedQueue(max(config.output_queue_depth, n * 2))

    preprocess_worker = _MultiClipPreprocessWorker(
        manifests=manifests,
        config=config.preprocess,
        preprocess_queue=preprocess_queue,
        inference_queue=inference_queue,
        events=_events,
        cancel_event=_cancel,
    )

    models, resolved_refiner_mode = _load_models(devices, config)
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

    postwrite_worker = _MultiClipPostWriteWorker(
        inference_queue=inference_queue,
        default_postprocess_config=config.postprocess,
        default_write_config=config.write,
        total_frames=total_frames,
        n_gpus=n,
        events=_events,
        cancel_event=_cancel,
    )

    all_threads = [preprocess_worker.start(), *inference_threads, postwrite_worker.start()]
    for t in all_threads:
        t.join()

    if _cancel.is_set():
        clip_names = ", ".join(m.clip_name for m in manifests)
        logger.info("frame_loop: cancelled (clips: %s)", clip_names)
        # Report cancellation for the first clip (Engine handles per-clip accounting)
        raise JobCancelledError(manifests[0].clip_name)

    for manifest in manifests:
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
    """Pulls _FrameWork items, runs inference, pushes _InferenceWork onto inference_queue."""

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
        from corridorkey.stages.inference.orchestrator import run_inference

        if self.events:
            self.events.stage_start(f"inference[{self.config.device}]", 0)

        # Create the copy stream here, on the inference thread, so the CUDA
        # context is bound to the correct thread (required for torch.compile
        # and CUDA graph TLS correctness).
        copy_stream: torch.cuda.Stream | None = None
        if torch.device(self.config.device).type == "cuda":
            copy_stream = torch.cuda.Stream(device=self.config.device)

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

                assert isinstance(item, _FrameWork)
                frame = item.frame
                manifest = item.manifest

                try:
                    if self.events:
                        self.events.inference_start(frame.meta.frame_index)
                    result = run_inference(
                        frame, self.model, self.config, resolved_refiner_mode=self.resolved_refiner_mode
                    )
                    transfer = DeferredTransfer.start(result.alpha, result.fg, result.meta, copy_stream)
                    self.inference_queue.put(_InferenceWork(transfer=transfer, manifest=manifest))
                    if self.events:
                        self.events.inference_queued(frame.meta.frame_index)
                        self.events.queue_depth(len(self.preprocess_queue), len(self.inference_queue))
                except Exception as e:
                    from corridorkey.errors import InferenceError, VRAMInsufficientError

                    detail = str(e)
                    if "out of memory" in detail.lower() or "cuda out of memory" in detail.lower():
                        typed: Exception = VRAMInsufficientError(detail)
                    else:
                        typed = InferenceError(frame.meta.frame_index, detail)
                    logger.error(
                        "inference[%s]: skipping frame %d — %s", self.config.device, frame.meta.frame_index, typed
                    )
                    if self.events:
                        self.events.frame_error(f"inference[{self.config.device}]", frame.meta.frame_index, typed)
        finally:
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


class _MultiClipPreprocessWorker:
    """Reads and preprocesses frames from multiple clips into a single shared queue.

    Iterates clips in order, feeding all frames from each clip before moving
    to the next. RAM throttling is applied across the combined stream.
    """

    def __init__(
        self,
        manifests: list[ClipManifest],
        config: PreprocessConfig,
        preprocess_queue: BoundedQueue,
        inference_queue: BoundedQueue | None = None,
        events: PipelineEvents | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.manifests = manifests
        self.config = config
        self.preprocess_queue = preprocess_queue
        self.inference_queue = inference_queue
        self.events = events
        self.cancel_event = cancel_event

    def run(self) -> None:
        import time

        from corridorkey.errors import FrameReadError
        from corridorkey.stages.loader.validator import list_frames
        from corridorkey.stages.preprocessor import preprocess_frame

        total = sum(m.frame_count for m in self.manifests)
        if self.events:
            self.events.stage_start("preprocess", total)

        _frame_bytes_estimate: list[int] = [0]

        def _mem_ok() -> bool:
            est = _frame_bytes_estimate[0]
            if est == 0:
                return True
            return _available_ram() - est > _MEM_HEADROOM

        try:
            for manifest in self.manifests:
                if self.cancel_event and self.cancel_event.is_set():
                    break

                image_files = list_frames(manifest.frames_dir)
                alpha_files = list_frames(manifest.alpha_frames_dir)  # type: ignore[arg-type]

                for i in range(*manifest.frame_range):
                    if self.cancel_event and self.cancel_event.is_set():
                        logger.debug("preprocess_worker: cancelled at frame %d of '%s'", i, manifest.clip_name)
                        break

                    if not _mem_ok():
                        logger.debug("preprocess_worker: RAM pressure, waiting before frame %d", i)
                        while not _mem_ok():
                            if self.cancel_event and self.cancel_event.is_set():
                                break
                            time.sleep(0.1)

                    try:
                        frame = preprocess_frame(
                            manifest,
                            i,
                            self.config,
                            image_files=image_files,
                            alpha_files=alpha_files,
                        )
                        if _frame_bytes_estimate[0] == 0:
                            _frame_bytes_estimate[0] = frame.tensor.nbytes

                        work = _FrameWork(frame=frame, manifest=manifest)
                        if self.cancel_event is not None:
                            enqueued = self.preprocess_queue.put_unless_cancelled(work, self.cancel_event)
                            if not enqueued:
                                logger.debug("preprocess_worker: cancelled while enqueuing frame %d", i)
                                return
                        else:
                            self.preprocess_queue.put(work)

                        logger.debug("preprocess_worker: queued frame %d of '%s'", i, manifest.clip_name)
                        if self.events:
                            self.events.preprocess_queued(i)
                            self.events.queue_depth(
                                len(self.preprocess_queue),
                                len(self.inference_queue) if self.inference_queue else 0,
                            )
                    except FrameReadError as e:
                        logger.error("preprocess_worker: skipping frame %d of '%s' — %s", i, manifest.clip_name, e)
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


class _MultiClipPostWriteWorker:
    """Pulls _InferenceWork items, postprocesses, and writes frames to disk.

    Each _InferenceWork carries its source ClipManifest so the correct
    output directory is used regardless of which clip the frame came from.
    """

    def __init__(
        self,
        inference_queue: BoundedQueue,
        default_postprocess_config: PostprocessConfig,
        default_write_config: WriteConfig | None,
        total_frames: int = 0,
        n_gpus: int = 1,
        events: PipelineEvents | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.inference_queue = inference_queue
        self.default_postprocess_config = default_postprocess_config
        self.default_write_config = default_write_config
        self.total_frames = total_frames
        self.n_gpus = n_gpus
        self.events = events
        self.cancel_event = cancel_event

    def run(self) -> None:
        from concurrent.futures import ThreadPoolExecutor

        from corridorkey.errors import PostprocessError, WriteFailureError

        if self.events:
            self.events.stage_start("postwrite", self.total_frames)

        # Scale pool to GPU count: each GPU can produce frames concurrently,
        # so the write pool needs enough threads to drain all of them.
        n_workers = max(4, self.n_gpus * 4)
        futures: list = []

        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="postwrite") as pool:
            while True:
                if self.cancel_event and self.cancel_event.is_set():
                    logger.debug("postwrite_worker: cancelled")
                    self.inference_queue.put_stop()
                    break
                item = self.inference_queue.get()
                if item is STOP:
                    self.inference_queue.put_stop()
                    break
                assert isinstance(item, _InferenceWork)
                frame_index = item.meta.frame_index  # type: ignore[union-attr]
                future = pool.submit(self._process_one, item)
                futures.append((frame_index, future))

        for frame_index, future in futures:
            try:
                future.result()
                logger.debug("postwrite_worker: wrote frame %d", frame_index)
                if self.events:
                    self.events.frame_written(frame_index, self.total_frames)
            except WriteFailureError as e:
                logger.error("postwrite_worker: write failed frame %d — %s", frame_index, e)
                if self.events:
                    self.events.frame_error("postwrite", frame_index, e)
            except OSError as e:
                typed = WriteFailureError(str(getattr(e, "filename", "unknown")), str(e))
                logger.error("postwrite_worker: write failed frame %d — %s", frame_index, typed)
                if self.events:
                    self.events.frame_error("postwrite", frame_index, typed)
            except Exception as e:
                typed = PostprocessError(frame_index, str(e))
                logger.error("postwrite_worker: postprocess failed frame %d — %s", frame_index, typed)
                if self.events:
                    self.events.frame_error("postwrite", frame_index, typed)

        logger.debug("postwrite_worker: done")
        if self.events:
            self.events.stage_done("postwrite")

    def _process_one(self, item: _InferenceWork) -> None:
        from corridorkey.stages.inference.contracts import InferenceResult
        from corridorkey.stages.writer import WriteConfig

        alpha_cpu, fg_cpu, meta = item.transfer.resolve()
        result = InferenceResult(alpha=alpha_cpu, fg=fg_cpu, meta=meta)

        manifest = item.manifest
        write_cfg = self.default_write_config or WriteConfig(output_dir=manifest.output_dir)
        processed = postprocess_frame(result, self.default_postprocess_config, output_dir=manifest.output_dir)
        write_frame(processed, write_cfg)

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name="postwrite-worker", daemon=True)
        t.start()
        return t
