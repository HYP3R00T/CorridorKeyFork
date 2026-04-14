"""CorridorKey Engine.

The Engine coordinates three stages:

  Stage 1 — Prerequisite (hardcoded): scan paths, load clips
  Stage 2 — Alpha Generation (pluggable): generate alpha hint frames
  Stage 3 — Inference (hardcoded, threaded): preprocess → infer → postprocess → write

Usage::

    config = load_config()
    engine = Engine(config)
    engine.set_alpha_generator(MyAlphaGenerator())
    engine.on("frame_done", lambda i, t: print(f"{i}/{t}"))
    engine.run([Path("/clips")])
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from corridorkey.errors import AlphaGeneratorError, EngineError, JobCancelledError
from corridorkey.infra.config.pipeline import CorridorKeyConfig
from corridorkey.runtime.clip_state import ClipRecord, ClipState
from corridorkey.runtime.job_stats import JobStats
from corridorkey.runtime.runner import PipelineConfig
from corridorkey.stages.loader.contracts import ClipManifest

logger = logging.getLogger(__name__)


class Engine:
    """Top-level pipeline orchestrator.

    Construct with a loaded config, register an alpha generator and event
    handlers, then call run(). Construction is cheap — no I/O happens until
    run().

    Args:
        config: Loaded CorridorKeyConfig from load_config().
    """

    def __init__(self, config: CorridorKeyConfig) -> None:
        self._config = config
        self._alpha_generator: Any | None = None
        self._handlers: dict[str, list[Callable]] = {}
        self._cancel_event = threading.Event()

    def set_alpha_generator(self, plugin: Any) -> None:
        """Register an AlphaGenerator for the alpha slot.

        The alpha generator is called for any clip where alpha hint frames
        are absent. It must satisfy the AlphaGenerator protocol:
        generate(manifest: ClipManifest) -> ClipManifest.

        Optional: declare a Config inner class (Pydantic BaseModel) to
        receive settings from [plugins.alpha] in corridorkey.toml.

        Args:
            plugin: Any object with a generate(manifest) method.

        Raises:
            EngineError: If the plugin does not have a generate method.
        """
        if not callable(getattr(plugin, "generate", None)):
            raise EngineError(
                f"{type(plugin).__name__} does not satisfy AlphaGenerator — "
                "must implement generate(manifest) -> ClipManifest"
            )
        self._alpha_generator = plugin
        logger.debug("engine: registered alpha generator %s", type(plugin).__name__)

    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler.

        Handlers run on the thread that fires them — keep them fast or
        dispatch to a queue. Unknown events are silently ignored, keeping
        the event system forward-compatible.

        Args:
            event: Event name (e.g. "frame_done", "clip_error").
            handler: Callable invoked when the event fires.
        """
        self._handlers.setdefault(event, []).append(handler)

    def _emit(self, event: str, *args: Any) -> None:
        """Fire all handlers registered for event."""
        for handler in self._handlers.get(event, []):
            try:
                handler(*args)
            except Exception as e:
                logger.warning("engine: event handler for '%s' raised %s", event, e)

    def cancel(self) -> None:
        """Request cancellation. Thread-safe, callable from any thread.

        Processing stops after the current frame. Frames already written
        are not removed.
        """
        logger.info("engine: cancel requested")
        self._cancel_event.set()

    def run(self, paths: list[Path]) -> JobStats:
        """Run the full pipeline for all clips found under paths.

        Initialisation (logging, device resolution, model download) happens
        here before any processing starts. If anything is wrong it raises
        before the first clip is touched.

        Clips already in the COMPLETE state (all output frames written) are
        skipped automatically and counted as clips_skipped.

        Args:
            paths: Paths to scan. Each may be a clips directory, a single
                clip folder, or a video file. Multiple paths are supported
                so clips at unrelated locations can be processed together.

        Returns:
            JobStats summary of the completed run.

        Raises:
            EngineError: If plugin config validation fails.
            ModelError: If the model cannot be verified or downloaded.
        """
        # Reset per-run state so calling run() twice works correctly.
        self._cancel_event.clear()
        self._resolved_device: str | None = None
        self._pipeline_config: PipelineConfig | None = None
        self._frame_times: list[float] = []  # timestamps of frame_done events
        start_time = time.monotonic()
        stats = JobStats()

        self._initialise()
        self._emit("job_started")
        logger.info("engine: job started, scanning %d path(s)", len(paths))

        # Stage 1 — scan all paths in one call
        from corridorkey.stages.scanner.orchestrator import scan

        try:
            result = scan(paths, reorganise=True)
        except Exception as e:
            logger.error("engine: scan failed — %s", e)
            stats.elapsed_seconds = time.monotonic() - start_time
            self._emit("job_complete", stats)
            return stats

        for skipped in result.skipped:
            self._emit("clip_skipped", skipped, skipped.reason)
            stats.clips_skipped += 1

        # Build ClipRecord for each clip — resolves initial state from disk.
        records = [ClipRecord.from_clip(clip) for clip in result.clips]

        for record in records:
            self._emit("clip_found", record.clip)
            if self._cancel_event.is_set():
                break

            # 3a — skip clips that are already fully processed.
            if record.state == ClipState.COMPLETE:
                logger.info("engine: skipping clip '%s' — already complete", record.name)
                self._emit("clip_skipped", record.clip, "already complete")
                stats.clips_skipped += 1
                continue

            self._process_clip(record, stats)

        stats.elapsed_seconds = time.monotonic() - start_time
        stats.frames_per_second = self._compute_fps()
        self._emit("job_complete", stats)
        logger.info(
            "engine: job complete — processed=%d failed=%d skipped=%d cancelled=%d elapsed=%.1fs",
            stats.clips_processed,
            stats.clips_failed,
            stats.clips_skipped,
            stats.clips_cancelled,
            stats.elapsed_seconds,
        )
        return stats

    def _initialise(self) -> None:
        """Set up logging, resolve device, wire plugin configs, verify model."""
        from corridorkey.infra.logging import setup_logging
        from corridorkey.infra.model_hub import ensure_model

        setup_logging(self._config)
        self._wire_plugin_configs()

        from corridorkey.infra.device_utils import resolve_device, resolve_devices

        devices: list[str] = []
        if self._config.device == "all":
            devices = resolve_devices("all")
            self._resolved_device = devices[0]
            logger.info("engine: device resolved to %s (%d devices)", self._resolved_device, len(devices))
        else:
            self._resolved_device = resolve_device(self._config.device)
            logger.info("engine: device resolved to %s", self._resolved_device)

        checkpoint = self._config.inference.checkpoint_path
        self._emit("model_loading")
        logger.info("engine: verifying model at %s", checkpoint)

        def _on_progress(done: int, total: int) -> None:
            self._emit("download_progress", done, total)

        try:
            ensure_model(dest_dir=checkpoint.parent, on_progress=_on_progress)
        except Exception as e:
            from corridorkey.errors import ModelError

            raise ModelError(f"Model verification failed: {e}") from e

        self._emit("model_ready")
        logger.info("engine: model ready")

        self._pipeline_config = self._config.to_pipeline_config(
            device=self._resolved_device,
            devices=devices or None,
        )

    def _wire_plugin_configs(self) -> None:
        """Wire TOML [plugins.*] config sections into registered plugins."""
        plugins = {"alpha": self._alpha_generator}
        for name, plugin in plugins.items():
            if plugin is None:
                continue
            config_cls = getattr(type(plugin), "Config", None)
            if config_cls is None:
                continue
            raw = getattr(self._config, "plugins", {})
            section = raw.get(name, {}) if isinstance(raw, dict) else {}
            try:
                plugin.config = config_cls(**section)
            except Exception as e:
                raise EngineError(f"Plugin '{name}' config validation failed: {e}") from e

    def _process_clip(self, record: ClipRecord, stats: JobStats) -> None:
        """Run all three stages for a single clip."""
        clip = record.clip
        self._emit("clip_loading", clip)
        logger.info("engine: processing clip '%s'", clip.name)

        record.set_processing(True)
        try:
            # Stage 1 — load
            from corridorkey.stages.loader.orchestrator import load

            try:
                manifest = load(clip)
            except Exception as e:
                e._stage = "load"  # type: ignore[attr-defined]
                raise

            record.manifest = manifest

            # Stage 2 — alpha (only when needed)
            if manifest.needs_alpha:
                manifest = self._run_alpha(manifest, record)

            record.transition_to(ClipState.READY)
            self._emit("clip_ready", manifest)

            # Stage 3 — inference (threaded assembly line)
            self._run_inference(manifest)

            record.transition_to(ClipState.COMPLETE)
            stats.clips_processed += 1
            stats.total_frames += manifest.frame_count
            self._emit("clip_complete", manifest)
            logger.info("engine: clip '%s' complete", clip.name)

        except JobCancelledError:
            record.set_error("cancelled")
            stats.clips_cancelled += 1
            self._emit("clip_cancelled", clip)
            logger.info("engine: clip '%s' cancelled", clip.name)

        except Exception as e:
            stage = getattr(e, "_stage", "unknown")
            record.set_error(f"{stage}: {e}")
            stats.clips_failed += 1
            self._emit("clip_error", stage, e)
            logger.error("engine: clip '%s' failed at stage '%s' — %s", clip.name, stage, e)

        finally:
            record.set_processing(False)

    def _on_frame_done(self, index: int, total: int) -> None:
        """Record frame completion time and emit frame_done event."""
        self._frame_times.append(time.monotonic())
        self._emit("frame_done", index, total)

    def _compute_fps(self) -> float:
        """Compute sustained fps as a simple average over steady-state frames.

        Uses the timestamps of all frame_done events. Skips the first frame
        so Triton/torch.compile JIT cost doesn't drag down the displayed rate.
        Returns 0.0 if fewer than 2 steady-state frames were processed.
        """
        times = self._frame_times
        # Need at least 2 frames after warmup (index 0 = warmup, index 1+ = steady)
        if len(times) < 2:
            return 0.0
        steady = times[1:]  # exclude first frame
        elapsed = steady[-1] - steady[0]
        if elapsed <= 0:
            return 0.0
        # frames / elapsed between first and last steady-state frame
        return (len(steady) - 1) / elapsed

    def _run_alpha(self, manifest: ClipManifest, record: ClipRecord) -> ClipManifest:
        """Stage 2 — call the registered alpha generator."""
        if self._alpha_generator is None:
            raise AlphaGeneratorError(manifest.clip_name)
        try:
            result = self._alpha_generator.generate(manifest)
        except Exception as e:
            e._stage = "alpha"  # type: ignore[attr-defined]
            raise
        if result.needs_alpha:
            raise AlphaGeneratorError(
                f"AlphaGenerator returned needs_alpha=True for '{manifest.clip_name}' — "
                "alpha_frames_dir must be set on the returned manifest"
            )
        record.manifest = result
        self._emit("alpha_resolved", result)
        return result

    def _run_inference(self, manifest: ClipManifest) -> None:
        """Stage 3 — run the threaded assembly line for a single clip."""
        from corridorkey.events import PipelineEvents
        from corridorkey.runtime.runner import run_clip

        assert self._pipeline_config is not None, (
            "_pipeline_config is None — _initialise() must be called before _run_inference()"
        )
        total = manifest.frame_count
        events = PipelineEvents(
            on_frame_written=lambda i, _t: self._on_frame_done(i, total),
            on_frame_error=lambda s, i, e: self._emit("frame_error", s, i, e),
            on_stage_start=lambda s, t: self._emit("stage_start", s, t),
            on_stage_done=lambda s: self._emit("stage_done", s),
            on_preprocess_queued=lambda i: self._emit("preprocess_queued", i),
            on_inference_start=lambda i: self._emit("inference_start", i),
            on_inference_queued=lambda i: self._emit("inference_queued", i),
            on_queue_depth=lambda p, w: self._emit("queue_depth", p, w),
            on_clip_complete=lambda name, n: None,  # engine emits clip_complete itself
        )
        run_clip(manifest, self._pipeline_config, events=events, cancel_event=self._cancel_event)
