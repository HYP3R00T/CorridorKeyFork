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
from corridorkey.runtime.job_stats import JobStats
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.scanner.contracts import Clip

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

    # ------------------------------------------------------------------
    # Alpha generator registration
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Event registration
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation. Thread-safe, callable from any thread.

        Processing stops after the current frame. Frames already written
        are not removed.
        """
        logger.info("engine: cancel requested")
        self._cancel_event.set()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, paths: list[Path]) -> JobStats:
        """Run the full pipeline for all clips found under paths.

        Initialisation (logging, device resolution, model download) happens
        here before any processing starts. If anything is wrong it raises
        before the first clip is touched.

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
        self._cancel_event.clear()
        start_time = time.monotonic()
        stats = JobStats()

        self._initialise()
        self._emit("job_started")
        logger.info("engine: job started, scanning %d path(s)", len(paths))

        # Stage 1 — scan all paths in one call
        from corridorkey.stages.scanner.orchestrator import scan

        try:
            result = scan(paths)
        except Exception as e:
            logger.error("engine: scan failed — %s", e)
            stats.elapsed_seconds = time.monotonic() - start_time
            self._emit("job_complete", stats)
            return stats

        for skipped in result.skipped:
            self._emit("clip_skipped", skipped, skipped.reason)
            stats.clips_skipped += 1

        for clip in result.clips:
            self._emit("clip_found", clip)
            if self._cancel_event.is_set():
                break
            self._process_clip(clip, stats)

        stats.elapsed_seconds = time.monotonic() - start_time
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

    # ------------------------------------------------------------------
    # Internal — initialisation
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal — clip processing
    # ------------------------------------------------------------------

    def _process_clip(self, clip: Clip, stats: JobStats) -> None:
        """Run all three stages for a single clip."""
        self._emit("clip_loading", clip)
        logger.info("engine: processing clip '%s'", clip.name)

        try:
            # Stage 1 — load
            from corridorkey.stages.loader.orchestrator import load

            try:
                manifest = load(clip)
            except Exception as e:
                e._stage = "load"  # type: ignore[attr-defined]
                raise

            # Stage 2 — alpha (only when needed)
            if manifest.needs_alpha:
                manifest = self._run_alpha(manifest)

            self._emit("clip_ready", manifest)

            # Stage 3 — inference (threaded assembly line)
            self._run_inference(manifest)

            stats.clips_processed += 1
            stats.total_frames += manifest.frame_count
            self._emit("clip_complete", manifest)
            logger.info("engine: clip '%s' complete", clip.name)

        except JobCancelledError:
            stats.clips_cancelled += 1
            self._emit("clip_cancelled", clip)
            logger.info("engine: clip '%s' cancelled", clip.name)

        except Exception as e:
            stage = getattr(e, "_stage", "unknown")
            stats.clips_failed += 1
            self._emit("clip_error", stage, e)
            logger.error("engine: clip '%s' failed at stage '%s' — %s", clip.name, stage, e)

    def _run_alpha(self, manifest: ClipManifest) -> ClipManifest:
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
        self._emit("alpha_resolved", result)
        return result

    def _run_inference(self, manifest: ClipManifest) -> None:
        """Stage 3 — run the threaded assembly line for a single clip."""
        from corridorkey.events import PipelineEvents
        from corridorkey.runtime.runner import run_clip

        total = manifest.frame_count
        events = PipelineEvents(
            on_frame_written=lambda i, _t: self._emit("frame_done", i, total),
            on_frame_error=lambda s, i, e: self._emit("frame_error", s, i, e),
            on_stage_start=lambda s, t: self._emit("stage_start", s, t),
            on_stage_done=lambda s: self._emit("stage_done", s),
            on_clip_complete=lambda name, n: None,  # engine emits clip_complete itself
        )
        run_clip(manifest, self._pipeline_config, events=events, cancel_event=self._cancel_event)
