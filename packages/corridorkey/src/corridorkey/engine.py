"""CorridorKey Engine.

The Engine is the top-level orchestrator. It owns the full clip lifecycle —
scanning, loading, alpha generation, frame processing, and writing.

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

# Ordered catalog of stage names — anchor points for insert_after.
_CLIP_STAGES = ("scan", "load", "alpha", "validate")
_FRAME_STAGES = ("preprocess", "inference", "postprocess", "write")
_ALL_STAGES = _CLIP_STAGES + _FRAME_STAGES


class Engine:
    """Top-level pipeline orchestrator.

    Construct with a loaded config, register plugins and event handlers,
    then call run(). Construction is cheap — no I/O happens until run().

    Args:
        config: Loaded CorridorKeyConfig from load_config().
    """

    def __init__(self, config: CorridorKeyConfig) -> None:
        self._config = config
        self._alpha_generator: Any | None = None
        self._validate_plugin: Any | None = None
        self._stage_overrides: dict[str, Any] = {}
        self._insertions: dict[str, list[Any]] = {name: [] for name in _ALL_STAGES}
        self._handlers: dict[str, list[Callable]] = {}
        self._cancel_event = threading.Event()

    # ------------------------------------------------------------------
    # Plugin registration
    # ------------------------------------------------------------------

    def set_alpha_generator(self, plugin: Any) -> None:
        """Register an AlphaGenerator for the alpha slot.

        Args:
            plugin: Any object with a generate(manifest, config) method.

        Raises:
            EngineError: If the plugin does not satisfy the AlphaGenerator protocol.
        """
        if not callable(getattr(plugin, "generate", None)):
            raise EngineError(
                f"{type(plugin).__name__} does not satisfy AlphaGenerator — "
                "must implement generate(manifest, config) -> ClipManifest"
            )
        self._alpha_generator = plugin
        logger.debug("engine: registered alpha generator %s", type(plugin).__name__)

    def set_stage(self, name: str, plugin: Any) -> None:
        """Replace a named built-in stage with a custom implementation.

        Args:
            name: Stage name from the catalog (e.g. "inference").
            plugin: Replacement implementation.

        Raises:
            EngineError: If name is not in the stage catalog.
        """
        if name not in _ALL_STAGES:
            raise EngineError(f"Unknown stage '{name}'. Valid stages: {', '.join(_ALL_STAGES)}")
        self._stage_overrides[name] = plugin
        logger.debug("engine: replaced stage '%s' with %s", name, type(plugin).__name__)

    def insert_after(self, name: str, plugin: Any) -> None:
        """Insert a plugin after a named stage.

        Multiple plugins inserted after the same stage execute in registration
        order. The plugin must be callable — it receives the output of the
        preceding stage and must return the same type.

        Args:
            name: Stage name to insert after.
            plugin: Callable plugin. Must accept and return the boundary type.

        Raises:
            EngineError: If name is not in the stage catalog.
        """
        if name not in _ALL_STAGES:
            raise EngineError(f"Unknown stage '{name}'. Valid stages: {', '.join(_ALL_STAGES)}")
        if not callable(plugin):
            raise EngineError(f"{type(plugin).__name__} is not callable. insert_after plugins must be callable.")
        self._insertions[name].append(plugin)
        logger.debug("engine: inserted plugin %s after '%s'", type(plugin).__name__, name)

    # ------------------------------------------------------------------
    # Event registration
    # ------------------------------------------------------------------

    def on(self, event: str, handler: Callable) -> None:
        """Register an event handler.

        Handlers registered for unknown events are silently ignored at
        registration time and never called. This keeps the event system
        forward-compatible — new events can be added without breaking
        existing integrations.

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

    def run(
        self,
        paths: list[Path],
        frame_range: Any | None = None,
    ) -> JobStats:
        """Run the full pipeline for all clips found under paths.

        This is where all initialisation happens: slot validation, plugin
        config wiring, logging setup, device resolution, VRAM probe, and
        model verification. If anything is wrong it raises before processing
        starts.

        Args:
            paths: List of Path objects to scan. Each may be a clips directory,
                a single clip folder, or a video file.
            frame_range: Optional FrameRange. If provided, only that subset of
                frames is processed for each clip. If None, full clip is processed.

        Returns:
            JobStats summary of the completed run.

        Raises:
            EngineError: If required slots are unfilled or plugin configs are invalid.
        """
        self._cancel_event.clear()
        start_time = time.monotonic()
        stats = JobStats()

        # Step 1 — initialise (logging, device, model)
        self._initialise()

        self._emit("job_started")
        logger.info("engine: job started, scanning %d path(s)", len(paths))

        # Step 2 — scan all paths
        clips, skipped_count = self._scan_paths(paths)
        stats.clips_skipped = skipped_count

        # Step 3 — process clips sequentially
        for clip in clips:
            if self._cancel_event.is_set():
                break
            self._process_clip(clip, frame_range, stats)

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
        """Set up logging, resolve device, verify model. Runs once per run()."""
        from corridorkey.infra.logging import setup_logging
        from corridorkey.infra.model_hub import ensure_model

        setup_logging(self._config)

        # Wire plugin configs from TOML [plugins.*] sections
        self._wire_plugin_configs()

        # Resolve device early so it's available for model loading
        from corridorkey.infra.device_utils import resolve_device, resolve_devices

        devices: list[str] = []
        if self._config.device == "all":
            devices = resolve_devices("all")
            self._resolved_device = devices[0]
            logger.info("engine: device resolved to %s (%d devices)", self._resolved_device, len(devices))
        else:
            self._resolved_device = resolve_device(self._config.device)
            logger.info("engine: device resolved to %s", self._resolved_device)

        # Verify / download model
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

        self._emit("download_complete")
        self._emit("model_ready")
        logger.info("engine: model ready")

        # Build and cache pipeline config (resolves VRAM, precision, img_size once)
        self._pipeline_config = self._config.to_pipeline_config(
            device=self._resolved_device,
            devices=devices or None,
        )

    def _wire_plugin_configs(self) -> None:
        """Pass validated plugin configs to registered plugins."""
        plugins: dict[str, Any] = {
            "alpha_generator": self._alpha_generator,
            "validate": self._validate_plugin,
        }
        plugins.update(self._stage_overrides)
        for name, plugin in plugins.items():
            if plugin is None:
                continue
            config_cls = getattr(type(plugin), "Config", None)
            if config_cls is None:
                continue
            # Load from TOML [plugins.<name>] section if present
            raw = getattr(self._config, "plugins", {})
            section = raw.get(name, {}) if isinstance(raw, dict) else {}
            try:
                plugin.config = config_cls(**section)
            except Exception as e:
                raise EngineError(f"Plugin '{name}' config validation failed: {e}") from e

    # ------------------------------------------------------------------
    # Internal — scanning
    # ------------------------------------------------------------------

    def _scan_paths(self, paths: list[Path]) -> tuple[list[Clip], int]:
        """Scan all paths and return discovered clips plus skipped count."""
        from corridorkey.stages.scanner.orchestrator import scan

        all_clips: list[Clip] = []
        skipped_count = 0

        for path in paths:
            try:
                result = scan(path)
            except Exception as e:
                logger.warning("engine: scan failed for %s — %s", path, e)
                continue

            for clip in result.clips:
                self._emit("clip_found", clip)
                all_clips.append(clip)

            for skipped in result.skipped:
                self._emit("clip_skipped", skipped, skipped.reason)
                skipped_count += 1

        logger.info("engine: found %d clip(s), %d skipped", len(all_clips), skipped_count)
        return all_clips, skipped_count

    # ------------------------------------------------------------------
    # Internal — clip processing
    # ------------------------------------------------------------------

    def _process_clip(
        self,
        clip: Clip,
        frame_range: Any | None,
        stats: JobStats,
    ) -> None:
        """Run the full clip lifecycle for a single clip."""
        self._emit("clip_loading", clip)
        logger.info("engine: processing clip '%s'", clip.name)

        try:
            # Load
            manifest = self._run_load(clip)

            # Alpha
            if manifest.needs_alpha:
                manifest = self._run_alpha(manifest)

            # Validate
            manifest = self._run_validate(manifest)

            # Apply frame_range if provided
            if frame_range is not None:
                manifest = manifest.model_copy(update={"frame_range": frame_range.to_frame_range()})

            self._emit("clip_ready", manifest)

            # Frame loop
            self._run_frame_loop(manifest, stats)

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

    def _run_load(self, clip: Clip) -> ClipManifest:
        from corridorkey.stages.loader.orchestrator import load

        try:
            return load(clip)
        except Exception as e:
            e._stage = "load"  # type: ignore[attr-defined]
            raise

    def _run_alpha(self, manifest: ClipManifest) -> ClipManifest:
        if self._alpha_generator is None:
            self._emit("alpha_needed", manifest)
            raise AlphaGeneratorError(manifest.clip_name)
        try:
            result = self._alpha_generator.generate(manifest, self._config)
        except Exception as e:
            e._stage = "alpha"  # type: ignore[attr-defined]
            raise
        if result.needs_alpha:
            raise AlphaGeneratorError(
                f"AlphaGenerator returned a manifest with needs_alpha=True for '{manifest.clip_name}'"
            )
        self._emit("alpha_resolved", result)
        return result

    def _run_validate(self, manifest: ClipManifest) -> ClipManifest:
        if self._validate_plugin is None:
            return manifest
        try:
            return self._validate_plugin(manifest, self._config)
        except Exception as e:
            e._stage = "validate"  # type: ignore[attr-defined]
            raise

    def _run_frame_loop(self, manifest: ClipManifest, stats: JobStats) -> None:
        """Run the frame-loop assembly line for a single clip."""
        from corridorkey.events import PipelineEvents
        from corridorkey.runtime.runner import run_clip

        total = manifest.frame_count

        def _on_frame_written(index: int, _total: int) -> None:
            self._emit("frame_done", index, total)

        events = PipelineEvents(on_frame_written=_on_frame_written)
        run_clip(manifest, self._pipeline_config, events=events, cancel_event=self._cancel_event)
