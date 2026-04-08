"""CorridorKey pipeline — public API.

This is the single import surface for any interface (CLI, GUI, TUI, plugin).
Import everything you need from here — do not import from submodules directly.

There are two integration paths. Choose the one that fits your use case.

-------------------------------------------------------------------------------
Path 1 — Managed pipeline (recommended)
-------------------------------------------------------------------------------
The package owns threading, queuing, multi-GPU dispatch, backpressure, and
shutdown. Your interface provides a clip and a config, then waits for
completion. Progress is delivered via PipelineEvents callbacks.

Use this for: CLI, TUI, GUI, web backends — anything that processes clips
from disk and does not need to control the frame loop itself.

    # 1. Startup
    config = load_config()
    setup_logging(config)
    device = resolve_device(config.device)
    ensure_model()

    # 2. Discover clips
    result = scan("/path/to/clips")

    # 3. Build config once — reuse across all clips in the session
    pipeline_config = config.to_pipeline_config(device=device)

    # 4. Attach progress callbacks (optional)
    pipeline_config.events = PipelineEvents(
        on_frame_written=lambda idx, total: print(f"{idx + 1}/{total}"),
        on_frame_error=lambda stage, idx, err: print(f"Error: {err}"),
    )

    # 5. Process each clip
    for clip in result.clips:
        manifest = load(clip)
        if manifest.needs_alpha:
            manifest = resolve_alpha(manifest, "/path/to/alpha_frames")
        Runner(manifest, pipeline_config).run()

    # Multi-GPU: pass devices to to_pipeline_config — everything else is the same
    pipeline_config = config.to_pipeline_config(devices=resolve_devices("all"))

-------------------------------------------------------------------------------
Path 2 — Frame loop (low-level)
-------------------------------------------------------------------------------
Your interface owns the loop. The package provides the stage functions; you
call them one frame at a time in whatever threading model suits your host.

Use this for: DaVinci Resolve Fusion nodes, Premiere/After Effects plugins,
or any host application that manages its own frame scheduling.

    # 1. Load a backend once at startup
    inference_config = config.to_inference_config(device=device)
    backend = load_backend(inference_config)

    # 2. Build stage configs
    preprocess_config = config.to_preprocess_config(device=device)
    postprocess_config = config.to_postprocess_config()
    write_config = config.to_writer_config(manifest.output_dir)

    # 3. Build file lists once per clip
    imgs = get_frame_files(manifest.frames_dir)
    alps = get_frame_files(manifest.alpha_frames_dir)

    # 4. Your host calls this per frame
    for i in range(*manifest.frame_range):
        preprocessed = preprocess_frame(manifest, i, preprocess_config,
                                        image_files=imgs, alpha_files=alps)
        result = backend.run(preprocessed)
        postprocessed = postprocess_frame(result, postprocess_config)
        write_frame(postprocessed, write_config)

-------------------------------------------------------------------------------
Clip state inspection
-------------------------------------------------------------------------------
Resolve what stage a clip has reached based on what is present on disk.
Useful for interfaces that need to resume a session or skip already-complete
clips without re-running them.

    state = resolve_clip_state(clip)   # ClipState.READY / RAW / COMPLETE / ...

-------------------------------------------------------------------------------
Session management
-------------------------------------------------------------------------------
``ClipEntry`` wraps a ``Clip`` with mutable processing state, making it the
right building block for a session list in a GUI or TUI. Construct one per
clip after scanning; the state machine tracks progress through the pipeline.

    entries = [ClipEntry.from_clip(c) for c in result.clips]

    # Check what stage each clip is at
    for entry in entries:
        print(entry.name, entry.state)   # ClipState.RAW / READY / COMPLETE / ...

    # Narrow a clip to an in/out range before processing
    entry.in_out_range = InOutRange(in_point=10, out_point=49)

    # After processing, inspect outputs
    print(entry.completed_frame_count())
    print(entry.has_outputs)

-------------------------------------------------------------------------------
Stages reference
-------------------------------------------------------------------------------
    Stage 0  scan()              -> ScanResult  (Clip, SkippedPath)
    Stage 1  load()              -> ClipManifest
    Stage 2  preprocess_frame()  -> PreprocessedFrame
    Stage 3  backend.run()       -> InferenceResult
    Stage 4  postprocess_frame() -> PostprocessedFrame
    Stage 5  write_frame()       -> files on disk
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__: str = _version("corridorkey")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from corridorkey.errors import (
    ClipScanError,
    CorridorKeyError,
    DeviceError,
    ExtractionError,
    FrameMismatchError,
    FrameReadError,
    InvalidStateTransitionError,
    JobCancelledError,
    ModelError,
    VRAMInsufficientError,
    WriteFailureError,
)
from corridorkey.events import PipelineEvents
from corridorkey.infra import (
    APP_NAME,
    CorridorKeyConfig,
    GPUInfo,
    InferenceSettings,
    LoggingSettings,
    PostprocessSettings,
    PreprocessSettings,
    SettingsMetadata,
    WriterSettings,
    clear_device_cache,
    default_checkpoint_path,
    detect_gpu,
    ensure_config_file,
    ensure_model,
    get_config_path,
    load_config,
    load_config_with_metadata,
    resolve_device,
    resolve_devices,
    setup_logging,
    write_config,
)
from corridorkey.runtime.clip_state import ClipEntry, ClipState, InOutRange, resolve_clip_state
from corridorkey.runtime.runner import PipelineConfig, Runner
from corridorkey.stages.inference import (
    InferenceConfig,
    InferenceResult,
    ModelBackend,
    load_backend,
)
from corridorkey.stages.loader import (
    ClipManifest,
    load,
    resolve_alpha,
)
from corridorkey.stages.loader.validator import get_frame_files
from corridorkey.stages.postprocessor import PostprocessConfig, PostprocessedFrame, postprocess_frame
from corridorkey.stages.preprocessor import (
    FrameMeta,
    PreprocessConfig,
    PreprocessedFrame,
    preprocess_frame,
)
from corridorkey.stages.scanner import Clip, ScanResult, SkippedPath, scan
from corridorkey.stages.writer import WriteConfig, write_frame

__all__ = [
    # ------------------------------------------------------------------ #
    # Package version                                                    #
    # ------------------------------------------------------------------ #
    "__version__",
    # ------------------------------------------------------------------ #
    # Startup                                                            #
    # ------------------------------------------------------------------ #
    "load_config",
    "load_config_with_metadata",
    "SettingsMetadata",
    "write_config",
    "ensure_config_file",
    "get_config_path",
    "setup_logging",
    "APP_NAME",
    # ------------------------------------------------------------------ #
    # Device                                                             #
    # ------------------------------------------------------------------ #
    "resolve_device",
    "resolve_devices",
    "clear_device_cache",
    "detect_gpu",
    "GPUInfo",
    # ------------------------------------------------------------------ #
    # Model hub                                                          #
    # ------------------------------------------------------------------ #
    "ensure_model",
    "default_checkpoint_path",
    # ------------------------------------------------------------------ #
    # Configuration                                                      #
    # ------------------------------------------------------------------ #
    "CorridorKeyConfig",
    "LoggingSettings",
    "PreprocessSettings",
    "InferenceSettings",
    "PostprocessSettings",
    "WriterSettings",
    # ------------------------------------------------------------------ #
    # Path 1 — Managed pipeline                                         #
    # ------------------------------------------------------------------ #
    "scan",
    "load",
    "resolve_alpha",
    "Runner",
    "PipelineConfig",
    "PipelineEvents",
    # ------------------------------------------------------------------ #
    # Path 2 — Frame loop                                                #
    # ------------------------------------------------------------------ #
    "load_backend",
    "preprocess_frame",
    "postprocess_frame",
    "write_frame",
    # ------------------------------------------------------------------ #
    # Stage contracts (both paths)                                       #
    # ------------------------------------------------------------------ #
    # Scanner
    "Clip",
    "ScanResult",
    "SkippedPath",
    # Loader
    "ClipManifest",
    "get_frame_files",
    # Preprocessor
    "PreprocessConfig",
    "PreprocessedFrame",
    "FrameMeta",
    "FrameReadError",
    # Inference
    "InferenceConfig",
    "InferenceResult",
    "ModelBackend",
    # Postprocessor
    "PostprocessConfig",
    "PostprocessedFrame",
    # Writer
    "WriteConfig",
    # ------------------------------------------------------------------ #
    # Clip state inspection & session management                        #
    # ------------------------------------------------------------------ #
    "ClipState",
    "ClipEntry",
    "InOutRange",
    "resolve_clip_state",
    # ------------------------------------------------------------------ #
    # Errors                                                             #
    # ------------------------------------------------------------------ #
    "CorridorKeyError",
    "ClipScanError",
    "ExtractionError",
    "FrameMismatchError",
    "FrameReadError",
    "WriteFailureError",
    "VRAMInsufficientError",
    "DeviceError",
    "ModelError",
    "InvalidStateTransitionError",
    "JobCancelledError",
]
