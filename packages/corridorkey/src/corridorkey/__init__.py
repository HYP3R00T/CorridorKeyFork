"""CorridorKey pipeline — public API.

This is the single import surface for any interface (CLI, GUI, TUI, plugin).
Import everything you need from here — do not import from submodules directly.

Startup
-------
Call these once before running the pipeline::

    config = load_config()  # load + validate from file / env vars
    setup_logging(config)  # configure file logging for this run
    device = resolve_device(config.device)  # validate compute device

Model
-----
Download the model on first run, then load a backend::

    ensure_model()  # downloads if absent, verifies checksum
    backend = load_backend(config.to_inference_config(device=device))
    print(backend.resolved_config)  # {"backend": "torch", "device": "cuda", ...}

Pipeline — high-level (recommended)
------------------------------------
Use PipelineRunner for a single GPU or MultiGPURunner for parallel GPUs::

    result = scan("/path/to/clips")
    for clip in result.clips:
        manifest = load(clip)
        if manifest.needs_alpha:
            manifest = resolve_alpha(manifest, "/path/to/alpha_frames")
        pipeline_config = config.to_pipeline_config(device=device)
        PipelineRunner(manifest, pipeline_config).run()

Pipeline — low-level (frame loop)
-----------------------------------
For custom integrations (DaVinci node, Premiere plugin) that need per-frame
control::

    preprocess_config = config.to_preprocess_config(device=device)
    postprocess_config = config.to_postprocess_config()
    write_config = config.to_writer_config(manifest.output_dir)

    for i in range(*manifest.frame_range):
        preprocessed = preprocess_frame(manifest, i, preprocess_config)
        result = backend.run(preprocessed)
        postprocessed = postprocess_frame(result, postprocess_config)
        write_frame(postprocessed, write_config)

Events
------
Attach callbacks to any runner for progress reporting::

    events = PipelineEvents(
        on_frame_written=lambda idx, total: print(f"{idx + 1}/{total}"),
        on_frame_error=lambda stage, idx, err: print(f"Error at {stage}:{idx}: {err}"),
    )
    PipelineRunner(manifest, pipeline_config, events=events).run()

Clip state machine
------------------
Track clip lifecycle across multiple runs::

    entry = ClipEntry.from_clip(clip)  # resolves state from disk
    entry.state  # ClipState.READY / RAW / COMPLETE / ...
    entry.transition_to(ClipState.COMPLETE)

Stages
------
    Stage 0  scan()             Clip, ScanResult, SkippedPath
    Stage 1  load()             ClipManifest, VideoMetadata, FrameScan
    Stage 2  preprocess_frame() PreprocessedFrame, FrameMeta
    Stage 3  backend.run()      InferenceResult
    Stage 4  postprocess_frame() PostprocessedFrame
    Stage 5  write_frame()      files on disk
"""

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
    MODEL_FILENAME,
    MODEL_URL,
    CorridorKeyConfig,
    GPUInfo,
    InferenceSettings,
    LoggingSettings,
    PostprocessSettings,
    PreprocessSettings,
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
from corridorkey.runtime.clip_state import ClipEntry, ClipState, InOutRange
from corridorkey.runtime.runner import MultiGPUConfig, MultiGPURunner, PipelineConfig, PipelineRunner
from corridorkey.stages.inference import (
    BackendChoice,
    InferenceConfig,
    InferenceResult,
    ModelBackend,
    RefinerMode,
    TorchBackend,
    discover_checkpoint,
    load_backend,
    load_model,
    run_inference,
)
from corridorkey.stages.inference.config import VALID_IMG_SIZES, adaptive_img_size
from corridorkey.stages.loader import (
    ClipManifest,
    FrameScan,
    VideoMetadata,
    load,
    load_video_metadata,
    resolve_alpha,
    scan_frames,
)
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
    # Startup                                                              #
    # ------------------------------------------------------------------ #
    "APP_NAME",
    "load_config",
    "load_config_with_metadata",
    "write_config",
    "ensure_config_file",
    "get_config_path",
    "setup_logging",
    # ------------------------------------------------------------------ #
    # Device                                                               #
    # ------------------------------------------------------------------ #
    "resolve_device",
    "resolve_devices",
    "clear_device_cache",
    "detect_gpu",
    "GPUInfo",
    # ------------------------------------------------------------------ #
    # Model hub                                                            #
    # ------------------------------------------------------------------ #
    "ensure_model",
    "default_checkpoint_path",
    "MODEL_URL",
    "MODEL_FILENAME",
    # ------------------------------------------------------------------ #
    # Configuration models                                                 #
    # ------------------------------------------------------------------ #
    "CorridorKeyConfig",
    "LoggingSettings",
    "PreprocessSettings",
    "InferenceSettings",
    "PostprocessSettings",
    "WriterSettings",
    # ------------------------------------------------------------------ #
    # Pipeline runners                                                     #
    # ------------------------------------------------------------------ #
    "PipelineRunner",
    "PipelineConfig",
    "MultiGPURunner",
    "MultiGPUConfig",
    "PipelineEvents",
    # ------------------------------------------------------------------ #
    # Pipeline stages — entry points                                       #
    # ------------------------------------------------------------------ #
    "scan",
    "load",
    "resolve_alpha",
    "load_video_metadata",
    "preprocess_frame",
    "load_backend",
    "load_model",
    "run_inference",
    "postprocess_frame",
    "write_frame",
    # ------------------------------------------------------------------ #
    # Stage contracts                                                      #
    # ------------------------------------------------------------------ #
    # Scanner
    "Clip",
    "ScanResult",
    "SkippedPath",
    # Loader
    "ClipManifest",
    "VideoMetadata",
    "FrameScan",
    "scan_frames",
    # Preprocessor
    "PreprocessConfig",
    "PreprocessedFrame",
    "FrameMeta",
    # Inference
    "InferenceConfig",
    "InferenceResult",
    "BackendChoice",
    "RefinerMode",
    "VALID_IMG_SIZES",
    "adaptive_img_size",
    # Inference backends
    "ModelBackend",
    "TorchBackend",
    "discover_checkpoint",
    # Postprocessor
    "PostprocessConfig",
    "PostprocessedFrame",
    # Writer
    "WriteConfig",
    # ------------------------------------------------------------------ #
    # Clip state machine                                                   #
    # ------------------------------------------------------------------ #
    "ClipState",
    "ClipEntry",
    "InOutRange",
    # ------------------------------------------------------------------ #
    # Errors                                                               #
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
