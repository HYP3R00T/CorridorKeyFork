"""CorridorKey pipeline — public API.

This is the single import surface for any interface (CLI, GUI, TUI, web).
Import everything you need from here — do not import from submodules directly.

Pipeline
--------
The pipeline runs in stages. Each stage takes the output of the previous
stage as input. Interfaces orchestrate the stages; they do not implement them.

Stage 0 — scan
    scan(path) -> list[Clip]
    Discover clips from a path. Accepts a clips directory, a single clip
    folder, or a single video file. Returns a list of Clip objects.

Stage 1 — load
    load(clip) -> ClipManifest
    Validate a Clip, extract video to frames if needed, and return a
    ClipManifest with resolved paths and metadata.

    Check ``manifest.needs_alpha`` after this call. If True, alpha frames are
    absent and the interface is responsible for generating them externally
    (e.g. via an alpha generator tool). Once done, call ``resolve_alpha()``
    to update the manifest and proceed to preprocessing.

resolve_alpha — bridge between external alpha generation and preprocessing
    resolve_alpha(manifest, alpha_frames_dir) -> ClipManifest
    Called by the interface after external alpha generation completes.
    Validates the alpha sequence matches the input frame count and returns
    an updated manifest with ``needs_alpha=False``, ready for preprocessing.

    Alpha generation is not a pipeline stage — it is the responsibility of
    the calling interface (CLI, GUI, etc.) and runs outside this pipeline.

Stage 3 — preprocess_frame
    preprocess_frame(manifest, i, config) -> PreprocessedFrame
    Preprocess one frame for model inference. Reads image and alpha from
    disk, converts color space if needed, resizes, applies ImageNet
    normalisation, and returns a tensor on the configured device.

    Build file lists once per clip and pass them on every call:
        imgs = get_frame_files(manifest.frames_dir)
        alps = get_frame_files(manifest.alpha_frames_dir)
        for i in range(*manifest.frame_range):
            result = preprocess_frame(manifest, i, config,
                                      image_files=imgs, alpha_files=alps)

Startup
-------
Before running the pipeline, initialise infrastructure:

    config = load_config()          # load and validate configuration
    setup_logging(config)           # configure file logging for this run
    device = resolve_device(config.device)  # validate and resolve compute device

Contracts
---------
Clip
    Output of scan(). Input to load().

ClipManifest
    Output of load(). Input to preprocess_frame().
    Contains: frames_dir, alpha_frames_dir, output_dir,
              needs_alpha, frame_count, frame_range, is_linear.

PreprocessedFrame
    Output of preprocess_frame(). Input to inference.
    Contains: tensor [1, 4, img_size, img_size] on device + FrameMeta.

FrameMeta
    Original frame dimensions (H, W) and index — carried through to
    postprocessing so outputs can be resized back to source resolution.
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
from corridorkey.infra import (
    APP_NAME,
    CorridorKeyConfig,
    GPUInfo,
    InferenceSettings,
    PreprocessSettings,
    detect_gpu,
    ensure_config_file,
    get_config_path,
    load_config,
    load_config_with_metadata,
    resolve_device,
    setup_logging,
    write_config,
)
from corridorkey.runtime.clip_state import ClipEntry, ClipState, InOutRange
from corridorkey.stages.inference import (
    InferenceConfig,
    InferenceResult,
    load_model,
    run_inference,
)
from corridorkey.stages.loader import ClipManifest, VideoMetadata, load, load_video_metadata, resolve_alpha
from corridorkey.stages.postprocessor import PostprocessConfig, PostprocessedFrame, postprocess_frame
from corridorkey.stages.preprocessor import (
    FrameMeta,
    PreprocessConfig,
    PreprocessedFrame,
    preprocess_frame,
)
from corridorkey.stages.scanner import Clip, scan
from corridorkey.stages.writer import WriteConfig, write_frame

__all__ = [
    # Pipeline
    "scan",
    "load",
    "resolve_alpha",
    "load_video_metadata",
    "preprocess_frame",
    # Contracts
    "Clip",
    "ClipManifest",
    "VideoMetadata",
    "PreprocessConfig",
    "PreprocessedFrame",
    "FrameMeta",
    # Clip state machine
    "ClipState",
    "ClipEntry",
    "InOutRange",
    # Errors
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
    # Startup
    "APP_NAME",
    "load_config",
    "load_config_with_metadata",
    "write_config",
    "ensure_config_file",
    "get_config_path",
    "setup_logging",
    "resolve_device",
    "detect_gpu",
    # Config / device types
    "CorridorKeyConfig",
    "PreprocessSettings",
    "InferenceSettings",
    "GPUInfo",
    # Inference
    "load_model",
    "run_inference",
    "InferenceConfig",
    "InferenceResult",
    # Postprocessor
    "postprocess_frame",
    "PostprocessConfig",
    "PostprocessedFrame",
    # Writer
    "write_frame",
    "WriteConfig",
]
