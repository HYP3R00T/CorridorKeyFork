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
    Output of load(). Input to preprocessing.
    Contains: frames_dir, alpha_frames_dir, output_dir,
              needs_alpha, frame_count, frame_range, is_linear.
"""

from corridorkey_new.entrypoint import Clip, scan
from corridorkey_new.infra import (
    CorridorKeyConfig,
    GPUInfo,
    detect_gpu,
    load_config,
    resolve_device,
    setup_logging,
)
from corridorkey_new.loader import ClipManifest, load, resolve_alpha

__all__ = [
    # Pipeline
    "scan",
    "load",
    "resolve_alpha",
    # Contracts
    "Clip",
    "ClipManifest",
    # Startup
    "load_config",
    "setup_logging",
    "resolve_device",
    "detect_gpu",
    # Config / device types
    "CorridorKeyConfig",
    "GPUInfo",
]
