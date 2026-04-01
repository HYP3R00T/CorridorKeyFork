"""Infrastructure — configuration, logging, device setup, and model hub.

Call these once at application startup before running the pipeline.

Public API
----------
load_config([overrides]) -> CorridorKeyConfig
    Load and validate configuration from config files, environment variables,
    and optional runtime overrides. Always call this first.

setup_logging(config)
    Configure the file handler for this run. The interface layer is
    responsible for adding its own console/GUI handler on top.

resolve_device([requested]) -> str
    Validate and resolve a single device string to a PyTorch device.
    Pass config.device — returns "cuda", "mps", or "cpu".

resolve_devices([requested]) -> list[str]
    Resolve one or more device strings. Pass "all" to get every available
    CUDA device. Use this for multi-GPU runners.

clear_device_cache(device)
    Release GPU memory cache for the given device. Call between clips in
    long-running hosts (GUI, plugin) to avoid VRAM fragmentation.

detect_gpu() -> GPUInfo
    Probe the system for available GPU hardware. Useful for displaying
    device info to the user before starting a run.

ensure_model([dest_dir, on_progress]) -> Path
    Download the model checkpoint if absent, verify its checksum, and
    return the local path. Pass on_progress for GUI download progress.

default_checkpoint_path() -> Path
    Return the expected local path for the default model checkpoint
    (~/.config/corridorkey/models/CorridorKey_v1.0.pth).
"""

from utilityhub_config import ensure_config_file, get_config_path, write_config

from corridorkey.infra.config import (
    APP_NAME,
    CorridorKeyConfig,
    InferenceSettings,
    LoggingSettings,
    PostprocessSettings,
    PreprocessSettings,
    WriterSettings,
    load_config,
    load_config_with_metadata,
)
from corridorkey.infra.device_utils import GPUInfo, clear_device_cache, detect_gpu, resolve_device, resolve_devices
from corridorkey.infra.logging import setup_logging
from corridorkey.infra.model_hub import MODEL_FILENAME, MODEL_URL, default_checkpoint_path, ensure_model

__all__ = [
    # Config loading
    "APP_NAME",
    "load_config",
    "load_config_with_metadata",
    "write_config",
    "ensure_config_file",
    "get_config_path",
    # Logging
    "setup_logging",
    # Device
    "resolve_device",
    "resolve_devices",
    "clear_device_cache",
    "detect_gpu",
    # Config models
    "CorridorKeyConfig",
    "LoggingSettings",
    "PreprocessSettings",
    "InferenceSettings",
    "PostprocessSettings",
    "WriterSettings",
    # GPU info
    "GPUInfo",
    # Model hub
    "ensure_model",
    "default_checkpoint_path",
    "MODEL_URL",
    "MODEL_FILENAME",
]
