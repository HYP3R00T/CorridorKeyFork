"""Infrastructure — configuration, logging, device setup, and model hub.

Call these once at application startup before running the pipeline.

Public API
----------
load_config([overrides]) -> CorridorKeyConfig
    Load and validate configuration from config files, environment variables,
    and optional runtime overrides. Always call this first.
    ``CorridorKeyConfig`` is the single entry point for all configuration —
    use its ``.to_pipeline_config()``, ``.to_preprocess_config()``,
    ``.to_inference_config()``, ``.to_postprocess_config()``, and
    ``.to_writer_config()`` bridge methods to produce stage configs.
    Never construct internal stage configs directly.

load_config_with_metadata([overrides]) -> tuple[CorridorKeyConfig, SettingsMetadata]
    Like load_config, but also returns per-field source attribution. Use this
    when you need to show the user where each config value came from (e.g. a
    "show config" command or a settings UI that highlights overridden fields).
    ``metadata.get_source("field_name")`` returns a FieldSource with .source
    ("defaults", "global", "project", or "env") and .raw_value.

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
from utilityhub_config.metadata import SettingsMetadata

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
from corridorkey.infra.model_hub import default_checkpoint_path, ensure_model

__all__ = [
    # Config loading
    "APP_NAME",
    "load_config",
    "load_config_with_metadata",
    "SettingsMetadata",
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
]
