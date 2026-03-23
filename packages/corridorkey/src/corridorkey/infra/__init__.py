"""Infrastructure — configuration, logging, and device setup.

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
    Validate and resolve a device string to a PyTorch device.
    Pass config.device — returns "cuda", "mps", or "cpu".

detect_gpu() -> GPUInfo
    Probe the system for available GPU hardware. Useful for displaying
    device info to the user before starting a run.

CorridorKeyConfig
    The validated top-level configuration model. Inspect fields to understand
    what can be configured and what the defaults are.

GPUInfo
    Dataclass describing the detected GPU backend, vendor, and VRAM.
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
from corridorkey.infra.device_utils import GPUInfo, detect_gpu, resolve_device
from corridorkey.infra.logging import setup_logging

__all__ = [
    "APP_NAME",
    "load_config",
    "load_config_with_metadata",
    "write_config",
    "ensure_config_file",
    "get_config_path",
    "setup_logging",
    "resolve_device",
    "detect_gpu",
    "CorridorKeyConfig",
    "LoggingSettings",
    "PreprocessSettings",
    "InferenceSettings",
    "PostprocessSettings",
    "WriterSettings",
    "GPUInfo",
]
