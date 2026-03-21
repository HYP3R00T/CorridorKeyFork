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
    The validated configuration model. Inspect fields to understand
    what can be configured and what the defaults are.

GPUInfo
    Dataclass describing the detected GPU backend, vendor, and VRAM.
"""

from corridorkey_new.infra.config import (
    CorridorKeyConfig,
    InferenceSettings,
    PreprocessSettings,
    ensure_config_file,
    export_config,
    get_config_path,
    load_config,
)
from corridorkey_new.infra.device_utils import GPUInfo, detect_gpu, resolve_device
from corridorkey_new.infra.logging import setup_logging

__all__ = [
    "load_config",
    "export_config",
    "ensure_config_file",
    "get_config_path",
    "setup_logging",
    "resolve_device",
    "detect_gpu",
    "CorridorKeyConfig",
    "PreprocessSettings",
    "InferenceSettings",
    "GPUInfo",
]
