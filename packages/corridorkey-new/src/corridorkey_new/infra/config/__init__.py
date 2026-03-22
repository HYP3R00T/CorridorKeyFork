"""CorridorKey configuration package.

Public API::

    from corridorkey_new.infra.config import (
        load_config,
        load_config_with_metadata,
        CorridorKeyConfig,
        LoggingSettings,
        PreprocessSettings,
        InferenceSettings,
        PostprocessSettings,
        WriterSettings,
        APP_NAME,
    )

Config file structure (``corridorkey.toml``)::

    [logging]
    level = "INFO"
    dir = "~/.config/corridorkey/logs"

    [preprocess]
    img_size = 0  # 0 = auto-select based on VRAM
    image_upsample_mode = "bicubic"
    sharpen_strength = 0.3
    half_precision = false
    source_passthrough = true

    [inference]
    # checkpoint_path = "~/models/greenformer.pth"
    use_refiner = true
    mixed_precision = true
    model_precision = "auto"
    refiner_mode = "auto"
    refiner_scale = 1.0

    [postprocess]
    fg_upsample_mode = "bicubic"
    alpha_upsample_mode = "lanczos4"
    despill_strength = 1.0
    auto_despeckle = true
    despeckle_size = 400
    source_passthrough = true
    edge_erode_px = 3
    edge_blur_px = 7

    [writer]
    alpha_enabled = true
    alpha_format = "png"
    fg_enabled = true
    fg_format = "png"
    processed_enabled = true
    processed_format = "png"
    comp_enabled = true
    exr_compression = "dwaa"
"""

from corridorkey_new.infra.config._loader import APP_NAME, load_config, load_config_with_metadata
from corridorkey_new.infra.config.inference import InferenceSettings
from corridorkey_new.infra.config.logging import LoggingSettings
from corridorkey_new.infra.config.pipeline import CorridorKeyConfig
from corridorkey_new.infra.config.postprocess import PostprocessSettings
from corridorkey_new.infra.config.preprocess import PreprocessSettings
from corridorkey_new.infra.config.writer import WriterSettings

__all__ = [
    "APP_NAME",
    "load_config",
    "load_config_with_metadata",
    "CorridorKeyConfig",
    "LoggingSettings",
    "PreprocessSettings",
    "InferenceSettings",
    "PostprocessSettings",
    "WriterSettings",
]
