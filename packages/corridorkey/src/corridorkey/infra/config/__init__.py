"""CorridorKey configuration package.

Public API::

    from corridorkey.infra.config import (
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

Bridge methods on ``CorridorKeyConfig``
---------------------------------------
Build stage runtime configs from a loaded ``CorridorKeyConfig``::

    config = load_config()

    inference_config = config.to_inference_config(device=device)
    preprocess_config = config.to_preprocess_config(device=device)
    postprocess_config = config.to_postprocess_config()
    write_config = config.to_writer_config(output_dir)

    # Or build everything at once for the high-level Runner:
    pipeline_config = config.to_pipeline_config(device=device)

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
    fg_upsample_mode = "lanczos4"
    alpha_upsample_mode = "lanczos4"
    despill_strength = 0.5
    auto_despeckle = true
    despeckle_size = 400
    despeckle_dilation = 25
    despeckle_blur = 5
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

from corridorkey.infra.config._loader import APP_NAME, load_config, load_config_with_metadata
from corridorkey.infra.config.inference import InferenceSettings
from corridorkey.infra.config.logging import LoggingSettings
from corridorkey.infra.config.pipeline import CorridorKeyConfig
from corridorkey.infra.config.postprocess import PostprocessSettings
from corridorkey.infra.config.preprocess import PreprocessSettings
from corridorkey.infra.config.writer import WriterSettings

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
