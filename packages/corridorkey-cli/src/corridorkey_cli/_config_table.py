"""Render a Rich table showing resolved config values with source attribution."""

from __future__ import annotations

from pathlib import Path

from rich import box
from rich.table import Table
from rich.text import Text

from corridorkey_cli._console import console

# Source → colour mapping
_SOURCE_STYLE: dict[str, str] = {
    "defaults": "dim",
    "global": "cyan",
    "project": "cyan",
    "env": "yellow",
    "overrides": "green",
}


def _source_text(source: str, source_path: str | None) -> Text:
    style = _SOURCE_STYLE.get(source, "")
    label = source
    if source_path and source != "defaults":
        label = f"{source} ({Path(source_path).name})"
    return Text(label, style=style)


def print_config_table(config, metadata) -> None:
    """Print a Rich table of all config fields with their resolved source.

    Args:
        config: A ``CorridorKeyConfig`` instance.
        metadata: The ``SettingsMetadata`` returned by ``load_config_with_metadata``.
    """
    table = Table(
        title="Active Configuration",
        show_header=True,
        header_style="bold",
        box=box.SIMPLE,
        padding=(0, 1),
    )
    table.add_column("Section", style="dim")
    table.add_column("Field")
    table.add_column("Value", style="cyan")
    table.add_column("Source")

    def _add(section: str, field: str, value: object, meta_key: str) -> None:
        fs = metadata.get_source(meta_key)
        src_text = _source_text(fs.source, fs.source_path) if fs else Text("?", style="dim")
        table.add_row(section, field, str(value), src_text)

    # Top-level
    _add("", "device", config.device, "device")

    # [logging]
    _add("logging", "level", config.logging.level, "logging.level")
    _add("logging", "dir", config.logging.dir, "logging.dir")

    # [preprocess]
    _add("preprocess", "img_size", config.preprocess.img_size or "auto", "preprocess.img_size")
    _add("preprocess", "image_upsample_mode", config.preprocess.image_upsample_mode, "preprocess.image_upsample_mode")
    _add("preprocess", "sharpen_strength", config.preprocess.sharpen_strength, "preprocess.sharpen_strength")

    # [inference]
    _add("inference", "checkpoint_path", config.inference.checkpoint_path, "inference.checkpoint_path")
    _add("inference", "use_refiner", config.inference.use_refiner, "inference.use_refiner")
    _add("inference", "mixed_precision", config.inference.mixed_precision, "inference.mixed_precision")
    _add("inference", "model_precision", config.inference.model_precision, "inference.model_precision")
    _add("inference", "refiner_mode", config.inference.refiner_mode, "inference.refiner_mode")
    _add("inference", "refiner_scale", config.inference.refiner_scale, "inference.refiner_scale")

    # [postprocess]
    _add("postprocess", "fg_upsample_mode", config.postprocess.fg_upsample_mode, "postprocess.fg_upsample_mode")
    _add(
        "postprocess", "alpha_upsample_mode", config.postprocess.alpha_upsample_mode, "postprocess.alpha_upsample_mode"
    )
    _add("postprocess", "despill_strength", config.postprocess.despill_strength, "postprocess.despill_strength")
    _add("postprocess", "auto_despeckle", config.postprocess.auto_despeckle, "postprocess.auto_despeckle")

    # [writer]
    _add("writer", "alpha_format", config.writer.alpha_format, "writer.alpha_format")
    _add("writer", "fg_format", config.writer.fg_format, "writer.fg_format")
    _add("writer", "processed_format", config.writer.processed_format, "writer.processed_format")
    _add("writer", "exr_compression", config.writer.exr_compression, "writer.exr_compression")

    console.print(table)
