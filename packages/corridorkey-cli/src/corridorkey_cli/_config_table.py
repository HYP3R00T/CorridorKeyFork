"""Render a Rich table showing resolved config values with source attribution."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich import box
from rich.table import Table
from rich.text import Text

from corridorkey_cli._console import console

if TYPE_CHECKING:
    from corridorkey import CorridorKeyConfig
    from utilityhub_config.metadata import SettingsMetadata

# Source → colour mapping
_SOURCE_STYLE: dict[str, str] = {
    "defaults": "dim",
    "global": "cyan",
    "project": "cyan",
    "env": "yellow",
    "overrides": "green",
}


def _source_text(source: str, source_path: str | None) -> Text:
    style = _SOURCE_STYLE.get(source, "dim")
    label = source
    if source_path and source != "defaults":
        label = f"{source} ({Path(source_path).name})"
    return Text(label, style=style)


def print_config_table(config: CorridorKeyConfig, metadata: SettingsMetadata) -> None:
    """Print a Rich table of all config fields with their resolved source.

    Iterates the config model dynamically — every field in every section is
    shown automatically, including any new fields added in future. No manual
    registration needed.

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

    # Top-level fields (not nested under a section)
    for field_name, field_value in config.model_dump().items():
        if not isinstance(field_value, dict):
            display = "auto" if field_name == "img_size" and field_value == 0 else field_value
            _add("", field_name, display, field_name)

    # Nested section fields — iterate each sub-model dynamically
    for section_name, section_value in config.model_dump().items():
        if not isinstance(section_value, dict):
            continue
        for field_name, raw_value in section_value.items():
            # Special display: img_size=0 means auto
            display = raw_value or "auto" if section_name == "preprocess" and field_name == "img_size" else raw_value
            _add(section_name, field_name, display, f"{section_name}.{field_name}")

    console.print(table)
