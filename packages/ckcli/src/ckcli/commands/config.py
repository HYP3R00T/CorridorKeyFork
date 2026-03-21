"""``ck config`` — show and optionally write the resolved configuration."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.table import Table

from ckcli._console import console


def config(
    write: Annotated[bool, typer.Option("--write", "-w", help="Write config file (overwrites if exists).")] = False,
) -> None:
    """Show the resolved configuration. Pass --write to save it to disk."""
    from corridorkey_new.infra import export_config, get_config_path, load_config

    cfg = load_config()
    config_path = get_config_path()

    table = Table(title="CorridorKey Config", show_header=True, header_style="bold cyan")
    table.add_column("Section", style="dim")
    table.add_column("Field")
    table.add_column("Value", style="cyan")

    for field_name in ("device", "log_level", "log_dir"):
        table.add_row("", field_name, str(getattr(cfg, field_name)))

    for section_name in ("preprocess", "inference"):
        section = getattr(cfg, section_name)
        for field_name in section.model_fields:
            table.add_row(f"[{section_name}]", field_name, str(getattr(section, field_name)))

    console.print(table)
    console.print(
        f"\n[dim]Sources (lowest → highest): defaults → {config_path} → ./corridorkey.toml → CK_* env vars[/dim]"
    )

    if write:
        written = export_config(cfg, path=config_path)
        console.print(f"\n[green]Config written:[/green] {written}")
