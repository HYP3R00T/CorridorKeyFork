"""``ck config`` — show and optionally write the resolved configuration."""

from __future__ import annotations

from typing import Annotated

import typer

from corridorkey_cli._config_table import print_config_table
from corridorkey_cli._console import console


def config(
    write: Annotated[bool, typer.Option("--write", "-w", help="Write config file (overwrites if exists).")] = False,
) -> None:
    """Show the resolved configuration. Pass --write to save it to disk."""
    from corridorkey.infra import APP_NAME, get_config_path, load_config_with_metadata
    from utilityhub_config import write_config

    cfg, metadata = load_config_with_metadata()
    config_path = get_config_path(APP_NAME)

    print_config_table(cfg, metadata)
    console.print(
        f"[dim]Sources (lowest → highest): defaults → {config_path} → ./corridorkey.toml → CK_* env vars[/dim]"
    )

    if write:
        written = write_config(cfg, APP_NAME, path=config_path)
        console.print(f"\n[green]Config written:[/green] {written}")
