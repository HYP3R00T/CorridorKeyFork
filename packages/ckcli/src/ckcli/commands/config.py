"""``ck config`` subcommands."""

from __future__ import annotations

import typer
from corridorkey_new.infra import export_config, load_config
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Manage CorridorKey configuration.")

_CONFIG_PATH = "~/.config/corridorkey/corridorkey.toml"

console = Console()


@app.command("show")
def config_show() -> None:
    """Print the resolved configuration (all sources merged)."""
    config = load_config()

    table = Table(title="CorridorKey Config", show_header=True, header_style="bold cyan")
    table.add_column("Section", style="dim")
    table.add_column("Field")
    table.add_column("Value", style="cyan")

    for field_name in ("device", "log_level", "log_dir"):
        table.add_row("", field_name, str(getattr(config, field_name)))

    for section_name in ("preprocess", "inference"):
        section = getattr(config, section_name)
        for field_name in section.model_fields:
            table.add_row(f"[{section_name}]", field_name, str(getattr(section, field_name)))

    console.print(table)
    console.print(
        f"\n[dim]Sources (lowest → highest): defaults → {_CONFIG_PATH} → ./corridorkey.toml → CK_* env vars[/dim]"
    )


@app.command("init")
def config_init(
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing config file."),
) -> None:
    """Write a starter config file to ~/.config/corridorkey/corridorkey.toml."""
    from pathlib import Path

    dest = Path(_CONFIG_PATH).expanduser()

    if dest.exists() and not force:
        console.print(f"[yellow]Config already exists:[/yellow] {dest}")
        console.print("[dim]Pass --force to overwrite.[/dim]")
        return

    config = load_config()
    written = export_config(config, path=dest)
    console.print(f"[green]Config written:[/green] {written}")
