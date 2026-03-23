"""``ck reset`` — delete the CorridorKey config directory."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer

from corridorkey_cli._console import console

_CONFIG_DIR = Path("~/.config/corridorkey")


def reset(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
) -> None:
    """Delete ~/.config/corridorkey and all its contents.

    Removes the config file, downloaded models, and logs.
    Run `ck init` afterwards to set up again.
    """
    target = _CONFIG_DIR.expanduser()

    if not target.exists():
        console.print(f"[yellow]Nothing to remove — {target} does not exist.[/yellow]")
        raise typer.Exit()

    contents = list(target.iterdir())
    console.print(f"\n[bold]This will permanently delete:[/bold] {target}")
    console.print(f"  {len(contents)} item(s) inside, including models and config.\n")

    if not yes and not typer.confirm("Are you sure?", default=False):
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit()

    shutil.rmtree(target)
    console.print(f"[green]Removed {target}[/green]")
    console.print("Run [bold]ck init[/bold] to set up again.")
