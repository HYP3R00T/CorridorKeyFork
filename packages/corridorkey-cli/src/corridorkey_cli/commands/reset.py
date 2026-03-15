"""corridorkey reset - delete the CorridorKey config directory."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer

from corridorkey_cli._helpers import console


def reset(
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompt.")] = False,
) -> None:
    """Delete the CorridorKey config directory and all its contents.

    Removes ``~/.config/corridorkey`` including the config file, downloaded
    models, and any cached data. Use this to start fresh or recover from a
    broken installation.

    Args:
        yes: If True, skip the confirmation prompt.
    """
    from corridorkey.config import load_config

    try:
        config = load_config()
        target = config.app_dir
    except Exception:
        target = Path("~/.config/corridorkey").expanduser()

    if not target.exists():
        console.print(f"[yellow]Nothing to remove - {target} does not exist.[/yellow]")
        raise typer.Exit()

    contents = list(target.iterdir())
    console.print(f"\n[bold]This will permanently delete:[/bold] {target}")
    console.print(f"  {len(contents)} item(s) inside, including models and config.\n")

    if not yes:
        confirmed = typer.confirm("Are you sure?", default=False)
        if not confirmed:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit()

    shutil.rmtree(target)
    console.print(f"[green]Removed {target}[/green]")
    console.print("Run [bold]corridorkey init[/bold] to set up again.")
