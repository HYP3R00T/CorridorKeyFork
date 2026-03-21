"""``ck scan`` — show clip states without processing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from ckcli._console import console, err_console


def scan(
    clips_dir: Annotated[Path, typer.Argument(help="Directory to scan for clips.")],
) -> None:
    """Scan CLIPS_DIR and print a clip table. No processing is performed."""
    from corridorkey_new import scan as ck_scan

    if not clips_dir.exists():
        err_console.print(f"[red]Error:[/red] Path does not exist: {clips_dir}")
        raise typer.Exit(1)

    clips = ck_scan(clips_dir)

    if not clips:
        console.print("[yellow]No clips found.[/yellow]")
        return

    table = Table(title=f"Clips in {clips_dir}", show_header=True, header_style="bold")
    table.add_column("Clip")
    table.add_column("Input")
    table.add_column("Alpha")

    for clip in clips:
        has_alpha = "[green]yes[/green]" if clip.alpha_path else "[dim]none[/dim]"
        table.add_row(
            clip.name,
            str(clip.input_path),
            has_alpha,
        )

    console.print(table)
    console.print(f"[bold]{len(clips)}[/bold] clip(s) found.")
