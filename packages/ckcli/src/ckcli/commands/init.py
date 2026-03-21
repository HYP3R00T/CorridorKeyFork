"""``ck init`` — one-time environment setup."""

from __future__ import annotations

import contextlib

import typer
from rich.prompt import Confirm

from ckcli._console import console, err_console


def init() -> None:
    """Set up CorridorKey for first use.

    - Runs the environment health check
    - Creates the config file if missing
    - Offers to download the inference model if missing
    """
    from corridorkey_new import load_config
    from corridorkey_new.infra import ensure_config_file
    from corridorkey_new.infra.model_hub import MODEL_FILENAME, MODEL_URL, default_checkpoint_path

    console.print("[bold cyan]CorridorKey — Init[/bold cyan]\n")

    # 1. Doctor
    from ckcli.commands.doctor import doctor

    with contextlib.suppress(SystemExit, typer.Exit):
        doctor()

    console.print()

    # 2. Config file
    config_path = ensure_config_file()
    config = load_config()
    console.print(f"[green]Config:[/green] {config_path}")
    console.print()

    # 3. Inference model
    model_path = default_checkpoint_path()
    if model_path.is_file():
        console.print("[green]Inference model found.[/green]")
        console.print("\n[bold green]Init complete. Run `ck wizard` to get started.[/bold green]")
        return

    console.print(f"[yellow]Inference model not found:[/yellow] {model_path}")
    console.print(f"URL: [dim]{MODEL_URL}[/dim]\n")

    if not Confirm.ask("Download inference model now?", default=True):
        console.print(
            f"\nTo download manually, place [bold]{MODEL_FILENAME}[/bold] in:\n"
            f"  {model_path.parent}\n"
            "Then run [bold]ck doctor[/bold] to verify."
        )
        return

    _download_with_progress(model_path)
    console.print("\n[bold green]Init complete. Run `ck wizard` to get started.[/bold green]")


def _download_with_progress(dest_path) -> None:
    from corridorkey_new.infra.model_hub import ensure_model
    from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn, TransferSpeedColumn

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
        transient=True,
    )

    with progress:
        task = progress.add_task("Downloading inference model...", total=None)

        def on_progress(downloaded: int, total: int) -> None:
            progress.update(task, completed=downloaded, total=total or None)

        try:
            dest = ensure_model(dest_dir=dest_path.parent, on_progress=on_progress)
            progress.update(task, completed=1, total=1)
        except RuntimeError as e:
            err_console.print(f"\n[red]Download failed:[/red] {e}")
            raise typer.Exit(1) from e

    console.print(f"[green]Model saved:[/green] {dest}")
