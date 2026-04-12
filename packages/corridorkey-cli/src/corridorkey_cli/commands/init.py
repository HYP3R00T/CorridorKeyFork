"""``ck init`` — one-time environment setup."""

from __future__ import annotations

import platform
import sys

import typer
from corridorkey import Engine, default_checkpoint_path, detect_gpu
from corridorkey.infra import APP_NAME, get_config_path, load_config_with_metadata
from corridorkey.infra.config import CorridorKeyConfig
from rich.progress import BarColumn, DownloadColumn, Progress, SpinnerColumn, TextColumn, TransferSpeedColumn
from rich.prompt import Confirm
from rich.table import Table
from utilityhub_config import ensure_config_file

from corridorkey_cli._config_table import print_config_table
from corridorkey_cli._console import console, err_console

_PASS = "[green]OK[/green]"
_FAIL = "[red]FAIL[/red]"
_WARN = "[yellow]WARN[/yellow]"


def init() -> None:
    """Set up CorridorKey for first use: health check, config, model download."""
    console.print("[bold cyan]CorridorKey — Init[/bold cyan]\n")

    # 1. Health check
    _run_health_check()
    console.print()

    # 2. Config file
    config_path = ensure_config_file(CorridorKeyConfig(), APP_NAME, format="yaml")
    console.print(f"[green]Config:[/green] {config_path}\n")

    config_obj, metadata = load_config_with_metadata()
    print_config_table(config_obj, metadata)
    console.print()

    # 3. Inference model
    model_path = default_checkpoint_path()
    if model_path.is_file():
        console.print("[green]Inference model found.[/green]")
        console.print("\n[bold green]Init complete. Run `ck <clips_dir>` to get started.[/bold green]")
        return

    console.print(f"[yellow]Inference model not found:[/yellow] {model_path}\n")

    if not Confirm.ask("Download inference model now?", default=True):
        console.print(
            f"\nTo download manually, place the model file in:\n"
            f"  {model_path.parent}\n"
            "Then run [bold]ck init[/bold] to verify."
        )
        return

    _download_with_progress(config_obj)
    console.print("\n[bold green]Init complete. Run `ck <clips_dir>` to get started.[/bold green]")


def _run_health_check() -> None:
    rows: list[tuple[str, str, str]] = []
    all_ok = True

    # Python
    major, minor = sys.version_info[:2]
    py_ok = major == 3 and minor >= 13
    rows.append(("Python >= 3.13", _PASS if py_ok else _FAIL, f"{major}.{minor}.{sys.version_info.micro}"))
    if not py_ok:
        all_ok = False

    # Device + VRAM
    try:
        gpu = detect_gpu()
        vram_str = f"  VRAM: {', '.join(f'{v:.1f} GB' for v in gpu.vram_gb)}" if gpu.vram_gb else ""
        rows.append(("compute device", _PASS, f"{gpu.backend} ({gpu.vendor}){vram_str}"))
    except Exception as e:
        rows.append(("compute device", _WARN, str(e)))

    # Config file
    config_file = get_config_path(APP_NAME, format="yaml")
    rows.append((
        "config file",
        _PASS if config_file.exists() else _WARN,
        str(config_file) if config_file.exists() else "not found — will be created",
    ))

    # Model
    model_path = default_checkpoint_path()
    model_ok = model_path.is_file()
    rows.append((
        "inference model",
        _PASS if model_ok else _WARN,
        str(model_path) if model_ok else "model file not found — will offer download",
    ))

    # Platform
    rows.append(("platform", _PASS, f"{platform.system()} {platform.machine()} Python {sys.version.split()[0]}"))

    table = Table(title="Environment Check", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status", justify="center")
    table.add_column("Detail")
    for check, status, detail in rows:
        table.add_row(check, status, detail)
    console.print(table)

    if all_ok:
        console.print("[green]All checks passed.[/green]")


def _download_with_progress(config_obj) -> None:
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

        engine = Engine(config_obj)
        engine.on("download_progress", on_progress)

        try:
            engine.run([])
            progress.update(task, completed=1, total=1)
        except Exception as e:
            err_console.print(f"\n[red]Download failed:[/red] {e}")
            raise typer.Exit(1) from e

    console.print("[green]Model downloaded & verified successfully.[/green]")
