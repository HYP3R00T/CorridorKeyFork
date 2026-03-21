"""``ck init`` — one-time environment setup."""

from __future__ import annotations

import platform
import sys

import typer
from rich.prompt import Confirm
from rich.table import Table

from ckcli._console import console, err_console

_PASS = "[green]OK[/green]"
_FAIL = "[red]FAIL[/red]"
_WARN = "[yellow]WARN[/yellow]"


def init() -> None:
    """Set up CorridorKey for first use: health check, config, model download."""
    from corridorkey_new.infra import ensure_config_file
    from corridorkey_new.infra.model_hub import MODEL_FILENAME, MODEL_URL, default_checkpoint_path

    console.print("[bold cyan]CorridorKey — Init[/bold cyan]\n")

    # 1. Health check
    _run_health_check()
    console.print()

    # 2. Config file
    config_path = ensure_config_file()
    console.print(f"[green]Config:[/green] {config_path}\n")

    # 3. Inference model
    model_path = default_checkpoint_path()
    if model_path.is_file():
        console.print("[green]Inference model found.[/green]")
        console.print("\n[bold green]Init complete. Run `ck <clips_dir>` to get started.[/bold green]")
        return

    console.print(f"[yellow]Inference model not found:[/yellow] {model_path}")
    console.print(f"URL: [dim]{MODEL_URL}[/dim]\n")

    if not Confirm.ask("Download inference model now?", default=True):
        console.print(
            f"\nTo download manually, place [bold]{MODEL_FILENAME}[/bold] in:\n"
            f"  {model_path.parent}\n"
            "Then run [bold]ck init[/bold] to verify."
        )
        return

    _download_with_progress(model_path)
    console.print("\n[bold green]Init complete. Run `ck <clips_dir>` to get started.[/bold green]")


def _run_health_check() -> None:
    from corridorkey_new import detect_gpu, load_config
    from corridorkey_new.infra.model_hub import MODEL_FILENAME, default_checkpoint_path

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
    config = load_config()
    config_file = config.log_dir.expanduser().parent / "corridorkey.toml"
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
        str(model_path) if model_ok else f"{MODEL_FILENAME} not found — will offer download",
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
