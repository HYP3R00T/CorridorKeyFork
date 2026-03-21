"""``ck doctor`` — read-only environment health check."""

from __future__ import annotations

import platform
import sys

import typer
from rich.table import Table

from ckcli._console import console

_PASS = "[green]OK[/green]"
_FAIL = "[red]FAIL[/red]"
_WARN = "[yellow]WARN[/yellow]"


def doctor() -> None:
    """Run a read-only environment health check and print a results table."""
    from corridorkey_new import detect_gpu, load_config
    from corridorkey_new.infra.model_hub import MODEL_FILENAME, default_checkpoint_path

    rows: list[tuple[str, str, str]] = []
    all_ok = True

    # Python version
    major, minor = sys.version_info[:2]
    py_ok = major == 3 and minor >= 13
    rows.append((
        "Python >= 3.13",
        _PASS if py_ok else _FAIL,
        f"{major}.{minor}.{sys.version_info.micro}",
    ))
    if not py_ok:
        all_ok = False

    # Compute device + VRAM
    try:
        gpu = detect_gpu()
        vram_str = ""
        if gpu.vram_gb:
            vram_str = f"  VRAM: {', '.join(f'{v:.1f} GB' for v in gpu.vram_gb)}"
        rows.append(("compute device", _PASS, f"{gpu.backend} ({gpu.vendor}){vram_str}"))
    except Exception as e:
        rows.append(("compute device", _WARN, str(e)))

    # Config file
    config = load_config()
    config_file = config.log_dir.expanduser().parent / "corridorkey.toml"
    config_exists = config_file.exists()
    rows.append((
        "config file",
        _PASS if config_exists else _WARN,
        str(config_file) if config_exists else "not found — run `ck init` to create it",
    ))

    # Inference model
    model_path = default_checkpoint_path()
    model_ok = model_path.is_file()
    rows.append((
        "inference model",
        _PASS if model_ok else _FAIL,
        str(model_path) if model_ok else f"{MODEL_FILENAME} not found — run `ck init`",
    ))
    if not model_ok:
        all_ok = False

    # Platform
    rows.append(("platform", _PASS, f"{platform.system()} {platform.machine()} Python {sys.version.split()[0]}"))

    _render_table(rows)

    if all_ok:
        console.print("\n[green]All checks passed. Ready to run.[/green]")
    else:
        console.print("\n[red]Some checks failed. Run `ck init` to fix setup issues.[/red]")
        raise typer.Exit(1)


def _render_table(rows: list[tuple[str, str, str]]) -> None:
    table = Table(title="Environment Check", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status", justify="center")
    table.add_column("Detail")
    for check, status, detail in rows:
        table.add_row(check, status, detail)
    console.print(table)
