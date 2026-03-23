"""corridorkey_cli — command-line interface for CorridorKey."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from corridorkey_cli._console import console, err_console
from corridorkey_cli._printer import RichPrinter
from corridorkey_cli.commands.config import config
from corridorkey_cli.commands.init import init
from corridorkey_cli.commands.reset import reset

_ENGINE_PRESET_ALIASES: dict[str, str] = {
    "d": "full_frame",
    "default": "full_frame",
    "s": "full_frame",
    "b": "balanced",
    "q": "quality",
    "mq": "max_quality",
    "max": "max_quality",
    "l": "tiled",
    "m": "manual",
    "man": "manual",
}

_PRESETS: dict[str, tuple[str, str, int]] = {
    "full_frame": ("full_frame", "float16", 1024),
    "balanced": ("auto", "auto", 1536),
    "quality": ("full_frame", "bfloat16", 2048),
    "max_quality": ("full_frame", "float32", 2048),
    "tiled": ("tiled", "float16", 1024),
}

app = typer.Typer(
    name="ck",
    help="CorridorKey — AI green screen keyer.",
    add_completion=False,
    no_args_is_help=False,
)


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    """Fall back to the wizard when no subcommand is given."""
    if ctx.invoked_subcommand is None:
        wizard()


@app.command()
def wizard(
    clips_dir: Annotated[
        Path | None,
        typer.Argument(help="Directory containing clips to process."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip prompts and use config defaults."),
    ] = False,
) -> None:
    """Scan, configure, and process clips. The default command."""
    from corridorkey import load, resolve_alpha, resolve_device, scan, setup_logging
    from corridorkey.infra import APP_NAME, ensure_config_file, load_config_with_metadata
    from corridorkey.infra.config import CorridorKeyConfig
    from corridorkey.infra.model_hub import ensure_model
    from corridorkey.runtime.runner import PipelineRunner
    from corridorkey.stages.inference.loader import load_model as _load_model

    from corridorkey_cli._config_table import print_config_table

    console.print(Panel("[bold cyan]CorridorKey[/bold cyan]", expand=False))

    ensure_config_file(CorridorKeyConfig(), APP_NAME, format="yaml")
    config_obj, metadata = load_config_with_metadata()
    setup_logging(config_obj)
    print_config_table(config_obj, metadata)
    console.print()

    if clips_dir is None:
        if yes:
            err_console.print("[red]Error:[/red] --yes requires a clips directory argument.")
            raise typer.Exit(1)
        raw = Prompt.ask("\nClips directory")
        clips_dir = Path(raw.strip())

    if not clips_dir.exists():
        err_console.print(f"[red]Error:[/red] Path does not exist: {clips_dir}")
        raise typer.Exit(1)

    clips = scan(clips_dir)
    if not clips.clip_count:
        console.print("[yellow]No clips found.[/yellow]")
        raise typer.Exit()

    _print_clip_table(clips, clips_dir)

    if yes:
        opt_mode = config_obj.inference.refiner_mode
        precision = config_obj.inference.model_precision
        img_size = config_obj.preprocess.img_size or 0
    else:
        opt_mode, precision, img_size = _prompt_engine_settings(config_obj)

    device = resolve_device(config_obj.device)

    run_config = CorridorKeyConfig.model_validate({
        **config_obj.model_dump(),
        "inference": {
            **config_obj.inference.model_dump(),
            "refiner_mode": opt_mode,
            "model_precision": precision,
        },
        "preprocess": {
            **config_obj.preprocess.model_dump(),
            "img_size": img_size,
        },
    })

    inference_config, resolved_refiner_mode = run_config.to_inference_config(
        device=device, _return_resolved_refiner_mode=True
    )
    ensure_model(dest_dir=inference_config.checkpoint_path.parent)

    console.print(f"\nLoading model from [cyan]{inference_config.checkpoint_path}[/cyan] ...")
    console.print(
        f"  img_size=[cyan]{inference_config.img_size}[/cyan]  "
        f"precision=[cyan]{inference_config.model_precision}[/cyan]  "
        f"refiner_mode=[cyan]{resolved_refiner_mode}[/cyan]"
    )

    model = _load_model(inference_config, resolved_refiner_mode=resolved_refiner_mode)
    console.print("[green]Model loaded.[/green]\n")

    for clip in clips.clips:
        manifest = load(clip)

        if manifest.needs_alpha:
            console.print(f"  Alpha required for '[cyan]{manifest.clip_name}[/cyan]'.")
            raw = Prompt.ask("  Enter path to generated alpha frames directory")
            manifest = resolve_alpha(manifest, Path(raw))

        printer = RichPrinter(manifest.frame_count)
        pipeline_config = run_config.to_pipeline_config(device=device, model=model)
        pipeline_config.events = printer.as_events()

        console.print(f"Processing '[bold]{manifest.clip_name}[/bold]' ({manifest.frame_count} frames)...\n")
        with printer:
            PipelineRunner(manifest, pipeline_config).run()

        console.print(f"[green]Done.[/green] Output: [cyan]{manifest.output_dir}[/cyan]\n")

    console.print("[bold green]All clips complete.[/bold green]")


app.command("init")(init)
app.command("config")(config)
app.command("reset")(reset)


def main() -> None:
    """Entry point for the ``ck`` console script."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(130)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _print_clip_table(clips, clips_dir: Path) -> None:
    table = Table(title=f"Clips in {clips_dir}", show_header=True, header_style="bold")
    table.add_column("Clip")
    table.add_column("Input")
    table.add_column("Alpha")
    for clip in clips.clips:
        has_alpha = "[green]yes[/green]" if clip.alpha_path else "[dim]none[/dim]"
        table.add_row(clip.name, str(clip.input_path), has_alpha)
    console.print(table)


def _prompt_engine_settings(config_obj) -> tuple[str, str, int]:
    console.print()
    console.print(Panel("[bold]Engine Settings[/bold]", border_style="cyan", expand=False))
    _show_settings([
        ("refiner_mode", config_obj.inference.refiner_mode, "auto / full_frame / tiled"),
        ("model_precision", config_obj.inference.model_precision, "auto / float16 / bfloat16 / float32"),
        ("img_size", str(config_obj.preprocess.img_size or "auto"), "0=auto / 512 / 1024 / 1536 / 2048"),
    ])

    preset = _ask_preset()
    if preset == "manual":
        return _prompt_manual(config_obj)

    opt_mode, precision, img_size = _PRESETS[preset]
    _show_settings([
        ("preset", preset, "selected"),
        ("refiner_mode", opt_mode, "resolved"),
        ("model_precision", precision, "resolved"),
        ("img_size", str(img_size), "resolved"),
    ])
    if not Confirm.ask("Use preset values?", default=True):
        return _prompt_manual(config_obj)

    return opt_mode, precision, img_size


def _prompt_manual(config_obj) -> tuple[str, str, int]:
    opt_mode = Prompt.ask(
        "refiner_mode",
        choices=["auto", "full_frame", "tiled"],
        default=config_obj.inference.refiner_mode,
    )
    precision = Prompt.ask(
        "model_precision",
        choices=["auto", "float16", "bfloat16", "float32"],
        default=config_obj.inference.model_precision,
    )
    img_size_str = Prompt.ask(
        "img_size",
        choices=["0", "512", "1024", "1536", "2048"],
        default=str(config_obj.preprocess.img_size or 0),
    )
    return opt_mode, precision, int(img_size_str)


def _ask_preset() -> str:
    choices = " / ".join(_PRESETS) + " / manual"
    while True:
        raw = Prompt.ask(f"preset [{choices}]", default="full_frame").strip().lower()
        if raw in _PRESETS or raw == "manual":
            return raw
        if raw in _ENGINE_PRESET_ALIASES:
            return _ENGINE_PRESET_ALIASES[raw]
        console.print(
            "[yellow]Invalid preset.[/yellow] Options: full_frame, balanced, quality, max_quality, tiled, manual"
        )


def _show_settings(rows: list[tuple[str, str, str]]) -> None:
    table = Table(show_header=True, header_style="bold", box=box.SIMPLE, padding=(0, 2), pad_edge=True)
    table.add_column("Setting")
    table.add_column("Value")
    table.add_column("Options", style="dim")
    for name, value, options in rows:
        table.add_row(name, f"[cyan]{value}[/cyan]", options)
    console.print(table)
