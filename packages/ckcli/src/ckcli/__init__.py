"""ckcli — command-line interface for CorridorKey."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from ckcli._console import console, err_console
from ckcli._printer import RichPrinter
from ckcli.commands.config import config
from ckcli.commands.init import init
from ckcli.commands.reset import reset

_ENGINE_PRESET_ALIASES: dict[str, str] = {
    "d": "speed",
    "default": "speed",
    "s": "speed",
    "b": "balanced",
    "q": "quality",
    "mq": "max_quality",
    "max": "max_quality",
    "l": "lowvram",
    "m": "manual",
    "man": "manual",
}

# (optimization_mode, model_precision, img_size)
_PRESETS: dict[str, tuple[str, str, int]] = {
    "speed": ("speed", "float16", 1024),
    "balanced": ("auto", "auto", 1536),
    "quality": ("speed", "bfloat16", 2048),
    "max_quality": ("speed", "float32", 2560),
    "lowvram": ("lowvram", "float16", 1024),
}

app = typer.Typer(
    name="ck",
    help="CorridorKey — AI green screen keyer.",
    add_completion=False,
    no_args_is_help=False,
)

app.command("init")(init)
app.command("config")(config)
app.command("reset")(reset)


@app.callback(invoke_without_command=True)
def run(
    ctx: typer.Context,
    clips_dir: Annotated[
        Path | None,
        typer.Argument(help="Directory containing clips to process."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip prompts and use config defaults."),
    ] = False,
) -> None:
    """Process clips in CLIPS_DIR. Prompts interactively if omitted."""
    if ctx.invoked_subcommand is not None:
        return

    from corridorkey_new import load, load_config, resolve_alpha, resolve_device, scan, setup_logging
    from corridorkey_new.inference import load_model
    from corridorkey_new.infra import ensure_config_file
    from corridorkey_new.infra.config import CorridorKeyConfig
    from corridorkey_new.infra.model_hub import ensure_model
    from corridorkey_new.pipeline import PipelineConfig, PipelineRunner

    console.print(Panel("[bold cyan]CorridorKey[/bold cyan]", expand=False))

    ensure_config_file()
    config_obj = load_config()
    setup_logging(config_obj)

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
    if not clips:
        console.print("[yellow]No clips found.[/yellow]")
        raise typer.Exit()

    _print_clip_table(clips, clips_dir)

    # Engine settings
    if yes:
        opt_mode = config_obj.inference.optimization_mode
        precision = config_obj.inference.model_precision
        img_size = config_obj.preprocess.img_size or 0
    else:
        opt_mode, precision, img_size = _prompt_engine_settings(config_obj)

    device = resolve_device(config_obj.device)

    run_config = CorridorKeyConfig.model_validate({
        **config_obj.model_dump(),
        "inference": {
            **config_obj.inference.model_dump(),
            "optimization_mode": opt_mode,
            "model_precision": precision,
        },
        "preprocess": {
            **config_obj.preprocess.model_dump(),
            "img_size": img_size,
        },
    })

    inference_config = run_config.to_inference_config(device=device)
    ensure_model(dest_dir=inference_config.checkpoint_path.parent)

    console.print(f"\nLoading model from [cyan]{inference_config.checkpoint_path}[/cyan] ...")
    console.print(
        f"  img_size=[cyan]{inference_config.img_size}[/cyan]  "
        f"precision=[cyan]{inference_config.model_precision}[/cyan]  "
        f"optimization=[cyan]{inference_config.optimization_mode}[/cyan]"
    )
    model = load_model(inference_config)
    console.print("[green]Model loaded.[/green]\n")

    for clip in clips:
        manifest = load(clip)

        if manifest.needs_alpha:
            console.print(f"  Alpha required for '[cyan]{manifest.clip_name}[/cyan]'.")
            raw = Prompt.ask("  Enter path to generated alpha frames directory")
            manifest = resolve_alpha(manifest, Path(raw))

        printer = RichPrinter(manifest.frame_count)
        pipeline_config = PipelineConfig(
            preprocess=run_config.to_preprocess_config(device=device, resolved_img_size=inference_config.img_size),
            inference=inference_config,
            model=model,
            events=printer.as_events(),
        )

        console.print(f"Processing '[bold]{manifest.clip_name}[/bold]' ({manifest.frame_count} frames)...\n")
        with printer:
            PipelineRunner(manifest, pipeline_config).run()

        console.print(f"[green]Done.[/green] Output: [cyan]{manifest.output_dir}[/cyan]\n")

    console.print("[bold green]All clips complete.[/bold green]")


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
    for clip in clips:
        has_alpha = "[green]yes[/green]" if clip.alpha_path else "[dim]none[/dim]"
        table.add_row(clip.name, str(clip.input_path), has_alpha)
    console.print(table)


def _prompt_engine_settings(config_obj) -> tuple[str, str, int]:
    console.print()
    console.print(Panel("[bold]Engine Settings[/bold]", border_style="cyan", expand=False))
    _show_settings([
        ("optimization_mode", config_obj.inference.optimization_mode, "auto / speed / lowvram"),
        ("model_precision", config_obj.inference.model_precision, "auto / float16 / bfloat16 / float32"),
        ("img_size", str(config_obj.preprocess.img_size or "auto"), "0=auto / 1024 / 1536 / 2048 / 2560"),
    ])

    preset = _ask_preset()
    if preset == "manual":
        return _prompt_manual(config_obj)

    opt_mode, precision, img_size = _PRESETS[preset]
    _show_settings([
        ("preset", preset, "selected"),
        ("optimization_mode", opt_mode, "resolved"),
        ("model_precision", precision, "resolved"),
        ("img_size", str(img_size), "resolved"),
    ])
    if not Confirm.ask("Use preset values?", default=True):
        return _prompt_manual(config_obj)

    return opt_mode, precision, img_size


def _prompt_manual(config_obj) -> tuple[str, str, int]:
    opt_mode = Prompt.ask(
        "optimization_mode",
        choices=["auto", "speed", "lowvram"],
        default=config_obj.inference.optimization_mode,
    )
    precision = Prompt.ask(
        "model_precision",
        choices=["auto", "float16", "bfloat16", "float32"],
        default=config_obj.inference.model_precision,
    )
    img_size_str = Prompt.ask(
        "img_size",
        choices=["0", "1024", "1536", "2048", "2560"],
        default=str(config_obj.preprocess.img_size or 0),
    )
    return opt_mode, precision, int(img_size_str)


def _ask_preset() -> str:
    choices = " / ".join(_PRESETS) + " / manual"
    while True:
        raw = Prompt.ask(f"preset [{choices}]", default="speed").strip().lower()
        if raw in _PRESETS or raw == "manual":
            return raw
        if raw in _ENGINE_PRESET_ALIASES:
            return _ENGINE_PRESET_ALIASES[raw]
        console.print(
            "[yellow]Invalid preset.[/yellow] Options: speed, balanced, quality, max_quality, lowvram, manual"
        )


def _show_settings(rows: list[tuple[str, str, str]]) -> None:
    table = Table(show_header=True, header_style="bold", box=box.SIMPLE, padding=(0, 2), pad_edge=True)
    table.add_column("Setting")
    table.add_column("Value")
    table.add_column("Options", style="dim")
    for name, value, options in rows:
        table.add_row(name, f"[cyan]{value}[/cyan]", options)
    console.print(table)
