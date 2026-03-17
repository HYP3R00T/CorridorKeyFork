"""``corridorkey wizard`` - interactive processing wizard (default entry point)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer
from corridorkey import (
    ClipEntry,
    ClipState,
    CorridorKeyConfig,
    CorridorKeyService,
    InferenceParams,
    OutputConfig,
    detect_unstructured,
    load_config,
    organize_clips,
)
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from corridorkey_cli._helpers import ProgressContext, console, err_console

app = typer.Typer(help="Interactive processing wizard.")

_STATE_COLOURS: dict[str, str] = {
    "EXTRACTING": "orange3",
    "RAW": "white",
    "MASKED": "blue",
    "READY": "yellow",
    "COMPLETE": "green",
    "ERROR": "red",
}


@app.callback(invoke_without_command=True)
def wizard(
    clips_dir: Annotated[
        Path | None,
        typer.Argument(help="Directory to process. Prompted interactively if omitted."),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip all prompts and use config defaults."),
    ] = False,
) -> None:
    """Interactive wizard: scan -> review -> configure -> process -> stitch.

    Run with no arguments for a fully guided experience.
    Pass --yes to skip all prompts and run with config defaults.
    """
    console.print(Panel("[bold cyan]CorridorKey[/bold cyan]", expand=False))

    from corridorkey_cli._helpers import setup_logging

    log_path = setup_logging(verbose=False)
    if log_path:
        console.print(f"[dim]Logging to {log_path}[/dim]")

    if clips_dir is None:
        if yes:
            err_console.print("[red]Error:[/red] --yes requires a clips directory argument.")
            raise typer.Exit(1)
        raw = Prompt.ask("\nClips directory")
        clips_dir = Path(raw.strip())

    if not clips_dir.exists():
        err_console.print(f"[red]Error:[/red] Path does not exist: {clips_dir}")
        raise typer.Exit(1)

    config = load_config()
    service = CorridorKeyService(config)

    while True:
        if not yes:
            _offer_organize(clips_dir)

        clips = service.scan_clips(str(clips_dir))
        _print_state_table(clips, clips_dir)

        ready = [c for c in clips if c.state == ClipState.READY]
        extracting = [c for c in clips if c.state == ClipState.EXTRACTING]
        skippable = [c for c in clips if c.state in (ClipState.RAW, ClipState.MASKED)]
        errors = [c for c in clips if c.state == ClipState.ERROR]

        if skippable:
            console.print(
                f"[yellow]{len(skippable)} clip(s) in RAW/MASKED state.[/yellow] "
                "Install an alpha generator package to process these."
            )
        if errors:
            console.print(f"[red]{len(errors)} clip(s) in ERROR state.[/red] Inspect errors above.")

        actionable = ready + extracting

        if not actionable:
            console.print("\n[yellow]No processable clips found.[/yellow]")
            if yes:
                break
            if not Confirm.ask("Re-scan?", default=False):
                break
            continue

        if yes:
            params, output_config, device, opt_mode, precision = _defaults_from_config(config)
        else:
            label_parts = []
            if ready:
                label_parts.append(f"{len(ready)} READY")
            if extracting:
                label_parts.append(f"{len(extracting)} to extract")
            actions: list[tuple[str, str]] = [
                ("p", f"Process {', '.join(label_parts)} clip(s)"),
                ("r", "Re-scan directory"),
                ("q", "Quit"),
            ]
            _print_menu(actions)
            choice = Prompt.ask("Select action", choices=[a[0] for a in actions], default="q")

            if choice == "q":
                break
            if choice == "r":
                console.print("[dim]Re-scanning...[/dim]")
                continue

            params, output_config, device, opt_mode, precision = _prompt_settings(config)

        _run_inference(service, actionable, params, output_config, device, opt_mode, precision)

        if output_config.stitch_enabled or (not yes and Confirm.ask("\nStitch outputs to video?", default=True)):
            _run_stitch(service, actionable, output_config)

        if yes:
            break

        Prompt.ask("\nPress Enter to re-scan")

    console.print("\n[bold green]Done.[/bold green]")


# ---------------------------------------------------------------------------
# Settings helpers
# ---------------------------------------------------------------------------


def _defaults_from_config(
    config: CorridorKeyConfig,
) -> tuple[InferenceParams, OutputConfig, str, str, str]:
    params = InferenceParams(
        input_is_linear=config.input_is_linear,
        despill_strength=config.despill_strength,
        auto_despeckle=config.auto_despeckle,
        despeckle_size=config.despeckle_size,
        refiner_scale=config.refiner_scale,
        source_passthrough=config.source_passthrough,
        edge_erode_px=config.edge_erode_px,
        edge_blur_px=config.edge_blur_px,
    )
    output_config = OutputConfig(
        fg_format=config.fg_format,
        matte_format=config.matte_format,
        comp_format=config.comp_format,
        processed_format=config.processed_format,
        exr_compression=config.exr_compression,
        stitch_enabled=True,
    )
    return params, output_config, config.device, config.optimization_mode, config.precision


def _prompt_settings(
    config: CorridorKeyConfig,
) -> tuple[InferenceParams, OutputConfig, str, str, str]:
    """Walk through three grouped settings sections."""

    # Group 1: Device & Engine
    console.print()
    console.print(Panel("[bold]Device & Engine[/bold]", border_style="cyan", expand=False))
    _show_group([
        ("device", config.device, "auto / cuda / mps / cpu"),
        ("optimization_mode", config.optimization_mode, "auto / speed / lowvram"),
        ("precision", config.precision, "auto / fp16 / bf16 / fp32"),
    ])
    if Confirm.ask("Accept device settings?", default=True):
        device, opt_mode, precision = config.device, config.optimization_mode, config.precision
    else:
        device = Prompt.ask("device", choices=["auto", "cuda", "mps", "cpu"], default=config.device)
        opt_mode = Prompt.ask(
            "optimization_mode", choices=["auto", "speed", "lowvram"], default=config.optimization_mode
        )
        precision = Prompt.ask("precision", choices=["auto", "fp16", "bf16", "fp32"], default=config.precision)

    # Group 2: Inference & Postprocess
    console.print()
    console.print(Panel("[bold]Inference & Postprocess[/bold]", border_style="cyan", expand=False))
    _show_group([
        ("input_is_linear", str(config.input_is_linear), "True if input is EXR/linear light"),
        ("despill_strength", str(config.despill_strength), "0.0–1.0, green spill removal"),
        ("auto_despeckle", str(config.auto_despeckle), "remove small matte artifacts"),
        ("despeckle_size", str(config.despeckle_size), "max artifact area in pixels"),
        ("refiner_scale", str(config.refiner_scale), "0.0–1.0, edge refiner strength"),
        ("source_passthrough", str(config.source_passthrough), "use original pixels in opaque regions"),
    ])
    if Confirm.ask("Accept inference settings?", default=True):
        input_is_linear = config.input_is_linear
        despill_strength = config.despill_strength
        auto_despeckle = config.auto_despeckle
        despeckle_size = config.despeckle_size
        refiner_scale = config.refiner_scale
        source_passthrough = config.source_passthrough
        edge_erode_px = config.edge_erode_px
        edge_blur_px = config.edge_blur_px
    else:
        colorspace = Prompt.ask(
            "input colorspace", choices=["linear", "srgb"], default="linear" if config.input_is_linear else "srgb"
        )
        input_is_linear = colorspace == "linear"
        despill_int = IntPrompt.ask("despill_strength (0–10, 10=max)", default=int(config.despill_strength * 10))
        despill_strength = max(0, min(10, despill_int)) / 10.0
        auto_despeckle = Confirm.ask("auto_despeckle?", default=config.auto_despeckle)
        despeckle_size = (
            IntPrompt.ask("despeckle_size (pixels)", default=config.despeckle_size)
            if auto_despeckle
            else config.despeckle_size
        )
        refiner_int = IntPrompt.ask("refiner_scale (0–10, 10=full)", default=int(config.refiner_scale * 10))
        refiner_scale = max(0, min(10, refiner_int)) / 10.0
        source_passthrough = Confirm.ask("source_passthrough?", default=config.source_passthrough)
        edge_erode_px = (
            IntPrompt.ask("edge_erode_px", default=config.edge_erode_px) if source_passthrough else config.edge_erode_px
        )
        edge_blur_px = (
            IntPrompt.ask("edge_blur_px", default=config.edge_blur_px) if source_passthrough else config.edge_blur_px
        )

    params = InferenceParams(
        input_is_linear=input_is_linear,
        despill_strength=despill_strength,
        auto_despeckle=auto_despeckle,
        despeckle_size=despeckle_size,
        refiner_scale=refiner_scale,
        source_passthrough=source_passthrough,
        edge_erode_px=edge_erode_px,
        edge_blur_px=edge_blur_px,
    )

    # Group 3: Output Formats
    console.print()
    console.print(Panel("[bold]Output Formats[/bold]", border_style="cyan", expand=False))
    _show_group([
        ("fg_format", config.fg_format, "exr / png"),
        ("matte_format", config.matte_format, "exr / png"),
        ("comp_format", config.comp_format, "exr / png"),
        ("processed_format", config.processed_format, "exr / png"),
        ("exr_compression", config.exr_compression, "dwaa / piz / zip / none"),
    ])
    if Confirm.ask("Accept output format settings?", default=True):
        fg_format = config.fg_format
        matte_format = config.matte_format
        comp_format = config.comp_format
        processed_format = config.processed_format
        exr_compression = config.exr_compression
    else:
        fg_format = Prompt.ask("fg_format", choices=["exr", "png"], default=config.fg_format)
        matte_format = Prompt.ask("matte_format", choices=["exr", "png"], default=config.matte_format)
        comp_format = Prompt.ask("comp_format", choices=["exr", "png"], default=config.comp_format)
        processed_format = Prompt.ask("processed_format", choices=["exr", "png"], default=config.processed_format)
        exr_compression = Prompt.ask(
            "exr_compression", choices=["dwaa", "piz", "zip", "none"], default=config.exr_compression
        )

    output_config = OutputConfig(
        fg_format=fg_format,
        matte_format=matte_format,
        comp_format=comp_format,
        processed_format=processed_format,
        exr_compression=exr_compression,
        stitch_enabled=False,  # wizard handles stitch separately
    )
    return params, output_config, device, opt_mode, precision


def _show_group(rows: list[tuple[str, str, str]]) -> None:
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("Setting")
    table.add_column("Current")
    table.add_column("Options", style="dim")
    for name, value, options in rows:
        table.add_row(name, f"[cyan]{value}[/cyan]", options)
    console.print(table)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _offer_organize(clips_dir: Path) -> None:
    loose_videos, unstructured_dirs = detect_unstructured(str(clips_dir))
    if not loose_videos and not unstructured_dirs:
        return
    console.print()
    if loose_videos:
        console.print(f"[yellow]{len(loose_videos)} loose video file(s) found:[/yellow]")
        for v in loose_videos:
            console.print(f"  - {v}")
    if unstructured_dirs:
        console.print(f"[yellow]{len(unstructured_dirs)} folder(s) with unstructured content:[/yellow]")
        for d in unstructured_dirs:
            console.print(f"  - {d}")
    console.print(
        "\n[dim]These will be restructured into clip folders with [bold]Input/[/bold] "
        "and empty [bold]AlphaHint/[/bold] subdirectories.[/dim]"
    )
    if Confirm.ask("Organise now?", default=False):
        n = organize_clips(str(clips_dir))
        console.print(f"[green]Organised {n} clip(s).[/green]")


def _print_state_table(clips: list[ClipEntry], clips_dir: Path) -> None:
    if not clips:
        console.print(f"[yellow]No clips found in {clips_dir}[/yellow]")
        return
    table = Table(title=f"Clips in {clips_dir}", show_header=True, header_style="bold")
    table.add_column("Clip")
    table.add_column("State")
    table.add_column("Input", justify="right")
    table.add_column("Alpha", justify="right")
    table.add_column("Error")
    for clip in clips:
        colour = _STATE_COLOURS.get(clip.state.value, "white")
        table.add_row(
            clip.name,
            f"[{colour}]{clip.state.value}[/{colour}]",
            str(clip.input_asset.frame_count) if clip.input_asset else "-",
            str(clip.alpha_asset.frame_count) if clip.alpha_asset else "-",
            f"[red]{clip.error_message}[/red]" if clip.error_message else "",
        )
    console.print(table)


def _print_menu(actions: list[tuple[str, str]]) -> None:
    lines = [f"  [[bold]{key}[/bold]] {label}" for key, label in actions]
    console.print(Panel("\n".join(lines), title="Actions", border_style="blue"))


def _stage(n: int, label: str, skipped: bool = False) -> str:
    if skipped:
        return f"[dim]Stage {n}: Skipped ({label})[/dim]"
    return f"[bold]Stage {n}:[/bold] {label}"


# ---------------------------------------------------------------------------
# Processing helpers
# ---------------------------------------------------------------------------


def _run_inference(
    service: CorridorKeyService,
    clips: list[ClipEntry],
    params: InferenceParams,
    output_config: OutputConfig,
    device: str,
    optimization_mode: str,
    precision: str,
) -> None:
    from rich.progress import Progress as RichProgress
    from rich.progress import SpinnerColumn, TextColumn

    failed: list[str] = []

    active_device, active_opt_mode, active_precision = service.configure_engine_settings(
        device=device,
        optimization_mode=optimization_mode,
        precision=precision,
    )
    console.print(
        "[dim]Engine request: "
        f"device={active_device}, optimization={active_opt_mode}, precision={active_precision}[/dim]"
    )

    for clip in clips:
        if clip.state != ClipState.EXTRACTING:
            continue
        total = clip.input_asset.frame_count if clip.input_asset else 0
        console.print(f"\nClip [cyan]{clip.name}[/cyan]")
        console.print(f"  {_stage(1, f'Extracting frames ({total} frames)')}")
        with ProgressContext() as prog:
            try:
                service.extract_clip(clip, on_progress=prog.on_progress)
            except Exception as e:
                console.print(f"  [red]Extraction failed:[/red] {e}")
                failed.append(clip.name)

    ready_clips = [c for c in clips if c.state == ClipState.READY]
    if not ready_clips:
        if failed:
            console.print(f"\n[red]Failed clips:[/red] {', '.join(failed)}")
        else:
            console.print("\n[yellow]No READY clips after extraction.[/yellow]")
        return

    if not service.is_engine_loaded():
        with RichProgress(
            SpinnerColumn(),
            TextColumn("[cyan]Loading model (first run compiles kernels, ~1 min)...[/cyan]"),
            console=console,
            transient=True,
        ) as spin:
            spin.add_task("")
            service.load_engine()

    runtime_cfg = service.get_engine_runtime_config()
    if runtime_cfg is not None:
        console.print(
            "[dim]Engine resolved: "
            f"backend={runtime_cfg['backend']}, "
            f"device={runtime_cfg['device']}, "
            f"optimization={runtime_cfg['optimization_mode']}, "
            f"precision={runtime_cfg['precision']}[/dim]"
        )
        if runtime_cfg["device"] == "cpu":
            console.print(
                "[yellow]CPU mode detected: inference will be significantly slower than CUDA/MPS.[/yellow]"
            )

    for clip in ready_clips:
        total = clip.input_asset.frame_count if clip.input_asset else 0
        has_alpha = clip.alpha_asset is not None and (clip.alpha_asset.frame_count or 0) > 0
        console.print(f"\nClip [cyan]{clip.name}[/cyan]  ({total} frames)")
        console.print(f"  {_stage(1, f'Loading frames ({total} frames)')}")
        console.print(
            f"  {_stage(2, 'alpha hints present', skipped=has_alpha) if has_alpha else _stage(2, 'Generating alpha hints')}"
        )
        console.print(f"  {_stage(3, 'Preprocessing')}  /  {_stage(4, 'Inference')}  /  {_stage(5, 'Postprocessing')}")
        with ProgressContext() as prog:
            t0 = time.monotonic()
            try:
                results = service.run_inference(
                    clip,
                    params,
                    on_progress=prog.on_progress,
                    on_warning=prog.on_warning,
                    output_config=output_config,
                )
                elapsed = time.monotonic() - t0
                ok = sum(1 for r in results if r.success)
                fps_rate = ok / elapsed if elapsed > 0 else 0
                console.print(f"  {_stage(6, 'Writing outputs')}")
                console.print(
                    f"  [green]✓ Done:[/green] {ok}/{len(results)} frames  "
                    f"[dim]{elapsed:.1f}s  ({fps_rate:.2f} fps)[/dim]"
                )
            except Exception as e:
                console.print(f"  [red]Failed:[/red] {e}")
                failed.append(clip.name)

    if failed:
        console.print(f"\n[red]Failed clips:[/red] {', '.join(failed)}")
    else:
        console.print("\n[green]All clips processed successfully.[/green]")


def _run_stitch(
    service: CorridorKeyService,
    clips: list[ClipEntry],
    output_config: OutputConfig,
) -> None:
    from corridorkey.errors import FFmpegNotFoundError

    console.print()
    total_stitched = 0
    failed: list[str] = []

    for clip in clips:
        console.print(f"  {_stage(7, f'Stitching [cyan]{clip.name}[/cyan]')}")
        try:
            stitched = service.stitch_clip_outputs(
                clip,
                codec=output_config.stitch_codec,
                crf=output_config.stitch_crf,
            )
            if stitched:
                for name, path in stitched.items():
                    console.print(f"    [green]{name}[/green] -> {path}")
                total_stitched += len(stitched)
            else:
                console.print("    [dim]No sequences found to stitch.[/dim]")
        except FFmpegNotFoundError:
            console.print("    [yellow]FFmpeg not found - skipping stitch.[/yellow]")
            return
        except Exception as e:
            console.print(f"    [red]Stitch failed:[/red] {e}")
            failed.append(clip.name)

    console.print(f"\n[bold]Stitch:[/bold] {total_stitched} video(s) written.")
    if failed:
        console.print(f"[red]Failed:[/red] {', '.join(failed)}")
