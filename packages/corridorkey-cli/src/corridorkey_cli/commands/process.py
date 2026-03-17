"""``corridorkey process`` - non-interactive batch processing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from corridorkey import (
    CorridorKeyService,
    InferenceParams,
    OutputConfig,
    PipelineResult,
    process_directory,
    validate_job_inputs,
)
from corridorkey.clip_state import ClipState
from rich.table import Table

from corridorkey_cli._helpers import ProgressContext, console, err_console, setup_logging

app = typer.Typer(help="Process all READY clips in a directory.")


@app.callback(invoke_without_command=True)
def process(
    clips_dir: Annotated[Path, typer.Argument(help="Directory containing clips to process.")],
    device: Annotated[str, typer.Option("-device", "-d", help="Compute device: auto, cuda, mps, cpu.")] = "auto",
    optimization_mode: Annotated[
        str, typer.Option("-opt-mode", help="Refiner tiling strategy: auto, speed, lowvram.")
    ] = "auto",
    precision: Annotated[
        str, typer.Option("-precision", help="Inference float format: auto, fp16, bf16, fp32.")
    ] = "auto",
    despill: Annotated[float, typer.Option("-despill", help="Green spill removal strength (0.0-1.0).")] = 1.0,
    despeckle: Annotated[bool, typer.Option("-despeckle/-no-despeckle", help="Remove small matte artifacts.")] = True,
    despeckle_size: Annotated[int, typer.Option("-despeckle-size", help="Min artifact area in pixels.")] = 400,
    refiner: Annotated[float, typer.Option("-refiner", help="Edge refiner scale (0.0 = disabled).")] = 1.0,
    linear: Annotated[bool, typer.Option("-linear", help="Treat input as linear light (not sRGB).")] = False,
    source_passthrough: Annotated[
        bool,
        typer.Option(
            "-source-passthrough/-no-source-passthrough",
            help="Use original source pixels in opaque interior regions.",
        ),
    ] = False,
    edge_erode_px: Annotated[
        int, typer.Option("-edge-erode", help="Interior mask erosion in pixels (source passthrough).")
    ] = 3,
    edge_blur_px: Annotated[
        int, typer.Option("-edge-blur", help="Transition seam blur radius in pixels (source passthrough).")
    ] = 7,
    fg_format: Annotated[str, typer.Option("-fg-format", help="FG output format: exr or png.")] = "exr",
    matte_format: Annotated[str, typer.Option("-matte-format", help="Matte output format: exr or png.")] = "exr",
    comp_format: Annotated[str, typer.Option("-comp-format", help="Comp output format: exr or png.")] = "png",
    exr_compression: Annotated[
        str, typer.Option("-exr-compression", help="EXR compression: dwaa, piz, zip, none.")
    ] = "dwaa",
    no_comp: Annotated[bool, typer.Option("-no-comp", help="Skip comp output.")] = False,
    no_processed: Annotated[bool, typer.Option("-no-processed", help="Skip processed RGBA output.")] = False,
    verbose: Annotated[bool, typer.Option("-verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Process all READY clips in CLIPS_DIR through the keying pipeline.

    Clips in RAW or MASKED state are skipped - they need an alpha generator
    first. Install an alpha generator package (e.g. corridorkey-gbm) to
    process those clips.
    """
    setup_logging(verbose)

    if not clips_dir.exists():
        err_console.print(f"[red]Error:[/red] Path does not exist: {clips_dir}")
        raise typer.Exit(1)

    params = InferenceParams(
        input_is_linear=linear,
        despill_strength=despill,
        auto_despeckle=despeckle,
        despeckle_size=despeckle_size,
        refiner_scale=refiner,
        source_passthrough=source_passthrough,
        edge_erode_px=edge_erode_px,
        edge_blur_px=edge_blur_px,
    )
    output_config = OutputConfig(
        fg_format=fg_format,
        matte_format=matte_format,
        comp_enabled=not no_comp,
        comp_format=comp_format,
        processed_enabled=not no_processed,
        exr_compression=exr_compression,
    )

    # Validate all READY clips before loading the engine.
    _svc = CorridorKeyService()
    _clips = _svc.scan_clips(str(clips_dir))
    _ready = _svc.get_clips_by_state(_clips, ClipState.READY)
    _validation_failed = False
    for _clip in _ready:
        _vr = validate_job_inputs(_clip)
        for _warn in _vr.warnings:
            err_console.print(f"[yellow]Warning:[/yellow] {_warn}")
        if not _vr.ok:
            for _err in _vr.errors:
                err_console.print(f"[red]Validation error:[/red] {_err}")
            _validation_failed = True
    if _validation_failed:
        raise typer.Exit(1)

    with ProgressContext() as prog:
        result = process_directory(
            clips_dir=str(clips_dir),
            params=params,
            output_config=output_config,
            device=device,
            optimization_mode=optimization_mode,
            precision=precision,
            on_progress=prog.on_progress,
            on_warning=prog.on_warning,
            on_clip_start=prog.on_clip_start,
        )

    _print_result(result)

    if result.failed:
        raise typer.Exit(1)


def _print_result(result: PipelineResult) -> None:
    table = Table(title="Pipeline Results", show_header=True, header_style="bold")
    table.add_column("Clip")
    table.add_column("State")
    table.add_column("Frames", justify="right")
    table.add_column("Status")

    for clip in result.clips:
        if clip.error:
            status = f"[red]FAILED: {clip.error}[/red]"
        elif clip.skipped:
            status = "[yellow]SKIPPED[/yellow]"
        else:
            status = "[green]OK[/green]"
        frames = f"{clip.frames_processed}/{clip.frames_total}" if clip.frames_total else "-"
        table.add_row(clip.name, clip.state, frames, status)

    console.print(table)
    console.print(
        f"[bold]Done:[/bold] {len(result.succeeded)} succeeded, "
        f"{len(result.failed)} failed, {len(result.skipped)} skipped"
    )
