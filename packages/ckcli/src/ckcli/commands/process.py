"""``ck process`` — non-interactive batch processing."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ckcli._console import console, err_console
from ckcli._printer import RichPrinter


def process(
    clips_dir: Annotated[Path, typer.Argument(help="Directory containing clip folders to process.")],
    device: Annotated[str, typer.Option("--device", "-d", help="Compute device: auto, cuda, mps, cpu.")] = "auto",
    optimization_mode: Annotated[
        str, typer.Option("--opt-mode", help="Refiner tiling: auto, speed, lowvram.")
    ] = "auto",
    precision: Annotated[
        str, typer.Option("--precision", help="Weight dtype: auto, float16, bfloat16, float32.")
    ] = "auto",
    img_size: Annotated[int, typer.Option("--img-size", help="Model resolution (0 = VRAM-adaptive auto).")] = 0,
    refiner_scale: Annotated[float, typer.Option("--refiner-scale", help="Edge refiner strength 0.0–1.0.")] = 1.0,
    use_refiner: Annotated[bool, typer.Option("--refiner/--no-refiner", help="Enable CNN edge refiner.")] = True,
    mixed_precision: Annotated[
        bool, typer.Option("--mixed-precision/--no-mixed-precision", help="fp16 autocast (ignored on CPU).")
    ] = True,
    clip_index: Annotated[int, typer.Option("--clip", "-c", help="Process only this clip index.")] = -1,
) -> None:
    """Run the CorridorKey keying pipeline on all clips in CLIPS_DIR."""
    from corridorkey_new import detect_gpu, load, load_config, resolve_alpha, resolve_device, scan, setup_logging
    from corridorkey_new.inference import load_model
    from corridorkey_new.infra import ensure_config_file
    from corridorkey_new.infra.config import CorridorKeyConfig
    from corridorkey_new.infra.model_hub import ensure_model
    from corridorkey_new.pipeline import PipelineConfig, PipelineRunner

    if not clips_dir.exists():
        err_console.print(f"[red]Error:[/red] Path does not exist: {clips_dir}")
        raise typer.Exit(1)

    ensure_config_file()
    config = load_config()
    setup_logging(config)

    gpu = detect_gpu()
    console.print_json(gpu.model_dump_json(indent=2))

    # Apply CLI overrides on top of config
    run_config = CorridorKeyConfig.model_validate({
        **config.model_dump(),
        "device": device if device != "auto" else config.device,
        "inference": {
            **config.inference.model_dump(),
            "optimization_mode": optimization_mode,
            "model_precision": precision,
            "refiner_scale": refiner_scale,
            "use_refiner": use_refiner,
            "mixed_precision": mixed_precision,
        },
        "preprocess": {
            **config.preprocess.model_dump(),
            "img_size": img_size,
        },
    })

    resolved_device = resolve_device(run_config.device)
    inference_config = run_config.to_inference_config(device=resolved_device)
    ensure_model(dest_dir=inference_config.checkpoint_path.parent)

    console.print(f"\nLoading model from [cyan]{inference_config.checkpoint_path}[/cyan] ...")
    console.print(
        f"  img_size=[cyan]{inference_config.img_size}[/cyan]  "
        f"precision=[cyan]{inference_config.model_precision}[/cyan]  "
        f"optimization=[cyan]{inference_config.optimization_mode}[/cyan]"
    )
    model = load_model(inference_config)
    console.print("[green]Model loaded.[/green]")

    clips = scan(clips_dir)
    if not clips:
        console.print(f"[yellow]No clips found in {clips_dir}[/yellow]")
        raise typer.Exit()

    targets = [clips[clip_index]] if clip_index >= 0 else clips

    for clip in targets:
        manifest = load(clip)
        console.print_json(manifest.model_dump_json(indent=2))

        if manifest.needs_alpha:
            console.print(f"  Alpha required for '[cyan]{manifest.clip_name}[/cyan]'.")
            raw = typer.prompt("  Enter path to generated alpha frames directory")
            manifest = resolve_alpha(manifest, Path(raw))

        printer = RichPrinter(manifest.frame_count)
        pipeline_config = PipelineConfig(
            preprocess=run_config.to_preprocess_config(
                device=resolved_device, resolved_img_size=inference_config.img_size
            ),
            inference=inference_config,
            model=model,
            events=printer.as_events(),
        )

        console.print(f"\nRunning '[bold]{manifest.clip_name}[/bold]' ({manifest.frame_count} frames)...\n")
        with printer:
            PipelineRunner(manifest, pipeline_config).run()

        console.print(f"\n[green]Done.[/green] Output: [cyan]{manifest.output_dir}[/cyan]")
