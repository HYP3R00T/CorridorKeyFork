"""``corridorkey stitch`` - stitch output frame sequences into MP4 videos."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from corridorkey import CorridorKeyService
from corridorkey.errors import FFmpegNotFoundError

from corridorkey_cli._helpers import ProgressContext, console, err_console

app = typer.Typer(help="Stitch output frame sequences into MP4 videos.")

_VALID_OUTPUTS = {"fg", "matte", "comp", "processed"}


@app.callback(invoke_without_command=True)
def stitch(
    clips_dir: Annotated[Path, typer.Argument(help="Directory containing clips to stitch.")],
    outputs: Annotated[
        list[str] | None,
        typer.Option(
            "--output",
            "-o",
            help="Output(s) to stitch: fg, matte, comp, processed. Repeatable. Default: all found.",
        ),
    ] = None,
    fps: Annotated[
        float | None,
        typer.Option("--fps", help="Frame rate. Reads from video metadata sidecar when omitted."),
    ] = None,
    codec: Annotated[str, typer.Option("--codec", help="FFmpeg video codec.")] = "libx264",
    crf: Annotated[int, typer.Option("--crf", help="Quality factor (0-51, lower = better).")] = 18,
) -> None:
    """Stitch output frame sequences in CLIPS_DIR into MP4 video files.

    Reads each clip's run manifest to discover which outputs were produced.
    Falls back to scanning Output/ subdirectories when no manifest is present.

    The source frame rate is recovered from the video metadata sidecar written
    during extraction. Supply --fps explicitly for image-sequence source clips.

    Examples:

        corridorkey stitch /path/to/clips

        corridorkey stitch /path/to/clips --output comp --output fg --fps 25
    """
    if not clips_dir.exists():
        err_console.print(f"[red]Error:[/red] Path does not exist: {clips_dir}")
        raise typer.Exit(1)

    # Validate requested output names.
    if outputs:
        invalid = [o for o in outputs if o not in _VALID_OUTPUTS]
        if invalid:
            err_console.print(
                f"[red]Error:[/red] Unknown output(s): {', '.join(invalid)}. Valid: {', '.join(sorted(_VALID_OUTPUTS))}"
            )
            raise typer.Exit(1)

    service = CorridorKeyService()
    clips = service.scan_clips(str(clips_dir))

    if not clips:
        console.print("[yellow]No clips found.[/yellow]")
        return

    total_stitched = 0
    failed: list[str] = []

    for clip in clips:
        console.print(f"\nStitching [cyan]{clip.name}[/cyan]")

        with ProgressContext() as prog:
            try:
                stitched = service.stitch_clip_outputs(
                    clip,
                    outputs=outputs or None,
                    fps=fps,
                    codec=codec,
                    crf=crf,
                    on_progress=prog.on_progress,
                )
            except FFmpegNotFoundError:
                err_console.print(
                    "[red]Error:[/red] ffmpeg not found on PATH. "
                    "Install FFmpeg and ensure it is available before running stitch."
                )
                raise typer.Exit(1) from None
            except Exception as e:
                console.print(f"  [red]Failed:[/red] {e}")
                failed.append(clip.name)
                continue

        if stitched:
            for name, path in stitched.items():
                console.print(f"  [green]{name}[/green] -> {path}")
            total_stitched += len(stitched)
        else:
            console.print("  [yellow]No sequences found to stitch.[/yellow]")

    console.print(f"\n[bold]Done:[/bold] {total_stitched} video(s) written.")
    if failed:
        console.print(f"[red]Failed clips:[/red] {', '.join(failed)}")
        raise typer.Exit(1)
