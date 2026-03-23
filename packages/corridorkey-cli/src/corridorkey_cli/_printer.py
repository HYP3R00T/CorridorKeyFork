"""Rich live progress panel for the CorridorKey pipeline."""

from __future__ import annotations

import threading
import time

from corridorkey.events import PipelineEvents
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from corridorkey_cli._console import console


class RichPrinter:
    """Renders a live assembly-line progress panel.

    Thread-safe — all PipelineEvents callbacks fire from worker threads.
    """

    _STAGE_LABELS = {
        "extract": "Extract    ",
        "preprocess": "Preprocess ",
        "inference": "Inference  ",
        "postwrite": "Write      ",
    }

    def __init__(self, total_frames: int) -> None:
        self._total = total_frames
        self._lock = threading.Lock()

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[status]}"),
            console=console,
            transient=False,
        )

        self._tasks: dict[str, TaskID] = {}
        self._inferring: int | None = None
        self._q_pre = 0
        self._q_post = 0
        self._fps_start: dict[str, float] = {}
        self._written = 0

        self._live = Live(self._build_renderable(), console=console, refresh_per_second=10)

    def _build_renderable(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column()
        table.add_row(self._progress)

        parts: list[str] = []
        if self._inferring is not None:
            parts.append(f"[yellow]GPU:[/yellow] frame_{self._inferring:06d}")
        if self._q_pre > 0:
            parts.append(f"[cyan]→ inference queue:[/cyan] {self._q_pre}")
        if self._q_post > 0:
            parts.append(f"[cyan]→ write queue:[/cyan] {self._q_post}")

        table.add_row(Text.from_markup("  ".join(parts) if parts else " "))
        return Panel(table, title="[bold]Pipeline[/bold]", border_style="bright_black")

    def _refresh(self) -> None:
        self._live.update(self._build_renderable())

    def __enter__(self) -> RichPrinter:
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._live.__exit__(exc_type, exc_val, exc_tb)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_stage_start(self, stage: str, total: int) -> None:
        with self._lock:
            label = self._STAGE_LABELS.get(stage, stage)
            tid = self._progress.add_task(label, total=total if total > 0 else self._total, status="")
            self._tasks[stage] = tid
            self._fps_start[stage] = time.monotonic()
            self._refresh()

    def _on_stage_done(self, stage: str) -> None:
        with self._lock:
            if stage in self._tasks:
                self._progress.update(self._tasks[stage], status="[green]✓ done[/green]")
            self._refresh()

    def _on_extract_frame(self, idx: int, total: int) -> None:
        with self._lock:
            if "extract" in self._tasks:
                self._progress.advance(self._tasks["extract"])
            self._refresh()

    def _on_preprocess_queued(self, idx: int) -> None:
        with self._lock:
            if "preprocess" in self._tasks:
                self._progress.advance(self._tasks["preprocess"])
            self._refresh()

    def _on_inference_start(self, idx: int) -> None:
        with self._lock:
            self._inferring = idx
            self._refresh()

    def _on_inference_queued(self, idx: int) -> None:
        with self._lock:
            if "inference" in self._tasks:
                self._progress.advance(self._tasks["inference"])
            self._refresh()

    def _on_frame_written(self, idx: int, total: int) -> None:
        with self._lock:
            self._written += 1
            if "postwrite" in self._tasks:
                elapsed = time.monotonic() - self._fps_start.get("postwrite", time.monotonic())
                fps = self._written / elapsed if elapsed > 0 else 0.0
                self._progress.advance(self._tasks["postwrite"])
                self._progress.update(self._tasks["postwrite"], status=f"[dim]{fps:.2f} fps[/dim]")
            self._refresh()

    def _on_queue_depth(self, pq: int, wq: int) -> None:
        with self._lock:
            self._q_pre = pq
            self._q_post = wq
            self._refresh()

    def _on_frame_error(self, stage: str, idx: int, err: Exception) -> None:
        with self._lock:
            self._live.console.print(f"[red]  ERROR[/red] {stage} frame_{idx:06d}: {err}")

    def as_events(self) -> PipelineEvents:
        return PipelineEvents(
            on_stage_start=self._on_stage_start,
            on_stage_done=self._on_stage_done,
            on_extract_frame=self._on_extract_frame,
            on_preprocess_queued=self._on_preprocess_queued,
            on_inference_start=self._on_inference_start,
            on_inference_queued=self._on_inference_queued,
            on_frame_written=self._on_frame_written,
            on_queue_depth=self._on_queue_depth,
            on_frame_error=self._on_frame_error,
        )
