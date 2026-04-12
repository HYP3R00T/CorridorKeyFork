"""Rich live progress panel for the CorridorKey pipeline."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn
from rich.table import Table

from corridorkey_cli._console import console

if TYPE_CHECKING:
    from corridorkey import Engine


class RichPrinter:
    """Renders a live progress panel."""

    def __init__(self, total_frames: int) -> None:
        self._total = total_frames
        self._lock = threading.Lock()

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("{task.fields[status]}"),
            console=console,
            transient=False,
        )

        self._task: TaskID | None = None
        self._fps_start = 0.0
        self._written = 0
        self._live = Live(self._build_renderable(), console=console, refresh_per_second=10)

    def _build_renderable(self) -> Panel:
        table = Table.grid(padding=(0, 1))
        table.add_column()
        table.add_row(self._progress)
        return Panel(table, title="[bold]Pipeline[/bold]", border_style="bright_black")

    def _refresh(self) -> None:
        self._live.update(self._build_renderable())

    def __enter__(self) -> RichPrinter:
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._live.__exit__(exc_type, exc_val, exc_tb)

    def attach(self, engine: Engine) -> None:
        """Attach printer listeners to the given engine instance."""
        engine.on("clip_loading", self._on_clip_loading)
        engine.on("frame_done", self._on_frame_done)
        engine.on("clip_complete", self._on_clip_complete)
        engine.on("clip_error", self._on_clip_error)

    def _on_clip_loading(self, clip) -> None:
        with self._lock:
            if self._task is not None:
                self._progress.update(self._task, visible=False)
            self._task = self._progress.add_task(f"Processing {clip.name}", total=self._total, status="")
            self._fps_start = time.monotonic()
            self._written = 0
            self._refresh()

    def _on_frame_done(self, idx: int, total: int) -> None:
        with self._lock:
            self._written += 1
            if self._task is not None:
                elapsed = time.monotonic() - self._fps_start
                fps = self._written / elapsed if elapsed > 0 else 0.0
                self._progress.update(
                    self._task, total=total, completed=self._written, status=f"[dim]{fps:.2f} fps[/dim]"
                )
            self._refresh()

    def _on_clip_complete(self, manifest) -> None:
        with self._lock:
            if self._task is not None:
                self._progress.update(self._task, status="[green]✓ done[/green]")
            self._refresh()

    def _on_clip_error(self, stage: str, err: Exception) -> None:
        with self._lock:
            self._live.console.print(f"[red]  ERROR ({stage}):[/red] {err}")
