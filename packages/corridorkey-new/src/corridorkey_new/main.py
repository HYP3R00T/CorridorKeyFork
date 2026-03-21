"""Development entry point — runs the full pipeline on a local sample clip.

Replace CLIPS_DIR with your own path, or wire this up to the CLI/GUI.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from corridorkey_new import (
    detect_gpu,
    load,
    load_config,
    resolve_alpha,
    resolve_device,
    scan,
    setup_logging,
)
from corridorkey_new.inference import load_model
from corridorkey_new.infra.model_hub import ensure_model
from corridorkey_new.pipeline import PipelineConfig, PipelineEvents, PipelineRunner

CLIPS_DIR = Path(r"C:\Users\Rajes\Downloads\Samples\sample_inputs_mod")

# ---------------------------------------------------------------------------
# Console progress printer
# ---------------------------------------------------------------------------

_BAR_WIDTH = 20


def _bar(done: int, total: int) -> str:
    if total <= 0:
        return f"[{'░' * _BAR_WIDTH}] {done:>4}"
    filled = int(_BAR_WIDTH * done / total)
    pct = int(100 * done / total)
    return f"[{'█' * filled}{'░' * (_BAR_WIDTH - filled)}] {done:>4}/{total} ({pct:>3}%)"


class ConsolePrinter:
    """Renders a live assembly-line view of the pipeline to the terminal.

    One line per stage, updated in-place via ANSI cursor movement.
    Thread-safe — all callbacks fire from worker threads.

    Layout::

        Extract      [████████████████████]  120/120 (100%)
        Preprocess   [████████░░░░░░░░░░░░]   80/120 ( 66%)  queued: 2
        Inference    [████░░░░░░░░░░░░░░░░]   40/120 ( 33%)  running: frame_000039
        Write        [██░░░░░░░░░░░░░░░░░░]   20/120 ( 16%)  queued: 1
    """

    _STAGES = ["extract", "preprocess", "inference", "postwrite"]
    _LABELS = {
        "extract": "Extract    ",
        "preprocess": "Preprocess ",
        "inference": "Inference  ",
        "postwrite": "Write      ",
    }

    def __init__(self, total_frames: int) -> None:
        self._total = total_frames
        self._lock = threading.Lock()

        # per-stage done counters
        self._done: dict[str, int] = dict.fromkeys(self._STAGES, 0)
        # which stages have been started (controls which lines are printed)
        self._active: set[str] = set()
        # current inference frame
        self._inferring: int | None = None
        # latest queue depths
        self._q_preprocess = 0
        self._q_postwrite = 0
        # whether the header has been printed
        self._header_printed = False
        # start time per stage
        self._stage_start: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        """Redraw all active stage lines in-place."""
        if not self._active:
            return

        lines: list[str] = []
        for stage in self._STAGES:
            if stage not in self._active:
                continue
            done = self._done[stage]
            bar = _bar(done, self._total)
            label = self._LABELS[stage]

            extra = ""
            if stage == "preprocess":
                extra = f"  queued→inference: {self._q_preprocess}"
            elif stage == "inference":
                if self._inferring is not None:
                    extra = f"  running: frame_{self._inferring:06d}"
                extra += f"  queued→write: {self._q_postwrite}"
            elif stage == "postwrite":
                elapsed = time.monotonic() - self._stage_start.get(stage, time.monotonic())
                fps = done / elapsed if elapsed > 0 and done > 0 else 0.0
                extra = f"  {fps:.2f} fps"

            lines.append(f"  {label}  {bar}{extra}")

        # Move cursor up by the number of lines we're about to overwrite,
        # then reprint each line.
        n = len(lines)
        print(f"\033[{n}A", end="")
        for line in lines:
            # Clear to end of line, then print.
            print(f"\033[2K{line}")

    def _on_stage_start(self, stage: str, total: int) -> None:
        with self._lock:
            self._active.add(stage)
            self._stage_start[stage] = time.monotonic()
            if not self._header_printed:
                self._header_printed = True
            # Print a blank placeholder line for this new stage.
            print(f"  {self._LABELS.get(stage, stage)}  [{'░' * _BAR_WIDTH}]    0/{self._total} (  0%)")
            self._render()

    def _on_stage_done(self, stage: str) -> None:
        with self._lock:
            self._render()

    def _tick(self, stage: str) -> None:
        """Increment done counter for a stage and redraw."""
        self._done[stage] += 1
        self._render()

    # ------------------------------------------------------------------
    # Public event factory
    # ------------------------------------------------------------------

    def as_events(self) -> PipelineEvents:
        return PipelineEvents(
            on_stage_start=self._on_stage_start,
            on_stage_done=self._on_stage_done,
            on_extract_frame=lambda idx, total: self._with_lock(lambda: self._tick("extract")),
            on_preprocess_queued=lambda idx: self._with_lock(lambda: self._tick("preprocess")),
            on_inference_start=lambda idx: self._with_lock(lambda: self._set_inferring(idx)),
            on_inference_queued=lambda idx: self._with_lock(lambda: self._tick("inference")),
            on_frame_written=lambda idx, total: self._with_lock(lambda: self._tick("postwrite")),
            on_queue_depth=lambda pq, wq: self._with_lock(lambda: self._set_depths(pq, wq)),
            on_frame_error=lambda stage, idx, err: self._print_error(stage, idx, err),
        )

    def _with_lock(self, fn) -> None:
        with self._lock:
            fn()

    def _set_inferring(self, idx: int) -> None:
        self._inferring = idx
        self._render()

    def _set_depths(self, pq: int, wq: int) -> None:
        self._q_preprocess = pq
        self._q_postwrite = wq
        self._render()

    def _print_error(self, stage: str, idx: int, err: Exception) -> None:
        with self._lock:
            print(f"\n  [ERROR] {stage} frame_{idx:06d}: {err}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _generate_alpha_externally(manifest) -> Path:
    print(f"  Alpha required for '{manifest.clip_name}'.")
    print(f"  Run your alpha generator on: {manifest.frames_dir}")
    raw = input("  Enter path to generated alpha frames directory: ").strip()
    return Path(raw)


def main() -> None:
    config = load_config()
    setup_logging(config)

    gpu = detect_gpu()
    print(gpu.model_dump_json(indent=2))

    clips = scan(CLIPS_DIR)
    manifest = load(clips[0])
    print(manifest.model_dump_json(indent=2))

    if manifest.needs_alpha:
        alpha_dir = _generate_alpha_externally(manifest)
        manifest = resolve_alpha(manifest, alpha_dir)
        print(f"alpha resolved: {manifest.alpha_frames_dir}")

    device = resolve_device(config.device)
    inference_config = config.to_inference_config(device=device)

    ensure_model(dest_dir=inference_config.checkpoint_path.parent)

    print(f"\nLoading model from {inference_config.checkpoint_path} ...")
    model = load_model(inference_config)
    print("Model loaded.")

    printer = ConsolePrinter(manifest.frame_count)

    pipeline_config = PipelineConfig(
        preprocess=config.to_preprocess_config(device=device),
        inference=inference_config,
        model=model,
        events=printer.as_events(),
    )

    print(f"\nRunning pipeline for '{manifest.clip_name}' ({manifest.frame_count} frames)...\n")
    PipelineRunner(manifest, pipeline_config).run()
    print(f"\nDone. Output written to: {manifest.output_dir}")
