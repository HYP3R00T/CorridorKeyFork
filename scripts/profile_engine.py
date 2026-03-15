"""Profile corridorkey-core inference engine performance.

Requires a real checkpoint and CUDA GPU. Set the checkpoint path via:

    CK_CHECKPOINT_PATH=/path/to/model.pth mise run profile

Optional flags:
    --resolution 1080p|4k     Frame resolution to profile (default: 1080p)
    --frames N                Number of frames to profile (default: 20)
    --no-compile              Disable torch.compile
    --export                  Export Chrome trace to profile_trace.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_CHECKPOINT_ENV = "CK_CHECKPOINT_PATH"

RESOLUTIONS = {
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}

WARM_UP_FRAMES = 3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile corridorkey-core inference engine")
    parser.add_argument("--resolution", choices=list(RESOLUTIONS), default="1080p")
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--export", action="store_true", help="Export Chrome trace to profile_trace.json")
    return parser.parse_args()


def _require_checkpoint() -> Path:
    raw = os.environ.get(_CHECKPOINT_ENV)
    if not raw:
        print(f"Error: set {_CHECKPOINT_ENV} to a valid .pth file", file=sys.stderr)
        sys.exit(1)
    p = Path(raw)
    if not p.is_file():
        print(f"Error: checkpoint not found: {p}", file=sys.stderr)
        sys.exit(1)
    return p


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        print("Error: CUDA GPU required for meaningful profiling", file=sys.stderr)
        sys.exit(1)


def _make_frame(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    image = np.random.rand(height, width, 3).astype(np.float32)
    mask = np.random.rand(height, width).astype(np.float32)
    return image, mask


def _print_header(resolution: str, width: int, height: int, frames: int, ckpt: Path) -> None:
    print()
    print("corridorkey-core engine profiler")
    print(f"  checkpoint : {ckpt.name}")
    print(f"  resolution : {resolution} ({width}x{height})")
    print(f"  frames     : {frames} (+ {WARM_UP_FRAMES} warm-up)")
    print()


def _print_timing_summary(times_ms: list[float]) -> None:
    import statistics

    print("Timing summary (ms/frame)")
    print(f"  mean   : {statistics.mean(times_ms):.1f}")
    print(f"  median : {statistics.median(times_ms):.1f}")
    print(f"  min    : {min(times_ms):.1f}")
    print(f"  max    : {max(times_ms):.1f}")
    print(f"  stdev  : {statistics.stdev(times_ms):.1f}")
    print(f"  fps    : {1000 / statistics.mean(times_ms):.1f}")
    print()


def main() -> None:
    args = _parse_args()
    _require_cuda()

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    ckpt = _require_checkpoint()
    width, height = RESOLUTIONS[args.resolution]

    _print_header(args.resolution, width, height, args.frames, ckpt)

    # Patch torch.compile out if requested
    if args.no_compile:
        torch.compile = lambda model, **kwargs: model  # type: ignore[assignment]
        print("torch.compile disabled")

    from corridorkey_core.inference_engine import CorridorKeyEngine

    print("Loading engine...")
    t0 = time.monotonic()
    engine = CorridorKeyEngine(checkpoint_path=ckpt, device="cuda", img_size=2048)
    print(f"Engine loaded in {time.monotonic() - t0:.1f}s")
    print()

    image, mask = _make_frame(width, height)

    # Warm-up: let torch.compile finish JIT and CUDA streams stabilize
    print(f"Warming up ({WARM_UP_FRAMES} frames)...")
    for _ in range(WARM_UP_FRAMES):
        engine.process_frame(image, mask)
    torch.cuda.synchronize()
    print("Warm-up done")
    print()

    print(f"Timing {args.frames} frames...")
    times_ms: list[float] = []
    for _ in range(args.frames):
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        engine.process_frame(image, mask)
        torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t_start) * 1000)

    _print_timing_summary(times_ms)

    print("Running torch.profiler (5 frames)...")
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        for _ in range(5):
            with record_function("process_frame"):
                engine.process_frame(image, mask)

    print()
    print("Top 20 ops by CUDA time:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("Top 10 ops by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    if args.export:
        trace_path = Path("profile_trace.json")
        prof.export_chrome_trace(str(trace_path))
        print(f"Chrome trace exported to {trace_path}")
        print("Open in chrome://tracing or https://ui.perfetto.dev")

    print()


if __name__ == "__main__":
    main()
