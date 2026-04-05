# Scanner Stage

The scanner is stage 0. It accepts a path from the interface layer and produces a `ScanResult` containing valid `Clip` objects and any skipped paths with reasons.

Source: [`corridorkey/stages/scanner/`](https://github.com/nikopueringer/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/stages/scanner/)

## Purpose

This is the only stage that touches the filesystem for discovery purposes, and the only stage that reorganises user files (video normalisation). All other stages receive resolved paths from the scanner's output.

## Entry Point

```python
from corridorkey import scan

result = scan(path)          # ScanResult
result.clips                 # tuple[Clip, ...]
result.skipped               # tuple[SkippedPath, ...]
```

## What It Accepts

The scanner accepts three input shapes:

1. A clips directory containing multiple clip subfolders.
2. A single clip folder (must contain an `Input/` subfolder).
3. A single video file (reorganised in-place into a clip folder structure).

## Steps

### Step 1 - Determine input shape

The scanner checks whether the path is a file, a single clip folder, or a clips directory, and routes to the appropriate handler.

If the path is a file and not a recognised video extension, `ClipScanError` is raised immediately.

### Step 2 - Video normalisation (loose video files only)

When a loose video file is found (either directly or inside a clips directory), `normalise_video()` reorganises it in-place:

```text
parent/clip.mp4
```

becomes:

```text
parent/
  Input/
    clip.mp4        <- moved here
  AlphaHint/        <- created empty
```

The move is performed as copy, size-verify, then delete. This avoids data loss on cross-filesystem moves where a plain rename would fail silently.

The operation is idempotent: if the destination already exists with the same file size, the source is not moved again.

### Step 3 - Clip discovery

For each candidate directory, `try_build_clip()` attempts to build a `Clip`:

1. `find_input()` looks for an `Input/` subfolder (case-insensitive). If it contains exactly one video file, that file is the `input_path`. If it contains no video files, the directory itself is the `input_path` (image sequence). If it contains multiple video files, the clip is skipped as ambiguous.

2. `find_alpha()` looks for an `AlphaHint/` subfolder (case-insensitive) using the same logic. If absent, `alpha_path` is `None` and the interface must generate alpha externally.

3. A `Clip` is constructed and validated. If validation fails, the path is added to `skipped` with a reason.

### Step 4 - Event callbacks

If a `PipelineEvents` instance is provided, `on_clip_found` fires for each valid clip as it is discovered, and `on_clip_skipped` fires for each skipped path. This allows a GUI to stream results as the scan progresses.

## Output Contract

`ScanResult` wraps both the valid clips and the skipped paths. `Clip` is frozen - all fields are immutable after construction.

```python
class Clip(BaseModel):
    name: str                # clip folder name
    root: Path               # absolute path to clip folder
    input_path: Path         # Input/ dir or video file inside Input/
    alpha_path: Path | None  # AlphaHint/ dir or video, None if absent
```

## Related

- [Loader Stage](loader.md) - Consumes `Clip` objects produced here.
- [Clip State Machine](../clip-state.md) - Uses `Clip` to resolve initial state.
