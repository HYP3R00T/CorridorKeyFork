# Loader Stage

The loader is stage 1. It validates a `Clip` and returns a `LoadResult` with resolved frame paths, output destination, and clip metadata ready for all downstream stages.

Source: [`corridorkey/stages/loader/`](https://github.com/nikopueringer/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/stages/loader/)

## Purpose

The loader is the last stage that touches the filesystem for setup purposes. It extracts video frames if needed, validates frame counts, and creates the output directory. Downstream stages receive only the resolved paths from the manifest - they never re-scan the filesystem.

## Entry Point

```python
from corridorkey import load

manifest = load(clip)        # LoadResult
manifest.needs_alpha         # True if alpha must be generated externally
```

## Steps

### Step 1 - Resolve input frames

`_resolve_frames()` is called for both the input and alpha paths.

For image sequence inputs, the existing directory is returned directly - no files are moved or copied.

For video inputs, frames are extracted into a sibling directory (`Frames/` for input, `AlphaFrames/` for alpha). Before extracting, the loader checks whether a previous extraction is already complete by comparing the frame count on disk against the container's reported frame count. If counts match, extraction is skipped. If they differ (partial extraction from a crashed run), the directory is re-extracted.

### Step 2 - Video extraction

Extraction uses PyAV (Python bindings to FFmpeg C libraries). PyAV bundles its own FFmpeg - no system install is required.

The extractor overlaps decoding and disk writes using a background writer thread. The writer thread pulls `(path, array)` items from a queue and calls `cv2.imwrite`. This keeps the disk busy while the decoder is working, giving a meaningful speedup on fast NVMe storage.

Frames are written as lossless PNGs. The default compression level is 1 (store-only), which is approximately 3x faster than OpenCV's default of 3 with only about 10% larger files for intermediate frames.

Video metadata (fps, resolution, codec, color space) is saved as `video_meta.json` in the clip root so the writer stage can re-encode output with matching properties.

### Step 3 - Validate frame counts

`validate()` performs a single `iterdir()` pass per directory and checks:

- Input directory has at least one image frame.
- If alpha is present, alpha frame count matches input frame count exactly. A mismatch raises `FrameMismatchError`.

The scan result is returned and reused for `frame_count` and `is_linear` - no second directory read.

### Step 4 - Create output directory

`clip/Output/` is created with `mkdir(exist_ok=True)`. Subdirectories (`alpha/`, `fg/`, `comp/`, `processed/`) are created by the writer stage on first write.

### Step 5 - Build and return LoadResult

The manifest is constructed from the resolved paths and scan results. It is frozen - all fields are immutable after construction.

## Alpha Generation Bridge

When `manifest.needs_alpha` is `True`, the interface layer is responsible for generating alpha frames using an external tool. Once done, it calls `resolve_alpha()`:

```python
from corridorkey import resolve_alpha

manifest = resolve_alpha(manifest, alpha_frames_dir)
# manifest.needs_alpha is now False
```

`resolve_alpha()` validates that the provided alpha directory contains the correct number of frames (using the count already in the manifest - no re-scan of the input directory), then returns an updated manifest ready for preprocessing.

Alpha generation is not a pipeline stage. It is entirely the interface's responsibility and runs outside this pipeline.

## Output Contract

`LoadResult` is the single output of this stage and the input to all downstream stages.

```python
class LoadResult(BaseModel):
    clip_name: str
    clip_root: Path
    frames_dir: Path           # Input/ or Frames/ (extracted)
    alpha_frames_dir: Path | None
    output_dir: Path           # clip/Output/
    needs_alpha: bool
    frame_count: int
    frame_range: tuple[int, int]  # half-open (start, end)
    is_linear: bool            # True for .exr inputs
    video_meta_path: Path | None
    png_compression: int
```

## Related

- [Scanner Stage](scanner.md) - Produces the `Clip` consumed here.
- [Preprocessor Stage](preprocessor.md) - Consumes `LoadResult`.
