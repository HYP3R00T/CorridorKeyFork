# Writer Stage

The writer is stage 5. It takes a `PostprocessedFrame` and writes all enabled output images to disk under the clip's `Output/` directory.

Source: [`corridorkey/stages/writer/`](https://github.com/nikopueringer/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/stages/writer/)

## Purpose

This is the only stage that writes to the filesystem. It owns format conversion, bit-depth selection, and output directory creation. It has no compute logic - it converts arrays to the correct dtype and calls `cv2.imwrite`.

## Entry Point

```python
from corridorkey import write_frame

write_frame(frame, config)
```

## Output Structure

All outputs are written under `config.output_dir`, which is `clip/Output/`. Subdirectories are created on first write.

```text
Output/
  alpha/      <- alpha matte
  fg/         <- straight sRGB foreground
  processed/  <- premultiplied linear RGBA (primary compositor output)
  comp/       <- checkerboard preview composite
```

## Steps

### Step 1 - Write alpha

If `config.alpha_enabled` is `True`, the alpha matte `[H, W, 1]` is converted to a 3-channel grayscale BGR array by broadcasting the single channel across all three, then written to `alpha/<stem>.<alpha_format>`.

### Step 2 - Write foreground

If `config.fg_enabled` is `True`, the straight sRGB foreground `[H, W, 3]` is converted from RGB to BGR (OpenCV convention) and written to `fg/<stem>.<fg_format>`.

### Step 3 - Write processed RGBA

If `config.processed_enabled` is `True`, the premultiplied linear RGBA `[H, W, 4]` is converted from RGBA to BGRA and written to `processed/<stem>.<processed_format>`.

This output uses higher bit depth than the others to preserve sub-pixel alpha precision for compositing:

- PNG format: written as 16-bit PNG (`uint16`, scaled to 0-65535).
- EXR format: written as `float32` EXR (not half-float, unlike alpha and fg).

### Step 4 - Write composite preview

If `config.comp_enabled` is `True`, the checkerboard composite `[H, W, 3]` is converted from RGB to BGR and written to `comp/<stem>.png`. The composite is always PNG.

## Format Conversion

All non-EXR outputs are clipped to `[0, 1]` before conversion:

- PNG (8-bit): `(clip(arr, 0, 1) * 255).astype(uint8)`
- PNG (16-bit, processed only): `(clip(arr, 0, 1) * 65535).astype(uint16)`
- EXR: written as-is in `float32`. The EXR compression codec is configurable.

## EXR Compression

The `exr_compression` field in `WriteConfig` controls the codec used for all EXR outputs. Supported values:

| Value | Description |
|---|---|
| `"none"` | Uncompressed |
| `"rle"` | Run-length encoding |
| `"zips"` | ZIP, single scanline |
| `"zip"` | ZIP, 16 scanlines |
| `"piz"` | Wavelet compression |
| `"pxr24"` | Lossy 24-bit float |
| `"dwaa"` | DWA lossy (default) |
| `"dwab"` | DWA lossy, larger blocks |

The default `"dwaa"` gives good compression with fast decode. Use `"zip"` for lossless.

## Error Handling

If `cv2.imwrite` returns `False`, `WriteFailureError` is raised with the path. Parent directories are created with `mkdir(parents=True, exist_ok=True)` before each write.

## Output Contract

This stage produces no output contract. It writes files to disk and returns `None`. The `PostprocessedFrame.stem` field determines the filename for all outputs.

## Related

- [Postprocessor Stage](postprocessor.md) - Produces `PostprocessedFrame` consumed here.
- [Configuration](../configuration.md) - `WriterSettings` reference.
