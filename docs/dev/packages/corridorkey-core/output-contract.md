# Output Contract

`process_frame` always returns the same dict regardless of which backend is running. This contract is stable across Torch and MLX.

## The Four Keys

| Key | Shape | Dtype | Range | Description |
|---|---|---|---|---|
| `alpha` | `[H, W, 1]` | float32 | 0.0-1.0 | Raw predicted alpha matte, before despeckle |
| `fg` | `[H, W, 3]` | float32 | 0.0-1.0 | Raw foreground color, sRGB straight (unpremultiplied) |
| `comp` | `[H, W, 3]` | float32 | 0.0-1.0 | Preview composite over checkerboard, sRGB |
| `processed` | `[H, W, 4]` | float32 | linear | Final RGBA, linear light, premultiplied |

## What Each Key Is For

`alpha` is the raw model prediction before any post-processing. Use it when you want to apply your own matte cleanup or inspect the unmodified network output.

`fg` is the raw foreground color prediction in sRGB, straight (not premultiplied). Use it when you need to composite the subject yourself against a custom background.

`comp` is a ready-to-display preview. The subject is composited over a grey checkerboard in sRGB. Use it for GUI thumbnails and real-time preview.

`processed` is the final deliverable for EXR export. It is linear light, premultiplied RGBA. This is what gets written to disk by the `corridorkey` Application Layer.

## Colour Space Notes

The pipeline works in two colour spaces depending on the stage:

1. The model receives sRGB input (normalized with ImageNet mean/std).
2. `fg` and `comp` outputs are sRGB.
3. `processed` is linear light. The conversion from sRGB to linear happens inside `process_frame` before premultiplication.

If your source footage is already linear (e.g. OpenEXR), pass `input_is_linear=True` to `process_frame`. The engine will convert to sRGB before feeding the model and keep the rest of the pipeline consistent.

## Premultiplication

`processed` stores premultiplied color. This means each RGB channel has already been multiplied by alpha. When writing to EXR, set the file's premultiplied flag accordingly. When compositing in a DCC tool, use the "premultiplied" or "associated alpha" import mode.

`fg` is straight (unpremultiplied). If you composite `fg` yourself, use the straight-alpha over formula:

```text
result = fg * alpha + bg * (1 - alpha)
```

## Related

- [inference-engine reference](../../api/corridorkey-core/inference-engine.md)
- [Backend selection](backend-selection.md)
