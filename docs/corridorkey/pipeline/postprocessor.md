# Postprocessor Stage

The postprocessor is stage 4. It takes raw model predictions from the inference stage and produces output-ready NumPy arrays at source resolution.

Source: [`corridorkey/stages/postprocessor/`](https://github.com/nikopueringer/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/stages/postprocessor/)

## Purpose

This stage owns all quality improvements applied after the model forward pass: resizing back to source resolution, hint-guided sharpening, source passthrough, despeckle, despill, and compositing. It produces no side effects - it takes tensors and returns arrays.

## Entry Point

```python
from corridorkey import postprocess_frame

frame = postprocess_frame(result, config)
```

## Steps

### Step 1 - Resize to source resolution (`resize.py`)

The alpha `[1, 1, H_model, W_model]` and fg `[1, 3, H_model, W_model]` tensors are moved from the device to CPU, converted to `float32` NumPy arrays, and resized to `(original_h, original_w)` from `FrameMeta`.

Since the preprocessor squishes the frame to a square with no padding, the postprocessor resizes directly back to source resolution with no crop step.

Resize strategy:

- Downscaling (model resolution larger than source): always `INTER_AREA` for both alpha and fg. Anti-aliased, no ringing.
- Upscaling alpha: configurable, default `lanczos4`.
- Upscaling fg: configurable, default `lanczos4`.

### Step 1.5 - Hint-guided sharpening (`hint_sharpen.py`)

When `config.hint_sharpen` is `True` and `meta.alpha_hint` is not `None`, a hard binary mask derived from the alpha hint is applied to both alpha and fg.

The model output at inference resolution is sharp, but a Lanczos upscale spreads each edge transition over many pixels at high source resolutions. This step restores sharpness by:

1. Binarising the alpha hint at its native resolution (threshold 0.5).
2. Dilating by `hint_sharpen_dilation` pixels to give breathing room so fine model edge detail is not clipped.
3. Upscaling to source resolution with `INTER_NEAREST` to preserve the hard boundary exactly.
4. Multiplying both alpha and fg by the mask. This zeros soft tails outside the hint region and eliminates white FG bleed in the background zone.

This step runs before source passthrough so the passthrough only fills the interior region that the mask has already confirmed as foreground.

### Step 2 - Source passthrough (`composite.py`)

When `config.source_passthrough` is `True` and `meta.source_image` is not `None`, the model's FG prediction in opaque interior regions is replaced with the original source pixels.

The model's FG prediction in semi-transparent edge regions is contaminated by the background colour (green spill, dark fringing). In fully opaque interior regions the source pixel is a better FG estimate than the model.

The process:

1. Build an interior mask from pixels where `alpha > 0.95`.
2. Erode inward by `edge_erode_px` pixels to create a safety buffer around edges.
3. Optionally blur the mask edge by `edge_blur_px` pixels for a soft blend seam.
4. Blend: `output = source * interior + model_fg * (1 - interior)`.

This step runs before despill so that despill is applied to the already-blended FG, including the passed-through source pixels.

### Step 3 - Despeckle (`despeckle.py`)

When `config.auto_despeckle` is `True`, small disconnected foreground regions are removed from the alpha matte using connected-component analysis.

The process:

1. Binarise alpha at threshold 0.5.
2. Run `cv2.connectedComponentsWithStats` with 8-connectivity.
3. Build a keep-mask: retain only components with area >= `despeckle_size` pixels.
4. Dilate the keep-mask by `despeckle_dilation` pixels to recover semi-transparent edge pixels that were excluded by the binary threshold.
5. Blur the mask edge by `despeckle_blur` pixels to soften the removal boundary.
6. Multiply the original alpha by the keep-mask. Removed regions are zeroed; kept regions retain their original alpha values unchanged.

### Step 4 - Green spill removal (`despill.py`)

Green spill is suppressed using a luminance-preserving algorithm. Excess green is redistributed equally to red and blue channels to neutralise the spill without darkening the subject.

```python
green_limit = (r + b) / 2.0
spill_amount = max(g - green_limit, 0.0)

g_new = g - spill_amount
r_new = r + spill_amount * 0.5
b_new = b + spill_amount * 0.5
```

`config.despill_strength` blends between the original and the fully despilled result. `0.0` disables despill entirely, `1.0` applies full suppression.

### Step 5 - Build outputs (`composite.py`)

Two composite outputs are built:

`make_processed()` builds the premultiplied linear RGBA output. This is the primary compositor output. FG is converted from sRGB to linear light, then premultiplied by alpha (`fg_linear * alpha`). Transparent regions are correctly zeroed out, so no black-blob artefacts appear when the file is opened in a compositor.

`make_preview()` builds a checkerboard preview composite for visual QC. FG and the checkerboard background are both converted to linear light, composited (`fg_linear * alpha + bg_linear * (1 - alpha)`), then converted back to sRGB. The checkerboard is cached per `(width, height, checker_size)` to avoid re-generating it on every frame.

### Step 6 - Return ProcessedFrame

All arrays are assembled into a `ProcessedFrame` and returned.

## Debug Dump

When `config.debug_dump` is `True`, PNG snapshots of alpha and fg are written to `output_dir/debug/` after each step. This is useful for diagnosing whether quality issues come from the model or from postprocessing.

## Output Contract

```python
@dataclass
class ProcessedFrame:
    alpha: np.ndarray      # [H, W, 1] float32, linear, range 0-1
    fg: np.ndarray         # [H, W, 3] float32, sRGB straight, range 0-1
    processed: np.ndarray  # [H, W, 4] float32, premultiplied linear RGBA, range 0-1
    comp: np.ndarray       # [H, W, 3] float32, sRGB checkerboard composite, range 0-1
    frame_index: int
    source_h: int
    source_w: int
    stem: str
```

## Related

- [Inference Stage](inference.md) - Produces `InferenceResult` consumed here.
- [Writer Stage](writer.md) - Consumes `ProcessedFrame`.
- [Configuration](../configuration.md) - `PostprocessSettings` reference.
