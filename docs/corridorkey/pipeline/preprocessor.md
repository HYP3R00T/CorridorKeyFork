# Preprocessor Stage

The preprocessor is stage 2. It reads one frame pair from disk, runs nine ordered steps, and returns a `PreprocessedFrame` containing a model-ready tensor on the configured device.

Source: [`corridorkey/stages/preprocessor/`](https://github.com/nikopueringer/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/stages/preprocessor/)

## Purpose

This stage is the boundary between filesystem I/O (NumPy, CPU) and device compute (PyTorch, CUDA/MPS/CPU). Everything from step 5 onward runs on the configured device. The stage produces no side effects - it reads files and returns a tensor.

## Entry Point

```python
from corridorkey import preprocess_frame

# Build file lists once per clip, pass on every frame call
imgs = list_clip_frames(manifest.frames_dir)
alps = list_clip_frames(manifest.alpha_frames_dir)

for i in range(*manifest.frame_range):
    frame = preprocess_frame(manifest, i, config, image_files=imgs, alpha_files=alps)
```

Passing pre-built file lists avoids an `iterdir()` call on every frame. For a 1000-frame clip this saves 2000 directory reads.

## Steps

### Step 1 - Validate inputs

Checks that `manifest.needs_alpha` is `False` and that the frame index `i` falls within `manifest.frame_range`. Raises `ValueError` immediately if either check fails.

### Step 2 - Read from disk (`reader.py`)

`_read_frame_pair()` reads the image and alpha files using `cv2.imread`. Both are returned as `float32` arrays in range `0.0-1.0`.

The normalisation strategy per source dtype:

- `uint8` - multiply by `1/255` (single allocation, no intermediate)
- `uint16` - multiply by `1/65535` (single allocation, no intermediate)
- `float32` (EXR) - clamp to `[0, 1]`

The image channels are kept in OpenCV's native BGR order. No CPU channel reorder is performed here. The `bgr=True` flag is returned so the reorder can happen on-device in step 5 as a near-zero-cost strided view.

If the alpha dimensions do not match the image dimensions, the alpha is resized with `INTER_LINEAR` and a warning is logged.

### Step 3 - Capture original dimensions

`original_h` and `original_w` are recorded from the image shape before any resizing. These are stored in `FrameMeta` and used by the postprocessor to resize outputs back to source resolution.

### Step 4 - Capture source image and alpha hint (CPU)

If `config.source_passthrough` is `True`, the original sRGB source image is captured as a NumPy array at this point, before the tensor moves to the device. This avoids a GPU-to-CPU transfer on every frame.

If the input is linear (EXR), the source image is converted to sRGB using the shared LUT from `infra.colorspace` before storing.

The raw alpha hint is also captured at source resolution as `[H, W, 1] float32`. This is stored in `FrameMeta` so the postprocessor can build a hard binary mask at native resolution, avoiding soft edges from upscaling.

### Step 5 - Move to device (`tensor.py`)

The image `[3, H, W]` and alpha `[1, H, W]` arrays are concatenated into a single `[4, H, W]` array on CPU, then transferred to the device in one `torch.Tensor.to(device)` call. This produces one DMA operation instead of two, halving the number of PCIe round-trips per frame.

After the transfer, the combined tensor is split back into `img_t [1, 3, H, W]` and `alp_t [1, 1, H, W]`.

The BGR-to-RGB reorder happens here on-device as `img_t[:, [2, 1, 0], :, :]`. On CUDA this is a strided view - no data is copied.

### Step 6 - Color space conversion (`colorspace.py`)

If `manifest.is_linear` is `True` (EXR input), the image tensor is converted from linear light to sRGB using the IEC 61966-2-1 piecewise transfer function. This runs on-device.

The model requires sRGB input. This is an input contract, not an optimisation.

### Step 7 - Resize to model resolution (`resize.py`)

Both image and alpha are resized to `img_size x img_size` using bilinear interpolation (`torch.nn.functional.interpolate` with `mode="bilinear"`). The frame is squished to a square - no aspect ratio preservation, no padding.

This matches the reference pipeline exactly:

```python
cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
cv2.resize(mask,  (img_size, img_size), interpolation=cv2.INTER_LINEAR)
```

The model was trained on bilinear-resized sRGB inputs. Any deviation from this changes the input distribution and degrades output quality.

Alpha is clamped to `[0, 1]` after resize to eliminate floating-point rounding artefacts.

### Step 8 - ImageNet normalisation (`normalise.py`)

The image tensor is normalised with ImageNet mean and std in-place using `sub_` and `div_`. This is a model input contract - the weights were trained exclusively on inputs in this distribution.

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

The mean and std tensors are cached per `(dtype, device)` pair using `functools.lru_cache`. For a 1000-frame clip this avoids 2000 small tensor allocations.

The alpha hint is never normalised - it is passed through as-is.

### Step 9 - Concatenate and return

Image and alpha are concatenated along the channel dimension: `torch.cat([img_t, alp_t], dim=1)` produces `[1, 4, img_size, img_size]`.

If `config.half_precision` is `True`, the tensor is cast to `float16`.

The `PreprocessedFrame` is returned with the tensor and `FrameMeta`.

## Output Contract

```python
@dataclass
class PreprocessedFrame:
    tensor: torch.Tensor  # [1, 4, img_size, img_size] on device
    meta: FrameMeta

@dataclass
class FrameMeta:
    frame_index: int
    original_h: int
    original_w: int
    source_image: np.ndarray | None  # [H, W, 3] float32 sRGB at source res
    alpha_hint: np.ndarray | None    # [H, W, 1] float32 at source res
```

## Related

- [Loader Stage](loader.md) - Produces the `ClipManifest` consumed here.
- [Inference Stage](inference.md) - Consumes `PreprocessedFrame`.
