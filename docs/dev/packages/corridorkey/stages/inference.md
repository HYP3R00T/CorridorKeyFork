# Inference Stage

The inference stage is stage 3. It runs the GreenFormer model on a single preprocessed frame and returns an `InferenceResult` with raw alpha and foreground predictions still on the device.

Source: [`corridorkey/stages/inference/`](https://github.com/nikopueringer/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/stages/inference/)

## Purpose

This stage owns the model forward pass, autocast, tiled refiner execution, and VRAM management. It does not own model loading (that is `loader.py`), postprocessing, or writing.

## Entry Points

```python
from corridorkey import load_model, run_inference

model = load_model(config)
result = run_inference(frame, model, config)
```

## Model Loading (`loader.py`)

`load_model()` constructs the GreenFormer architecture, moves it to the configured device, loads the checkpoint, and applies post-load fixups.

### Steps

1. Build `GreenFormer` with the configured `img_size` and `use_refiner` flag.
2. Move to device and cast to `model_precision`.
3. Set `eval()` mode.
4. Enable TF32 on Ampere+ GPUs for faster matmuls with minimal precision loss.
5. Load the checkpoint with `torch.load(..., weights_only=True)`.
6. Strip `_orig_mod.` prefixes added by `torch.compile` from checkpoint keys.
7. Resize position embeddings if the checkpoint was trained at a different `img_size`. Bicubic interpolation is used to reshape the spatial grid.
8. Load state dict with `strict=False` and log any missing or unexpected keys.
9. Keep the CNN refiner and all `BatchNorm2d` layers in `float32`. GroupNorm and BatchNorm do not support `bf16`/`fp16` on CUDA.
10. Apply `torch.compile` in `full_frame` mode on CUDA (Linux and Windows). Disabled in `tiled` mode because hooks and `torch.compile` are incompatible with Dynamo tracing. A warm-up forward pass is run immediately after compilation to avoid a stall on the first real frame.

## Inference (`orchestrator.py`)

`run_inference()` runs the model forward pass for a single frame.

### Steps

1. Resolve refiner mode. If `resolved_refiner_mode` is provided (pre-resolved by `PipelineRunner` to avoid a per-frame VRAM probe), use it directly. Otherwise probe VRAM to decide between `full_frame` and `tiled`.

2. Install a forward hook on the refiner module if needed:
   - In `tiled` mode: the hook intercepts the refiner's forward call and replaces it with `_run_refiner_tiled()`.
   - In `full_frame` mode with `refiner_scale != 1.0`: the hook scales the refiner's output by `refiner_scale`.

3. Cast the input tensor to `model_precision` so weights and activations match.

4. Run the forward pass under `torch.inference_mode()` and `torch.autocast`. Autocast handles mixed-precision boundaries inside the model (e.g. `float32` BatchNorm layers receiving `float16` activations).

5. Remove the hook in a `finally` block so it can never fire on a subsequent call or trigger recursion.

6. Wrap the model output in `InferenceResult`.

7. Call `_free_vram_if_needed()`. On GPUs with less than 6 GB VRAM, `torch.cuda.empty_cache()` is called after each frame to keep peak usage flat. Skipped on larger GPUs to avoid the sync overhead.

## Tiled Refiner

When `refiner_mode` is `tiled`, the CNN refiner runs on overlapping 512x512 tiles instead of the full frame. This keeps peak VRAM flat and is required on low-VRAM GPUs and Apple Silicon (MPS).

The tiled pass:

1. Splits the full-resolution tensor into overlapping tiles with a configurable stride.
2. Pads edge tiles to `tile_size x tile_size` if they are smaller.
3. Runs the refiner on each tile with `state.bypass = True` to prevent the hook from firing recursively.
4. Blends tile outputs back using a linear ramp weight in the overlap regions.
5. Divides by the accumulated weight to normalise the blend.

GroupNorm does not support `bf16`/`fp16` - the tiled pass always upcasts inputs to `float32` and casts the result back to the original dtype. The refiner's weights are already `float32` (fixed in `load_model`), so only the activations need upcasting.

Output quality is identical to full-frame mode.

## Backend Protocol

The `ModelBackend` protocol allows alternative inference backends to be used as drop-in replacements. Two implementations exist:

- `TorchBackend` - wraps `load_model` and `run_inference`. The default on all platforms.
- `MLXBackend` - wraps `corridorkey-mlx` for Apple Silicon. Optional, only instantiated when the resolved backend is `"mlx"`.

Both satisfy the same protocol and return the same `InferenceResult` contract.

## VRAM and Refiner Mode Resolution

`refiner_mode = "auto"` probes VRAM once at startup using pynvml (with a `torch.cuda` fallback). The threshold is 12 GB:

- Less than 12 GB VRAM - `tiled`
- 12 GB or more VRAM - `full_frame`
- MPS (Apple Silicon) - always `tiled` (Triton/inductor does not support Metal)

The probe result is stored in `PipelineConfig.resolved_refiner_mode` so it is not repeated on every frame.

## Output Contract

```python
@dataclass
class InferenceResult:
    alpha: torch.Tensor  # [1, 1, img_size, img_size] sigmoid-activated, range 0-1, on device
    fg: torch.Tensor     # [1, 3, img_size, img_size] sigmoid-activated sRGB, range 0-1, on device
    meta: FrameMeta      # carried through from PreprocessedFrame
```

## Related

- [Preprocessor Stage](preprocessor.md) - Produces `PreprocessedFrame` consumed here.
- [Postprocessor Stage](postprocessor.md) - Consumes `InferenceResult`.
- [Configuration](../configuration.md) - `InferenceSettings` reference.
