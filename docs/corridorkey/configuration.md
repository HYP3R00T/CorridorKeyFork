# Configuration

`CorridorKeyConfig` is the validated top-level configuration model for the pipeline. It is loaded by `load_config()` and nests one settings block per stage.

## Config File

The config file is a TOML file at `~/.config/corridorkey/corridorkey.toml` by default. A project-level `corridorkey.toml` in the working directory overrides the global file.

Source priority (lowest to highest): defaults, global file, project file, runtime overrides.

## Top-Level Fields

| Field | Default | Description |
|---|---|---|
| `device` | `"auto"` | Compute device for inference. See [Device values](#device-values) below. |

### Device values

| Value | Description |
|---|---|
| `"auto"` or `None` | Auto-detect: picks ROCm > CUDA > MPS > CPU. Resolves to a single device. |
| `"cuda"` | NVIDIA or AMD GPU 0 via CUDA. |
| `"cuda:N"` | Specific GPU by index, e.g. `"cuda:1"`. |
| `"rocm"` | AMD GPU 0 via ROCm (maps to `"cuda"` internally). |
| `"rocm:N"` | Specific ROCm GPU by index. |
| `"mps"` | Apple Silicon (M1+, macOS 12.3+). |
| `"cpu"` | CPU fallback. |

`"all"` is accepted by the config model and used by the Engine to enable multi-GPU dispatch. For Layer 2 callers, pass `resolve_devices("all")` to `PipelineConfig.devices`.

## Logging Settings

| Field | Default | Description |
|---|---|---|
| `level` | `"INFO"` | Log level. |
| `dir` | `~/.config/corridorkey/logs` | Directory for log files. |

## Preprocess Settings

| Field | Default | Description |
|---|---|---|
| `img_size` | `0` | Model resolution. `0` = auto-select based on VRAM: <6 GB -> 1024, 6-12 GB -> 1536, 12+ GB -> 2048. |
| `image_upsample_mode` | `"bicubic"` | Interpolation when upscaling source frames. `"bicubic"` or `"bilinear"`. |
| `sharpen_strength` | `0.3` | Unsharp mask strength after upscaling. `0.0` disables. |
| `half_precision` | `false` | Cast tensors to float16 before inference. |
| `source_passthrough` | `true` | Carry original source pixels through to postprocessing. |

## Inference Settings

| Field | Default | Description |
|---|---|---|
| `checkpoint_path` | `~/.config/corridorkey/models/CorridorKey_v1.0.pth` | Path to the model checkpoint. |
| `use_refiner` | `true` | Enable the CNN refiner for sharp edge mattes. |
| `mixed_precision` | `true` | Run forward pass under autocast. |
| `model_precision` | `"auto"` | Weight dtype. `"auto"` selects bfloat16 on Ampere+/Apple Silicon, float16 on older GPUs, float32 on CPU. |
| `refiner_mode` | `"auto"` | `"auto"` probes VRAM (<12 GB -> tiled, 12+ GB -> full_frame). `"full_frame"` or `"tiled"` to override. |
| `refiner_scale` | `1.0` | Blend factor for refiner corrections. `0.0` disables, `1.0` is full. |

## Postprocess Settings

| Field | Default | Description |
|---|---|---|
| `fg_upsample_mode` | `"lanczos4"` | Interpolation for upscaling foreground. |
| `alpha_upsample_mode` | `"lanczos4"` | Interpolation for upscaling alpha matte. |
| `despill_strength` | `0.5` | Green spill suppression. `0.0` = off, `1.0` = full. |
| `auto_despeckle` | `true` | Remove small disconnected alpha islands. |
| `despeckle_size` | `400` | Minimum region area in pixels to keep. |
| `despeckle_dilation` | `25` | Dilation radius after component removal. |
| `despeckle_blur` | `5` | Gaussian blur radius after dilation. |
| `source_passthrough` | `true` | Replace model FG in opaque regions with original source pixels. |
| `edge_erode_px` | `3` | Erosion radius for the interior mask used by source_passthrough. |
| `edge_blur_px` | `7` | Blur radius for the source_passthrough blend seam. `0` disables. |
| `hint_sharpen` | `true` | Apply binary alpha hint mask to eliminate soft edge tails. |
| `hint_sharpen_dilation` | `3` | Dilation radius applied to the binarised hint before masking. |
| `debug_dump` | `false` | Save intermediate debug images to a `debug/` subfolder. |

## Writer Settings

| Field | Default | Description |
|---|---|---|
| `alpha_enabled` | `true` | Write the alpha matte. |
| `alpha_format` | `"png"` | Format for alpha output. `"png"` or `"exr"`. |
| `fg_enabled` | `true` | Write the straight sRGB foreground. |
| `fg_format` | `"png"` | Format for foreground output. |
| `processed_enabled` | `true` | Write the premultiplied RGBA output. |
| `processed_format` | `"png"` | Format for processed RGBA output. |
| `comp_enabled` | `true` | Write the checkerboard preview composite. |
| `exr_compression` | `"dwaa"` | EXR compression codec. `"dwaa"`, `"zip"`, `"none"`, and others. |

## Related

- [Reference - configuration](reference/configuration.md) - Full symbol reference.
- [Knowledge - Configuration](../knowledge/configuration/index.md) - User-facing configuration guide.
