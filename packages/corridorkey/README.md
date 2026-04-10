# corridorkey

Core pipeline library for CorridorKey — AI green screen keying.

## Overview

This package implements the full inference pipeline as a set of composable stages:

- **scan** — discover clips from a directory
- **load** — validate a clip and extract video frames
- **preprocess_frame** — resize, normalise, and tensorise a frame
- **run_inference** — run the neural network (via `backend.run()`)
- **postprocess_frame** — despill, despeckle, upsample, composite
- **write_frame** — write alpha, foreground, and composite outputs

## Installation

```bash
# CUDA
pip install "corridorkey[cuda]"

# Apple Silicon (MLX)
pip install "corridorkey[mlx]"

# ROCm (Linux)
pip install "corridorkey[rocm]"
```

## Usage

```python
from corridorkey import (
    load_config, setup_logging, resolve_device,
    scan, load, resolve_alpha,
    preprocess_frame, postprocess_frame, write_frame,
    InferenceConfig, load_backend, list_clip_frames,
)

config = load_config()
setup_logging(config)
device = resolve_device(config.device)

# Discover clips
result = scan("/path/to/clips")

for clip in result.clips:
    manifest = load(clip)

    # If no alpha hint exists, generate one externally then:
    if manifest.needs_alpha:
        manifest = resolve_alpha(manifest, "/path/to/alpha_frames")

    # Build stage configs
    inference_config = config.to_inference_config(device=device)
    preprocess_config = config.to_preprocess_config(
        device=device, resolved_img_size=inference_config.img_size
    )
    postprocess_config = config.to_postprocess_config()
    write_config = config.to_writer_config(manifest.output_dir)

    backend = load_backend(inference_config)

    imgs = list_clip_frames(manifest.frames_dir)
    alps = list_clip_frames(manifest.alpha_frames_dir)

    for i in range(*manifest.frame_range):
        preprocessed = preprocess_frame(manifest, i, preprocess_config,
                                        image_files=imgs, alpha_files=alps)
        result = backend.run(preprocessed)
        postprocessed = postprocess_frame(result, postprocess_config)
        write_frame(postprocessed, write_config)
```

For a higher-level interface, use `Runner`:

```python
from corridorkey import Runner

pipeline_config = config.to_pipeline_config(device=device, model=model)
Runner(manifest, pipeline_config).run()
```

## Pipeline Stages

### scan

```python
clips = scan(path)  # -> ScanResult
# clips.clips: list[Clip]
# clips.clip_count: int
```

Accepts a clips root directory, a single clip folder, or a single video file.

### load

```python
manifest = load(clip)  # -> ClipManifest
# manifest.needs_alpha: bool — True if no alpha hint frames found
# manifest.frames_dir: Path
# manifest.alpha_frames_dir: Path
# manifest.output_dir: Path
# manifest.frame_count: int
# manifest.frame_range: tuple[int, int]
```

### resolve_alpha

```python
manifest = resolve_alpha(manifest, alpha_frames_dir)
```

Call after external alpha generation. Validates frame count matches and sets `needs_alpha=False`.

### preprocess_frame

```python
preprocessed = preprocess_frame(manifest, i, config, image_files=imgs, alpha_files=alps)
# preprocessed.tensor: torch.Tensor [1, 4, img_size, img_size]
# preprocessed.meta: FrameMeta
```

### run_inference

```python
result = backend.run(preprocessed)
# result.alpha: torch.Tensor [1, 1, img_size, img_size]
# result.fg: torch.Tensor [1, 3, img_size, img_size]
```

### postprocess_frame

```python
postprocessed = postprocess_frame(result, config)
# postprocessed.alpha: np.ndarray [H, W, 1] float32
# postprocessed.fg: np.ndarray [H, W, 3] float32
```

### write_frame

```python
write_frame(postprocessed, config)
```

Writes alpha, foreground, and/or composite frames to `config.output_dir`.

## Configuration

`CorridorKeyConfig` is a Pydantic model with one settings block per stage.

```python
from corridorkey import load_config
config = load_config()  # loads from file + env vars
```

Config is resolved from (lowest → highest priority):

1. Built-in defaults
2. `~/.config/corridorkey/corridorkey.yaml`
3. `./corridorkey.yaml`
4. `CK_*` environment variables

## Device Support

| Value | Description |
|---|---|
| `auto` | Best available: ROCm > CUDA > MPS > CPU |
| `cuda` / `cuda:N` | NVIDIA GPU (specific index optional) |
| `rocm` / `rocm:N` | AMD GPU (Linux) |
| `mps` | Apple Silicon |
| `cpu` | CPU fallback |
| `all` | All CUDA GPUs in parallel |
