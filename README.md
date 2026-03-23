# CorridorKey

AI-powered green screen keyer by Corridor Digital. Removes green screens from video clips using a neural network inference pipeline.

## Requirements

- Python 3.13
- CUDA GPU (recommended), Apple Silicon (MPS), ROCm, or CPU
- FFmpeg (for video extraction)

## Installation

```bash
# CUDA
uv sync --extra cuda

# Apple Silicon
uv sync --extra mlx

# ROCm (Linux only)
uv sync --extra rocm
```

## Quick Start

```bash
# First-time setup: health check, config, model download
ck init

# Process a clips directory (interactive wizard)
ck /path/to/clips

# Non-interactive — use config defaults
ck /path/to/clips --yes
```

## Clips Directory Layout

CorridorKey expects clips in this structure:

```sh
clips/
  ClipName/
    Input/          # source video or image sequence
    AlphaHint/      # (optional) pre-generated alpha matte
```

Outputs are written to `ClipName/Output/` alongside the input.

## CLI Commands

| Command | Description |
|---|---|
| `ck [clips_dir]` | Scan, configure, and process clips (default wizard) |
| `ck init` | One-time setup: health check, config file, model download |
| `ck config` | Show resolved configuration |
| `ck config --write` | Write config to disk |
| `ck reset` | Delete `~/.config/corridorkey` (config, models, logs) |

## Configuration

Config is loaded from (lowest → highest priority):

1. Built-in defaults
2. `~/.config/corridorkey/corridorkey.yaml`
3. `./corridorkey.yaml` (project-local)
4. `CK_*` environment variables

Example `corridorkey.yaml`:

```yaml
device: auto  # auto / cuda / cuda:N / rocm / mps / cpu / all

logging:
  level: INFO
  dir: ~/.config/corridorkey/logs

preprocess:
  img_size: 0        # 0=auto, or 512 / 1024 / 1536 / 2048
  half_precision: false

inference:
  refiner_mode: auto  # auto / full_frame / tiled
  model_precision: auto  # auto / float16 / bfloat16 / float32
  use_refiner: true
  mixed_precision: true

postprocess:
  despill_strength: 1.0
  auto_despeckle: false
  hint_sharpen: false

writer:
  alpha_format: exr   # exr / png
  fg_format: exr
  processed_format: exr
  comp_enabled: false
```

## Engine Presets

When running the wizard interactively, you can pick a preset:

| Preset | refiner_mode | precision | img_size |
|---|---|---|---|
| `full_frame` | full_frame | float16 | 1024 |
| `balanced` | auto | auto | 1536 |
| `quality` | full_frame | bfloat16 | 2048 |
| `max_quality` | full_frame | float32 | 2048 |
| `tiled` | tiled | float16 | 1024 |

## Packages

| Package | Description |
|---|---|
| `packages/corridorkey` | Core pipeline library |
| `packages/corridorkey-cli` | `ck` command-line interface |

## Development

```bash
# Install dev dependencies
uv sync

# Run tests
uv run pytest

# Type check
uv run ty check

# Lint
uv run ruff check
```

## License

CC-BY-NC-SA-4.0 — Corridor Digital
