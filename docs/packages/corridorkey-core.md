# corridorkey-core

`corridorkey-core` is the Core Layer of CorridorKey. It loads the GreenFormer model, runs inference, and returns composited output. It has no filesystem, pipeline, or UI dependencies and can be embedded in any workflow.

## Installation

```shell
uv add corridorkey-core
```

For CUDA (Windows/Linux):

```shell
uv add "corridorkey-core[cuda]"
```

For MLX (Apple Silicon):

```shell
uv add "corridorkey-core[mlx]"
```

## How It Works

`create_engine` resolves the backend, loads the checkpoint, and returns an engine. The engine exposes a single method, `process_frame`, which accepts a raw image and a green screen mask and returns a dict with four outputs.

Backend resolution order:

1. Explicit `backend` argument
2. `CORRIDORKEY_BACKEND` environment variable
3. Auto-detect: MLX if running on Apple Silicon (`darwin/arm64`) and `corridorkey-mlx` is installed, otherwise Torch

## Quick Start

```python
from corridorkey_core import create_engine
import numpy as np

engine = create_engine("/path/to/checkpoints")

# image: float32 [H, W, 3] sRGB, range 0.0-1.0
# mask:  float32 [H, W] or [H, W, 1], range 0.0-1.0
image = np.zeros((1080, 1920, 3), dtype=np.float32)
mask = np.zeros((1080, 1920), dtype=np.float32)

result = engine.process_frame(image, mask)
```

## Output

`process_frame` always returns the same dict regardless of backend:

| Key | Shape | Range | Description |
|---|---|---|---|
| `alpha` | `[H, W, 1]` | 0.0-1.0 | Raw predicted alpha matte |
| `fg` | `[H, W, 3]` | 0.0-1.0 | Raw foreground, sRGB straight |
| `comp` | `[H, W, 3]` | 0.0-1.0 | Preview composite over checkerboard, sRGB |
| `processed` | `[H, W, 4]` | float | Final RGBA, linear premultiplied |

## Backend Selection

```python
# Force Torch
engine = create_engine("/path/to/checkpoints", backend="torch")

# Force MLX (Apple Silicon only)
engine = create_engine("/path/to/checkpoints", backend="mlx")

# Via environment variable
# CORRIDORKEY_BACKEND=mlx uv run your_script.py
```

## Related

- [corridorkey-core API Overview](../api/corridorkey-core/index.md)
- [create_engine reference](../api/corridorkey-core/index.md)
- [inference-engine reference](../api/corridorkey-core/inference-engine.md)
