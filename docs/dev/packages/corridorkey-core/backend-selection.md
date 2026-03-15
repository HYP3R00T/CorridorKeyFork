# Backend Selection

`corridorkey-core` supports two inference backends: Torch and MLX. The backend is resolved once at engine creation time and is transparent to callers - `process_frame` behaves identically regardless of which backend is active.

## Resolution Order

The backend is resolved in this priority order:

1. Explicit `backend` argument passed to `create_engine`
2. `CORRIDORKEY_BACKEND` environment variable
3. Auto-detect

## Auto-detect Logic

Auto-detect selects MLX when all three conditions are true:

1. The platform is `darwin` (macOS)
2. The machine architecture is `arm64` (Apple Silicon)
3. The `corridorkey-mlx` package is importable

Otherwise Torch is used.

## Checkpoint Layout

Each backend expects a different checkpoint file format in the same directory:

| Backend | Extension | Example |
|---|---|---|
| Torch | `.pth` | `greenformer_v2.pth` |
| MLX | `.safetensors` | `greenformer_v2.safetensors` |

`create_engine` scans `checkpoint_dir` for exactly one file with the matching extension. It raises `FileNotFoundError` if none are found and `ValueError` if more than one are found. If you have the wrong extension for your backend, the error message will suggest the correct `--backend` flag.

Place only one checkpoint file per backend in the directory:

```text
checkpoints/
    greenformer_v2.pth           # used by Torch
    greenformer_v2.safetensors   # used by MLX
```

## MLX Adapter

The MLX backend (`corridorkey-mlx`) returns uint8 arrays and does not apply despill or despeckle natively. `corridorkey-core` wraps the MLX engine in an adapter that:

1. Converts float32 inputs to uint8 before calling the MLX engine
2. Applies despill and despeckle in Python after receiving the uint8 output
3. Converts uint8 outputs back to float32

The adapter is transparent. Callers always receive the same float32 output contract.

## Device Selection (Torch Only)

The `device` argument to `create_engine` is passed directly to PyTorch. It has no effect on the MLX backend. See the [inference-engine reference](../../api/corridorkey-core/inference-engine.md) for accepted values.

## Image Size

`img_size` controls the square resolution the model runs at internally. Inputs are resized to this resolution before inference and outputs are resized back to the original frame size. The default is `2048`. Reducing it speeds up inference at the cost of detail in fine edges.

## Related

- [create_engine reference](../../api/corridorkey-core/index.md)
- [Output contract](output-contract.md)
