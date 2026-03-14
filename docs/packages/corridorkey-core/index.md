# corridorkey-core

`corridorkey-core` is the Core Layer of CorridorKey. It loads the GreenFormer model, runs per-frame inference, and returns composited output. It has no filesystem, pipeline, or UI dependencies and can be embedded in any workflow.

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

## Documents in This Section

- [Output contract](output-contract.md) - The four keys returned by `process_frame` and how to use them.
- [Backend selection](backend-selection.md) - Torch vs MLX, auto-detect, environment variable, checkpoint layout.
- [Compositing utilities](compositing-utilities.md) - Pure math functions for color space conversion, despill, and matte cleanup.

## Related

- [corridorkey-core API overview](../../api/corridorkey-core/index.md)
- [inference-engine reference](../../api/corridorkey-core/inference-engine.md)
- [compositing reference](../../api/corridorkey-core/compositing.md)
- [model-transformer reference](../../api/corridorkey-core/model-transformer.md)
