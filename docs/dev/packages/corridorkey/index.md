# corridorkey

`corridorkey` is the Application Layer of CorridorKey. It owns the processing pipeline, clip state machine, job queue, project management, and frame I/O. It has no UI dependencies and can be consumed by any frontend - CLI, GUI, or web service.

## Installation

```shell
uv add corridorkey
```

For CUDA (Windows/Linux):

```shell
uv add "corridorkey[cuda]"
```

For MLX (Apple Silicon):

```shell
uv add "corridorkey[mlx]"
```

## Documents in This Section

- [Clip state machine](clip-state.md) - The six clip states, valid transitions, and how the pipeline routes each clip.
- [Project layout](project-layout.md) - On-disk folder structure, v1 vs v2 projects, JSON metadata files.
- [Job queue](job-queue.md) - Async GPU job management for GUI and multi-threaded consumers.
- [Configuration](configuration.md) - Config file, environment variables, precedence, and studio setup.

## Related

- [corridorkey API overview](../../api/corridorkey/index.md)
- [service reference](../../api/corridorkey/service.md)
- [pipeline reference](../../api/corridorkey/pipeline.md)
- [clip-state reference](../../api/corridorkey/clip-state.md)
- [job-queue reference](../../api/corridorkey/job-queue.md)
