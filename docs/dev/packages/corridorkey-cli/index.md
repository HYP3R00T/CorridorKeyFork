# corridorkey-cli

`corridorkey-cli` is the UI Layer of CorridorKey. It provides the `corridorkey` command-line tool that wraps the Application Layer (`corridorkey`) for interactive and scripted use. It has no inference or pipeline logic of its own.

## Installation

```shell
uv add corridorkey-cli
```

For CUDA (Windows/Linux):

```shell
uv add "corridorkey-cli[cuda]"
```

For MLX (Apple Silicon):

```shell
uv add "corridorkey-cli[mlx]"
```

## Documents in This Section

- [Setup commands](setup-commands.md) - `init` and `doctor`: first-run setup and environment health checks.
- [Processing commands](processing-commands.md) - `wizard`, `process`, and `scan`: running the keying pipeline.
- [Config commands](config-commands.md) - `config show` and `config init`: inspecting and writing configuration.

## Related

- [corridorkey-cli API overview](../../api/corridorkey-cli/index.md)
- [corridorkey package](../corridorkey/index.md)
- [Configuration](../corridorkey/configuration.md)
- [Clip state machine](../corridorkey/clip-state.md)
