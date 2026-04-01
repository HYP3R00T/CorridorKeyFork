# Architecture

`corridorkey-cli` is a thin interface layer built on top of the `corridorkey` library. It owns the terminal UI and user interaction flows. It does not implement any pipeline logic.

Source: [`corridorkey_cli/`](https://github.com/hyp3r00t/CorridorKey/blob/main/packages/corridorkey-cli/src/corridorkey_cli/)

## Dependency Direction

The CLI imports from `corridorkey`. The `corridorkey` package never imports from the CLI. This is the same boundary that applies to all interfaces built on the library.

## Entry Point

The `ck` command is registered in `pyproject.toml`:

```toml
[project.scripts]
ck = "corridorkey_cli:main"
```

`main()` in `corridorkey_cli.__init__` creates the Typer app and dispatches to the appropriate command.

## Key Modules

`_printer.py` contains `RichPrinter`, which implements the live assembly-line progress panel shown during inference. It subscribes to `PipelineEvents` callbacks and updates a Rich Live display as frames move through the pipeline.

`_config_table.py` contains `print_config_table`, which renders the resolved configuration as a Rich table with source attribution. It calls `load_config_with_metadata()` to get both the config values and the source of each value (defaults, global file, project file, or environment variable).

`_console.py` holds the shared Rich console instances used across all commands.

## How the Wizard Works

The wizard (`ck` with no subcommand) is the primary command. It follows the integration pattern described in the `corridorkey` integration guide:

1. Calls `load_config()` and `setup_logging()`.
2. Calls `ensure_model()` with a progress callback that updates the terminal.
3. Calls `resolve_device()` and `detect_gpu()`, then displays the result.
4. Calls `scan()` with a `PipelineEvents` instance so clips appear in the terminal as they are discovered.
5. Presents the clip table and action menu.
6. For each clip to process, calls `load()`, then `PipelineRunner.run()` with a `RichPrinter` attached to the events.

## Related

- [Commands](commands.md) - All `ck` commands with flags and behaviour.
- [corridorkey Integration Guide](../corridorkey/integration/index.md) - The integration patterns the CLI uses.
- [Reference](reference/index.md) - Auto-generated symbol reference.
