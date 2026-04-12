# Architecture

`corridorkey-cli` is a thin interface layer built on top of the `corridorkey` library. It owns the terminal UI and user interaction flows. It does not implement any pipeline logic.

Source: [`corridorkey_cli/`](https://github.com/hyp3r00t/CorridorKeyFork/blob/main/packages/corridorkey-cli/src/corridorkey_cli/)

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

`_config_table.py` contains `print_config_table`, which renders the resolved configuration as a Rich table with source attribution. It calls `load_config_with_metadata()` to get both the config values and the source of each value (defaults, global file, project file, or runtime override).

`_console.py` holds the shared Rich console instances used across all commands.

## How the Wizard Works

The wizard (`ck` with no subcommand) is the primary command. It now follows a much smaller flow:

1. Ensures the TOML config file exists and loads the resolved config with metadata.
2. Prompts for the clips directory if one was not passed on the command line.
3. Lets the user accept the current engine settings or choose a preset/manual values.
4. Builds a fresh `CorridorKeyConfig` with the chosen inference and preprocessing values.
5. Registers lightweight Engine event handlers for `clip_found`, `clip_skipped`, `clip_loading`, and `clip_complete`.
6. Calls `engine.run([clips_dir])`.

`_printer.py` still exists for interfaces that want a live assembly-line panel, but the default wizard no longer wires it in.

## Related

- [Commands](commands.md) - All `ck` commands with flags and behaviour.
- [corridorkey Integration Guide](../corridorkey/integration/index.md) - The integration patterns the CLI uses.
- [Reference](reference/index.md) - Auto-generated symbol reference.
