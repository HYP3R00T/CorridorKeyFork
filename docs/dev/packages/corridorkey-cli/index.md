# corridorkey-cli

The `corridorkey-cli` package is the command-line interface for CorridorKey. It provides the `ck` entry point and all user-facing commands.

## Purpose

This package owns the terminal UI, command definitions, and user interaction flows. It depends on `corridorkey` for all pipeline logic and never implements pipeline stages itself.

## Package Layout

```text
corridorkey_cli/
  __init__.py          # app entry point, wizard command
  _console.py          # shared Rich console instances
  _printer.py          # RichPrinter - live assembly-line progress panel
  _config_table.py     # print_config_table - resolved config display
  commands/
    __init__.py
    config.py          # ck config
    init.py            # ck init
    reset.py           # ck reset
```

## Commands

| Command | Description |
|---|---|
| `ck` (no args) | Runs the wizard - scan, configure, and process clips interactively. |
| `ck <clips_dir>` | Runs the wizard with a clips directory argument. |
| `ck init` | One-time setup: health check, config file creation, model download. |
| `ck config` | Show the resolved configuration with source attribution. |
| `ck config --write` | Write the resolved configuration to disk. |
| `ck reset` | Delete `~/.config/corridorkey` and all its contents. |

## Entry Point

The `ck` command is registered in `pyproject.toml` as:

```toml
[project.scripts]
ck = "corridorkey_cli:main"
```

## Documents in This Section

- [Commands](commands.md) - Detailed reference for each command.
