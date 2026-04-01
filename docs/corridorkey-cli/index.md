# corridorkey-cli

`corridorkey-cli` is the command-line interface for CorridorKey. It provides the `ck` entry point and all user-facing terminal commands.

This section is for anyone who wants to understand how the CLI is structured, how it calls into the `corridorkey` library, or how to extend it.

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

## Documents in This Section

- [Commands](commands.md) - All `ck` commands with flags, arguments, and behaviour.
- [Architecture](architecture.md) - How the CLI is structured and how it calls into `corridorkey`.
- [Reference](reference/index.md) - Auto-generated symbol reference.
