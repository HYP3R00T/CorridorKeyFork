# Commands

Reference for all `ck` commands. Each command is implemented in `corridorkey_cli.commands` or the package `__init__`.

## ck (wizard)

The default command. Runs when no subcommand is given or when a clips directory is passed directly.

```shell
ck
ck <clips_dir>
ck <clips_dir> --yes
```

Scans for clips, shows the resolved configuration, prompts for confirmation, then runs the full pipeline. Pass `--yes` to skip prompts and use config defaults.

Implemented in `corridorkey_cli.__init__`.

## ck init

One-time environment setup.

```shell
ck init
```

Runs three steps in order:

1. Health check - verifies Python version, compute device, config file, and inference model.
2. Config file - creates `~/.config/corridorkey/corridorkey.yaml` if it does not exist.
3. Model download - offers to download the inference model if not found.

Implemented in `corridorkey_cli.commands.init`.

## ck config

Show the resolved configuration with source attribution.

```shell
ck config
ck config --write
```

Displays a table of all configuration fields, their resolved values, and the source each value came from (defaults, global file, project file, or environment variable). Pass `--write` to save the resolved configuration to disk.

Implemented in `corridorkey_cli.commands.config`.

## ck reset

Delete the CorridorKey config directory.

```shell
ck reset
ck reset --yes
```

Removes `~/.config/corridorkey` and all its contents, including the config file, downloaded models, and logs. Prompts for confirmation unless `--yes` is passed. Run `ck init` afterwards to set up again.

Implemented in `corridorkey_cli.commands.reset`.

## Related

- [Reference](reference/index.md) - Auto-generated symbol reference.
