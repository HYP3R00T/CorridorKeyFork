# Config Commands

The `config` subcommand group inspects and writes the CorridorKey configuration file. Both subcommands operate on the resolved configuration, meaning all sources (defaults, global file, project file, environment variables) are merged before any output is produced.

## corridorkey config show

`config show` prints the fully resolved configuration as a table.

```shell
corridorkey config show
```

The table lists every field name and its resolved value. A note below the table shows the resolution order from lowest to highest priority:

1. Model field defaults
2. `~/.config/corridorkey/corridorkey.yaml` (global user config)
3. `./corridorkey.yaml` in the current working directory (project config)
4. `CORRIDORKEY_*` environment variables

Use `config show` to confirm that environment variables or a project-level config file are being picked up correctly.

## corridorkey config init

`config init` writes the resolved configuration to `~/.config/corridorkey/corridorkey.yaml`.

```shell
corridorkey config init
```

If the file already exists it is overwritten with the current resolved values. Fields that are `None` are omitted from the written file.

This command is equivalent to the config-file step inside `corridorkey init`. Use it to reset the global config file to current defaults without re-running the full init flow.

## Related

- [config reference](../../api/corridorkey-cli/config.md)
- [helpers reference](../../api/corridorkey-cli/helpers.md)
- [Configuration](../corridorkey/configuration.md)
- [Setup commands](setup-commands.md)
