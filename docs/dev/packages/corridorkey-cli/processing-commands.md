# Processing Commands

The processing commands run the keying pipeline against a directory of clips. All three read clip state from disk and only process clips that are in the `READY` state. See [Clip state machine](../corridorkey/clip-state.md) for how clips reach `READY`.

## corridorkey wizard

`wizard` is the interactive entry point. It scans a directory, shows a state table, and loops until the user quits.

```shell
corridorkey wizard [CLIPS_DIR]
```

If `CLIPS_DIR` is omitted, the wizard prompts for it interactively.

The main loop:

1. Scans the directory and prints a clip state table.
2. Presents an action menu: run inference, re-scan, or quit.
3. If inference is selected, shows the current inference settings (from config) and asks to confirm or override each parameter.
4. Loads the model (with a spinner on first load) and processes all `READY` clips with a progress bar.
5. Returns to the main loop after processing.

Inference settings available in the wizard:

| Setting | Description |
|---|---|
| `input_is_linear` | Whether input frames are linear light (e.g. EXR) |
| `despill_strength` | Green spill removal strength (0.0-1.0) |
| `auto_despeckle` | Remove small matte artifacts automatically |
| `despeckle_size` | Minimum artifact area in pixels to remove |
| `refiner_scale` | Edge refiner scale factor |

The wizard pre-loads the inference engine before the first clip so the model compilation spinner appears once rather than stalling silently on the first frame.

## corridorkey process

`process` is the non-interactive batch command. It processes all `READY` clips in a directory and exits.

```shell
corridorkey process CLIPS_DIR [OPTIONS]
```

All inference and output settings are passed as flags. Clips in `RAW` or `MASKED` state are skipped with a note in the results table.

Key options:

| Option | Default | Description |
|---|---|---|
| `--device` | `auto` | Compute device: `auto`, `cuda`, `mps`, `cpu` |
| `--despill` | `1.0` | Green spill removal strength (0.0-1.0) |
| `--despeckle / --no-despeckle` | on | Remove small matte artifacts |
| `--despeckle-size` | `400` | Minimum artifact area in pixels |
| `--refiner` | `1.0` | Edge refiner scale (0.0 disables) |
| `--linear` | off | Treat input as linear light |
| `--fg-format` | `exr` | Foreground output format |
| `--matte-format` | `exr` | Matte output format |
| `--comp-format` | `png` | Composite preview format |
| `--no-comp` | off | Skip composite output |
| `--no-processed` | off | Skip processed RGBA output |
| `--verbose` / `-v` | off | Enable debug logging |

`process` exits with code 1 if any clip fails. The results table is always printed, including skipped clips.

## corridorkey scan

`scan` prints a clip state table without processing anything. Use it to inspect a directory before running inference.

```shell
corridorkey scan CLIPS_DIR
```

The output table shows each clip's name, state, input frame count, alpha frame count, and any error message. A summary line below the table counts clips per state.

`scan` is read-only. It does not modify any files.

## Related

- [wizard reference](../../api/corridorkey-cli/wizard.md)
- [process reference](../../api/corridorkey-cli/process.md)
- [scan reference](../../api/corridorkey-cli/scan.md)
- [Clip state machine](../corridorkey/clip-state.md)
- [Configuration](../corridorkey/configuration.md)
- [Setup commands](setup-commands.md)
