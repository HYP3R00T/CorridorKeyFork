# corridorkey-cli

Command-line interface for CorridorKey. Provides the `ck` command.

## Installation

```bash
pip install corridorkey-cli
```

## Commands

### `ck [clips_dir]` — wizard (default)

Scan a clips directory, prompt for engine settings, and process all clips.

```bash
ck /path/to/clips          # interactive
ck /path/to/clips --yes    # non-interactive, use config defaults
```

Engine settings are selected via preset or manual entry:

| Preset | refiner_mode | precision | img_size |
|---|---|---|---|
| `full_frame` | full_frame | float16 | 1024 |
| `balanced` | auto | auto | 1536 |
| `quality` | full_frame | bfloat16 | 2048 |
| `max_quality` | full_frame | float32 | 2048 |
| `tiled` | tiled | float16 | 1024 |

### `ck init`

One-time setup: runs an environment health check, creates the config file, and offers to download the inference model.

```bash
ck init
```

Checks:

- Python >= 3.13
- Compute device and VRAM
- Config file presence
- Inference model presence

### `ck config`

Show the resolved configuration with source attribution (default / file / env).

```bash
ck config           # display only
ck config --write   # write to ~/.config/corridorkey/corridorkey.yaml
```

Config is resolved from (lowest → highest priority):

1. Built-in defaults
2. `~/.config/corridorkey/corridorkey.yaml`
3. `./corridorkey.yaml`
4. `CK_*` environment variables

### `ck reset`

Delete `~/.config/corridorkey` — removes config, downloaded models, and logs.

```bash
ck reset        # prompts for confirmation
ck reset --yes  # skip confirmation
```

Run `ck init` afterwards to set up again.

## Clips Directory Layout

```sh
clips/
  ClipName/
    Input/        # source video or image sequence
    AlphaHint/    # (optional) pre-generated alpha matte
```

Outputs are written to `ClipName/Output/`.
