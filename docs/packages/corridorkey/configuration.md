# Configuration

`CorridorKeyConfig` is a Pydantic model that centralises all runtime settings. It is loaded once at startup and passed to `CorridorKeyService`. All tool-managed files (models, logs, cache) live under `app_dir`, which defaults to `~/.config/corridorkey`.

## Config Fields

| Field | Default | Description |
|---|---|---|
| `app_dir` | `~/.config/corridorkey` | Root directory for all tool-managed files |
| `checkpoint_dir` | `~/.config/corridorkey/models` | Directory where model checkpoints are stored |
| `device` | `"auto"` | Compute device: `"auto"`, `"cuda"`, `"mps"`, or `"cpu"` |
| `despill_strength` | `1.0` | Default green-spill suppression strength (0.0-1.0) |
| `auto_despeckle` | `True` | Enable automatic matte despeckling by default |
| `despeckle_size` | `400` | Maximum speckle area in pixels to remove |
| `refiner_scale` | `1.0` | Scale factor for the optional refiner stage |
| `input_is_linear` | `False` | Treat input frames as linear light (e.g. EXR) by default |
| `fg_format` | `"exr"` | Default foreground output format |
| `matte_format` | `"exr"` | Default matte output format |
| `comp_format` | `"png"` | Default composite preview format |
| `processed_format` | `"png"` | Default processed input format |

## Resolution Order

Settings are resolved from lowest to highest priority:

1. Model field defaults (the table above)
2. `~/.config/corridorkey/corridorkey.toml` (global user config)
3. `./corridorkey.toml` in the current working directory (project config)
4. Environment variables prefixed with `CORRIDORKEY_`
5. `overrides` dict passed to `load_config()`

## Config File

Create `~/.config/corridorkey/corridorkey.toml` to set persistent defaults:

```toml
checkpoint_dir = "~/studio/shared/corridorkey/models"
device = "cuda"
despill_strength = 0.85
fg_format = "exr"
matte_format = "exr"
comp_format = "png"
```

A project-level `corridorkey.toml` in the current working directory overrides the global file. This is useful for per-show settings.

## Environment Variables

Every field can be overridden with a `CORRIDORKEY_` prefixed environment variable:

```shell
CORRIDORKEY_DEVICE=cuda
CORRIDORKEY_CHECKPOINT_DIR=/mnt/studio/models
CORRIDORKEY_DESPILL_STRENGTH=0.9
```

## Studio Setup

In a shared studio environment, point `checkpoint_dir` to a network path so all workstations share the same model files:

```toml
# ~/.config/corridorkey/corridorkey.toml on each workstation
checkpoint_dir = "/mnt/studio/corridorkey/models"
device = "cuda"
```

The `app_dir` can stay at the default per-user location since it holds user-specific logs and cache.

## Related

- [service reference](../../api/corridorkey/service.md)
- [corridorkey API overview](../../api/corridorkey/index.md)
