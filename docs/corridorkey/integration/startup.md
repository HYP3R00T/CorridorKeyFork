# Startup

Every interface runs a fixed startup sequence before any clip can be processed. This sequence initialises the four infrastructure concerns the package exposes: configuration, logging, device, and model.

Source: [`corridorkey/infra/`](https://github.com/hyp3r00t/CorridorKeyFork/blob/main/packages/corridorkey/src/corridorkey/infra/)

## Purpose

The startup sequence exists to fail fast. If the config file is malformed, the device is unavailable, or the model has never been downloaded, the interface should know before it presents a clip list to the user - not halfway through a render.

## How It Works

The four steps are independent and run in a fixed order. Each step produces a value that later steps or the pipeline itself depends on.

### Step 1 - Configuration

`load_config()` reads the user's TOML config file, applies any runtime overrides, and returns a validated `CorridorKeyConfig` object. If the file does not exist, built-in defaults are used. If a value fails validation (for example, an unrecognised device string), an error is raised immediately.

The config object is the single source of truth for all pipeline settings. Every stage config is derived from it via the bridge methods (`to_preprocess_config`, `to_inference_config`, and so on). The interface should never construct stage configs manually.

### Step 2 - Logging

`setup_logging(config)` attaches a timestamped file handler for the session. Log files are written to the directory specified in `config.logging.dir`, which defaults to `~/.config/corridorkey/logs`.

The package logs to the standard Python logging hierarchy. The interface is responsible for adding its own handler on top - a `StreamHandler` for a CLI, a custom handler that routes to a GUI log panel, or nothing at all if the interface does not surface logs.

### Step 3 - Device

`resolve_device(config.device)` validates the device string from the config and returns a concrete PyTorch device string. If the config says `"auto"`, the best available device is detected and returned. If the config names a specific device that is not available (for example, `"cuda"` on a machine with no NVIDIA GPU), a `DeviceError` is raised here rather than at inference time.

`detect_gpu()` is a separate call that returns a `GPUInfo` object describing the vendor, backend, version, device names, and VRAM amounts. This is purely informational - it does not affect which device is used. Interfaces should call it to populate a status panel or log the hardware configuration at startup.

### Step 4 - Model

`ensure_model()` checks whether the model checkpoint exists at the expected path. If it does, the call returns immediately. If it does not, the model is downloaded from HuggingFace, its SHA-256 checksum is verified, and the file is moved into place atomically.

The interface should pass an `on_progress` callback to receive `(bytes_downloaded, total_bytes)` notifications during the download. This is the hook for a download progress bar in a GUI or a progress indicator in a CLI. If no callback is provided, a simple text progress bar is printed to stdout.

`default_checkpoint_path()` returns the expected local path without triggering a download. Use this to check whether the model is present before deciding whether to show a download prompt.

## Key Concepts

Configuration, logging, device, and model are all initialised once per session, not once per clip. The resolved device string and the loaded model are passed into the pipeline for every clip that runs in that session.

The config object is immutable after loading. If the user changes settings mid-session, the interface must call `load_config()` again and rebuild the pipeline config from the new result.

## Related

- [Engine](runner.md) - How the startup outputs feed into the pipeline.
- [Configuration](../configuration.md) - All configuration fields and their defaults.
- [Reference - device-utils](../reference/device-utils.md) - `detect_gpu`, `resolve_device`, `GPUInfo`.
- [Reference - model-hub](../reference/model-hub.md) - `ensure_model`, `default_checkpoint_path`.
