# Setup Commands

The setup commands prepare the environment for first use. Run them once on a new machine before processing any clips.

## corridorkey init

`corridorkey init` is the recommended starting point. It runs `doctor` internally, creates the config file if it is missing, and offers to download the inference model.

```shell
corridorkey init
```

Steps performed in order:

1. Runs `corridorkey doctor` and prints the results table.
2. Checks for `~/.config/corridorkey/corridorkey.yaml`. Creates it with defaults if absent.
3. Checks whether the inference model is present in `checkpoint_dir`. Offers to download it if not.

`init` does not fail if `doctor` reports warnings. It continues through all steps so the user sees the full picture before deciding what to fix.

## corridorkey doctor

`corridorkey doctor` runs a read-only health check and prints a results table. It does not modify any files.

```shell
corridorkey doctor
```

Checks performed:

| Check | Pass condition |
|---|---|
| Python >= 3.13 | `sys.version_info >= (3, 13)` |
| git | Found on `PATH` |
| ffmpeg | Found on `PATH` and reports a version |
| ffprobe | Found on `PATH` |
| compute device | `detect_best_device()` returns without error |
| VRAM | CUDA device reports free memory (CUDA only) |
| config file | `~/.config/corridorkey/corridorkey.yaml` exists |
| checkpoint_dir | Directory exists |
| inference model | `.pth` or `.safetensors` file found in `checkpoint_dir` |
| platform | Always passes; shows OS and architecture |

`doctor` exits with code 1 if any check marked FAIL is present. Checks marked WARN do not affect the exit code.

The three checks that produce FAIL are: Python version, ffmpeg, and inference model. All three must pass before `corridorkey wizard` or `corridorkey process` will work.

## Related

- [init reference](../../api/corridorkey-cli/init.md)
- [doctor reference](../../api/corridorkey-cli/doctor.md)
- [Configuration](../corridorkey/configuration.md)
- [Processing commands](processing-commands.md)
