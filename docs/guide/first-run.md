# First Run

After installation, run `corridorkey init` once to complete setup. This command checks your environment, creates the config file, and downloads the inference model.

```shell
corridorkey init
```

## What Init Does

Init runs three steps in order:

1. Environment check - runs `corridorkey doctor` and prints a results table showing Python version, FFmpeg, GPU, and model status.
2. Config file - creates `~/.config/corridorkey/corridorkey.yaml` with default settings if it does not already exist.
3. Model download - checks whether the inference model is present and offers to download it if not.

## The Environment Check Table

The table shows a row for each check with a status of OK, WARN, or FAIL.

FAIL on any of these three means CorridorKey will not work until fixed:

- Python >= 3.13 - the installed Python version is too old
- ffmpeg - FFmpeg is not installed or not on PATH
- inference model - the model file has not been downloaded yet

WARN rows (git, config file, checkpoint directory) do not block processing. Init will continue past them.

## Downloading the Model

When init reaches the model step it shows the download URL and asks for confirmation:

```text
Inference model not found in: ~/.config/corridorkey/models
URL: https://huggingface.co/...
Download inference model now? [y/n] (y):
```

The model is approximately 400 MB. A progress bar shows download progress. The file is saved to `~/.config/corridorkey/models/` and verified against a checksum.

If you decline the download, init prints the manual instructions:

```text
To download manually, place greenformer_v2.pth in:
  ~/.config/corridorkey/models
Then run corridorkey doctor to verify.
```

## After Init

When init completes successfully it prints:

```text
Init complete. Run `corridorkey wizard` to get started.
```

Run `corridorkey doctor` at any time to re-check the environment without modifying anything.

## Related

- [Processing clips](processing.md)
- [Troubleshooting](troubleshooting.md)
- [Setup commands](../dev/packages/corridorkey-cli/setup-commands.md)
