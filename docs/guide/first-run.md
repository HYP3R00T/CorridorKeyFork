# First Run

After installation, run `ck init` once to complete setup. This command checks your environment, creates the config file, and downloads the inference model.

```shell
ck init
```

## What Init Does

Init runs three steps in order:

1. Environment check - the `ck init` health check prints a results table showing Python version, FFmpeg, GPU, and model status.
2. Config file - creates `~/.config/corridorkey/corridorkey.toml` with default settings if it does not already exist.
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
Then run `ck init` to verify.
```

## After Init

When init completes successfully it prints:

```text
Init complete. Run `ck` to get started.
```

Run `ck init` at any time to re-check the environment.

## Related

- [Processing clips](processing.md)
- [Troubleshooting](troubleshooting.md)
- [Commands](../corridorkey-cli/commands.md)
