# Installation

CorridorKey is installed as a command-line tool using the one-line installers below. The installer handles Python, the package manager, and the tool itself.

## System Requirements

| Requirement | Minimum |
|---|---|
| Operating system | Windows 10, macOS 12, or Ubuntu 20.04 |
| Python | 3.13 (installed automatically if missing) |
| Disk space | 2 GB (500 MB tool + ~1.4 GB model) |
| RAM | 8 GB |
| GPU | Optional but strongly recommended (see below) |

### GPU recommendations

Without a GPU, inference runs on CPU and is very slow (several minutes per frame). A GPU is strongly recommended for any practical use.

| Platform | Recommended |
|---|---|
| Windows / Linux | NVIDIA GPU with 4 GB VRAM or more (CUDA) |
| macOS Apple Silicon | M1 or later (MLX) |
| macOS Intel | CPU only |

## Windows

Open PowerShell and run:

```powershell
irm https://corridorkey.dev/install.ps1 | iex
```

The installer will ask which GPU you have, install `uv` if needed, install CorridorKey, run first-time setup, and create a `CorridorKey - Drop Clips Here.bat` shortcut on your Desktop.

## macOS and Linux

Open Terminal and run:

```shell
curl -sSf https://corridorkey.dev/install.sh | bash
```

The installer detects Apple Silicon automatically and selects the MLX build. On Linux it asks whether you have an NVIDIA GPU. After setup it creates a launcher on your Desktop.

## Manual Installation

If you prefer to install without the script, use `uv tool install` directly.

For NVIDIA GPU (Windows/Linux):

```shell
uv tool install "corridorkey-cli[cuda]" --python 3.13
```

For Apple Silicon (macOS):

```shell
uv tool install "corridorkey-cli[mlx]" --python 3.13
```

CPU only:

```shell
uv tool install corridorkey-cli --python 3.13
```

`uv` must be installed first. See the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

After manual installation, run `corridorkey init` to complete setup.

## Verifying the Installation

```shell
corridorkey --help
```

If the command is not found after installation, close and reopen your terminal to pick up the updated PATH.

## Related

- [First run](first-run.md)
- [Troubleshooting](troubleshooting.md)
