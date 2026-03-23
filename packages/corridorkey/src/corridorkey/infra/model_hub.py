"""Model hub — download and locate the inference checkpoint.

Constants:
    MODEL_URL       — HuggingFace download URL for CorridorKey_v1.0.pth
    MODEL_FILENAME  — expected filename on disk
    MODEL_CHECKSUM  — SHA-256 of the file (empty string disables verification)
    DEFAULT_MODEL_DIR — ~/.config/corridorkey/models

Public entry points:
    default_checkpoint_path() -> Path   — returns the expected local path
    ensure_model(dest_dir, on_progress) -> Path  — download if missing, verify, return path
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

from corridorkey.errors import ModelError

logger = logging.getLogger(__name__)

MODEL_URL = "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"
MODEL_FILENAME = "CorridorKey_v1.0.pth"
MODEL_CHECKSUM = "a03827f58e8c79b2ca26031bf67c77db5390dc1718c1ffc5b7aed8b57315788f"
DEFAULT_MODEL_DIR = Path("~/.config/corridorkey/models")


def default_checkpoint_path() -> Path:
    """Return the expected path for the default model checkpoint."""
    return DEFAULT_MODEL_DIR.expanduser() / MODEL_FILENAME


def ensure_model(
    dest_dir: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Return the checkpoint path, downloading it first if it doesn't exist.

    Args:
        dest_dir: Directory to store the model. Defaults to DEFAULT_MODEL_DIR.
        on_progress: Optional callback(bytes_downloaded, total_bytes).

    Returns:
        Absolute path to the verified checkpoint file.

    Raises:
        ModelError: If the download fails or the checksum doesn't match.
    """
    directory = (dest_dir or DEFAULT_MODEL_DIR).expanduser()
    dest = directory / MODEL_FILENAME

    if dest.is_file():
        logger.debug("model_hub: checkpoint already present at %s", dest)
        return dest

    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / (MODEL_FILENAME + ".tmp")

    logger.info("model_hub: downloading model from %s", MODEL_URL)
    print(f"Downloading model to {dest} ...")

    try:
        with urllib.request.urlopen(MODEL_URL) as response:  # noqa: S310
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 64 * 1024

            with open(tmp, "wb") as f:
                while True:
                    data = response.read(chunk)
                    if not data:
                        break
                    f.write(data)
                    downloaded += len(data)
                    if on_progress:
                        on_progress(downloaded, total)
                    else:
                        _print_progress(downloaded, total)

    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise ModelError(f"Model download failed: {e}") from e

    print()  # newline after progress bar

    if MODEL_CHECKSUM:
        logger.info("model_hub: verifying checksum")
        actual = _sha256(tmp)
        if actual != MODEL_CHECKSUM.lower():
            tmp.unlink(missing_ok=True)
            raise ModelError(
                f"Checksum mismatch for {MODEL_FILENAME}.\n"
                f"  Expected: {MODEL_CHECKSUM}\n"
                f"  Got:      {actual}\n"
                "The downloaded file has been removed. Try again."
            )
        logger.info("model_hub: checksum OK")

    os.replace(tmp, dest)
    logger.info("model_hub: model saved to %s", dest)
    print(f"Model saved to {dest}")
    return dest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _print_progress(downloaded: int, total: int) -> None:
    if total > 0:
        pct = downloaded / total * 100
        mb = downloaded / (1024 * 1024)
        total_mb = total / (1024 * 1024)
        print(f"\r  {pct:5.1f}%  {mb:.1f} / {total_mb:.1f} MB", end="", flush=True)
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)
