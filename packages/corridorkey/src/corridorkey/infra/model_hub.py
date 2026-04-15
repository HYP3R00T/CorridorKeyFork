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
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from corridorkey.errors import ModelError

logger = logging.getLogger(__name__)

MODEL_URL = "https://huggingface.co/nikopueringer/CorridorKey_v1.0/resolve/main/CorridorKey_v1.0.pth"
MODEL_FILENAME = "CorridorKey_v1.0.pth"
MODEL_CHECKSUM = "a03827f58e8c79b2ca26031bf67c77db5390dc1718c1ffc5b7aed8b57315788f"
DEFAULT_MODEL_DIR = Path("~/.config/corridorkey/models")

_DOWNLOAD_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt


def default_checkpoint_path() -> Path:
    """Return the expected path for the default model checkpoint."""
    return DEFAULT_MODEL_DIR.expanduser() / MODEL_FILENAME


def ensure_model(
    dest_dir: Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Return the checkpoint path, downloading it first if it doesn't exist.

    Retries up to 3 times with exponential backoff (2s, 4s, 8s) before
    raising ModelError.

    Args:
        dest_dir: Directory to store the model. Defaults to DEFAULT_MODEL_DIR.
        on_progress: Optional callback(bytes_downloaded, total_bytes).

    Returns:
        Absolute path to the verified checkpoint file.

    Raises:
        ModelError: If the download fails after all retries or the checksum
            doesn't match.
    """
    directory = (dest_dir or DEFAULT_MODEL_DIR).expanduser()
    dest = directory / MODEL_FILENAME

    if dest.is_file():
        logger.debug("model_hub: checkpoint already present at %s", dest)
        return dest

    directory.mkdir(parents=True, exist_ok=True)
    tmp = directory / (MODEL_FILENAME + ".tmp")

    _download_with_retry(tmp, on_progress)
    _verify_checksum(tmp)

    os.replace(tmp, dest)
    logger.info("model_hub: model saved to %s", dest)
    return dest


def _download_with_retry(tmp: Path, on_progress: Callable[[int, int], None] | None) -> None:
    """Download MODEL_URL to ``tmp``, retrying up to _DOWNLOAD_RETRIES times.

    Raises:
        ModelError: If all attempts fail.
    """
    last_error: Exception | None = None
    for attempt in range(1, _DOWNLOAD_RETRIES + 1):
        if attempt > 1:
            delay = _RETRY_BASE_DELAY * (2 ** (attempt - 2))
            logger.warning(
                "model_hub: download attempt %d/%d failed, retrying in %.0fs — %s",
                attempt - 1,
                _DOWNLOAD_RETRIES,
                delay,
                last_error,
            )
            time.sleep(delay)

        logger.info(
            "model_hub: downloading model from %s (attempt %d/%d)",
            MODEL_URL,
            attempt,
            _DOWNLOAD_RETRIES,
        )
        try:
            with urllib.request.urlopen(MODEL_URL) as response:  # noqa: S310
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk = 64 * 1024
                last_log_pct = -1

                with open(tmp, "wb") as f:
                    while True:
                        data = response.read(chunk)
                        if not data:
                            break
                        f.write(data)
                        downloaded += len(data)
                        last_log_pct = _report_progress(downloaded, total, last_log_pct, on_progress)
            last_error = None
            break  # success
        except Exception as e:
            last_error = e
            tmp.unlink(missing_ok=True)

    if last_error is not None:
        raise ModelError(f"Model download failed after {_DOWNLOAD_RETRIES} attempt(s): {last_error}") from last_error

    logger.info("model_hub: download complete")


def _report_progress(
    downloaded: int,
    total: int,
    last_log_pct: int,
    on_progress: Callable[[int, int], None] | None,
) -> int:
    """Fire progress callback or log at most once per 5%. Returns updated last_log_pct."""
    if on_progress:
        on_progress(downloaded, total)
        return last_log_pct
    if total > 0:
        pct = int(downloaded / total * 100)
        if pct >= last_log_pct + 5:
            mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            logger.info("model_hub: downloading %d%% (%.1f / %.1f MB)", pct, mb, total_mb)
            return pct
    return last_log_pct


def _verify_checksum(tmp: Path) -> None:
    """Verify the SHA-256 of ``tmp`` against MODEL_CHECKSUM.

    Raises:
        ModelError: If the checksum does not match. Removes ``tmp`` before raising.
    """
    if not MODEL_CHECKSUM:
        return
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


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
