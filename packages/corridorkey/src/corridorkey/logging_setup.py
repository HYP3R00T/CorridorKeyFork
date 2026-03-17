"""Structured file logging for CorridorKey.

Every CLI session calls ``setup_logging()`` once at startup. It wires two
handlers onto the root logger:

- ``RichHandler``  - console, WARNING+ by default (DEBUG when verbose=True).
- ``RotatingFileHandler`` - session log file, INFO+ by default (or config.log_level).
  Writes newline-delimited JSON so logs are machine-readable and easy to grep.

Log files live in ``config.log_dir`` (default ``~/.config/corridorkey/logs``).
Each session creates a new file named ``YYMMDD_HHMMSS_corridorkey.log``, so
runs never overwrite each other. Up to 5 rotations of 5 MB each are kept per
session file before the oldest rotation is discarded.

Sharing a bug report:
    Share the session file printed at startup, e.g.
    ``~/.config/corridorkey/logs/260317_142301_corridorkey.log``.
"""

from __future__ import annotations

import datetime
import json
import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corridorkey.config import CorridorKeyConfig

# Sentinel so setup_logging is idempotent within a process.
_LOGGING_CONFIGURED = False

# Rotation policy.
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
_BACKUP_COUNT = 5  # keep 5 rotations (~25 MB total)


class _JsonFormatter(logging.Formatter):
    """Format each log record as a single-line JSON object.

    Fields emitted:
        ts      ISO-8601 timestamp with milliseconds (UTC)
        level   Log level name
        logger  Logger name (module path)
        msg     Formatted message string
        exc     Exception traceback string (only when an exception is attached)
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = (
            datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            + "Z"
        )
        obj: dict = {
            "ts": ts,
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)


def setup_logging(
    verbose: bool = False,
    config: CorridorKeyConfig | None = None,
) -> Path | None:
    """Configure root logging for a CLI session.

    Safe to call multiple times - only the first call has any effect.

    Args:
        verbose: When True, the console handler drops to DEBUG level.
            The file handler always uses config.log_level (default INFO).
        config: Loaded CorridorKeyConfig. When None, falls back to
            ``~/.config/corridorkey/logs`` and INFO level.

    Returns:
        Path to the session log file, or None if file logging could not
        be initialised (e.g. permission error on the log directory).
    """
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return None

    from rich.console import Console
    from rich.logging import RichHandler

    err_console = Console(stderr=True)

    # ------------------------------------------------------------------ #
    # Console handler - WARNING by default, DEBUG when verbose            #
    # ------------------------------------------------------------------ #
    console_level = logging.DEBUG if verbose else logging.WARNING
    console_handler = RichHandler(
        console=err_console,
        show_path=False,
        markup=True,
        rich_tracebacks=True,
    )
    console_handler.setLevel(console_level)

    # ------------------------------------------------------------------ #
    # File handler - session-named JSON log                               #
    # ------------------------------------------------------------------ #
    log_path: Path | None = None
    file_handler: logging.Handler | None = None

    log_dir = Path(config.log_dir) if config else Path("~/.config/corridorkey/logs").expanduser()
    file_level_name = config.log_level if config else "INFO"
    file_level = getattr(logging, file_level_name, logging.INFO)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        session_ts = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        log_path = (log_dir / f"{session_ts}_corridorkey.log").resolve()
        rotating = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        rotating.setLevel(file_level)
        rotating.setFormatter(_JsonFormatter())
        file_handler = rotating
    except OSError as exc:
        # Non-fatal - log to console only.
        logging.getLogger(__name__).warning("Could not open log file at %s: %s. Logging to console only.", log_dir, exc)

    # ------------------------------------------------------------------ #
    # Root logger                                                          #
    # ------------------------------------------------------------------ #
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers filter; root must pass everything
    root.handlers.clear()
    root.addHandler(console_handler)
    if file_handler:
        root.addHandler(file_handler)

    _LOGGING_CONFIGURED = True

    # ------------------------------------------------------------------ #
    # Session header - written once at startup                            #
    # ------------------------------------------------------------------ #
    _write_session_header(config)

    return log_path


def _write_session_header(config: CorridorKeyConfig | None) -> None:
    """Emit a structured INFO record with session metadata.

    This is the first entry in every log file for a new session, giving
    enough context to reproduce the environment when debugging.
    """
    logger = logging.getLogger(__name__)

    cuda_info: dict = {}
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cuda_info = {
                "device": torch.cuda.get_device_name(0),
                "vram_total_gb": round(props.total_mem / (1024**3), 2),
                "cuda_version": torch.version.cuda,
            }
    except Exception:
        pass

    header = {
        "event": "session_start",
        "python": sys.version,
        "platform": platform.platform(),
        "pid": os.getpid(),
        "cuda": cuda_info or None,
        "config": {
            "device": config.device if config else "unknown",
            "optimization_mode": config.optimization_mode if config else "unknown",
            "precision": config.precision if config else "unknown",
            "checkpoint_dir": str(config.checkpoint_dir) if config else "unknown",
            "log_dir": str(config.log_dir) if config else "unknown",
            "log_level": config.log_level if config else "INFO",
        }
        if config
        else None,
    }

    logger.info("CorridorKey session started | %s", json.dumps(header))


def reset_logging() -> None:
    """Reset the logging configuration sentinel.

    Intended for use in tests only - allows setup_logging to be called
    multiple times within the same process without the idempotency guard
    blocking subsequent calls.
    """
    global _LOGGING_CONFIGURED
    _LOGGING_CONFIGURED = False
