"""Logging setup for CorridorKey.

Call ``setup_logging(config)`` once at application startup. All modules use
``logging.getLogger(__name__)`` and inherit this configuration automatically.

This module only configures the file handler — the interface layer (CLI, GUI,
TUI, web) is responsible for adding its own handler for user-facing output.

Log file:
- One timestamped file per run: ``corridorkey_YYYY-MM-DD_HH-MM-SS.log``
- Level controlled by ``config.log_level`` (default INFO)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from corridorkey.infra.config import CorridorKeyConfig

_FILE_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(config: CorridorKeyConfig) -> None:
    """Configure the root logger with a per-run timestamped file handler.

    Safe to call multiple times — existing handlers are cleared first.
    The interface layer is responsible for adding its own handler for
    user-facing output (console, GUI widget, web stream, etc.).

    Args:
        config: Loaded CorridorKeyConfig. Uses log_dir and log_level.

    Returns:
        Path to the log file created for this run.
    """
    log_dir = Path(config.logging.dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"corridorkey_{timestamp}.log"

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)  # handlers filter individually

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(config.logging.level)
    file_handler.setFormatter(logging.Formatter(_FILE_FORMAT))
    root.addHandler(file_handler)

    logging.getLogger(__name__).debug("Logging initialised: file=%s level=%s", log_file, config.logging.level)
