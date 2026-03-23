"""Logging configuration settings."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class LoggingSettings(BaseModel):
    """Cross-cutting logging settings.

    These apply to the whole pipeline regardless of which stage is running.
    The interface layer (CLI, GUI, web) adds its own handler on top — this
    controls only the file handler written to disk.

    In ``corridorkey.toml``::

        [logging]
        level = "INFO"
        dir = "~/.config/corridorkey/logs"
    """

    level: Annotated[
        Literal["DEBUG", "INFO", "WARNING", "ERROR"],
        Field(
            default="INFO",
            description=(
                "'DEBUG' adds verbose internal details. "
                "'INFO' captures all normal processing events (recommended). "
                "'WARNING' logs only problems."
            ),
        ),
    ] = "INFO"

    dir: Annotated[
        Path,
        Field(
            default=Path("~/.config/corridorkey/logs"),
            description=(
                "Directory where per-run timestamped log files are written. "
                "Share the latest log file when reporting bugs."
            ),
        ),
    ] = Path("~/.config/corridorkey/logs")
