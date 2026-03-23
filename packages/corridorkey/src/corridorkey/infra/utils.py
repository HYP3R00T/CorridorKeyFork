"""Shared utilities for the CorridorKey pipeline."""

from __future__ import annotations

import re

# Single source of truth for recognised video extensions.
# Imported by both the scanner (normaliser.py) and the loader (extractor.py).
VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4", ".mov", ".avi", ".mkv", ".mxf", ".webm", ".m4v",
})


def natural_sort_key(s: str) -> list[int | str]:
    """Return a sort key that orders strings with embedded numbers naturally.

    Examples:
        frame_2.exr  sorts before  frame_10.exr
        shot_1_v2    sorts before  shot_1_v10
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]
