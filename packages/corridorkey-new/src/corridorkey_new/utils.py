"""Shared utilities for the CorridorKey pipeline."""

from __future__ import annotations

import re


def natural_sort_key(s: str) -> list[int | str]:
    """Return a sort key that orders strings with embedded numbers naturally.

    Examples:
        frame_2.exr  sorts before  frame_10.exr
        shot_1_v2    sorts before  shot_1_v10
    """
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]
