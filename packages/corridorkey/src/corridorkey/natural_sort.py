"""Natural sort key for frame filenames.

Handles non-zero-padded frame numbers correctly so that frame_2 sorts
before frame_10 rather than after it (lexicographic order).

No external dependencies - pure Python implementation.
"""

from __future__ import annotations

import re

# Regex that splits a string on digit runs, capturing the digits.
_SPLIT_RE = re.compile(r"(\d+)")


def natural_sort_key(text: str) -> list[str | int]:
    """Return a sort key that orders numeric substrings numerically.

    Args:
        text: String to generate a sort key for.

    Returns:
        List of alternating string and int parts suitable for comparison.

    Example:
        >>> sorted(["f_1", "f_10", "f_2"], key=natural_sort_key)
        ['f_1', 'f_2', 'f_10']
    """
    parts: list[str | int] = []
    for chunk in _SPLIT_RE.split(text):
        try:
            parts.append(int(chunk))
        except ValueError:
            parts.append(chunk.lower())
    return parts


def natsorted(items: list[str]) -> list[str]:
    """Return a naturally sorted copy of a list of strings.

    Args:
        items: List of strings to sort.

    Returns:
        New list sorted using natural_sort_key.
    """
    return sorted(items, key=natural_sort_key)
