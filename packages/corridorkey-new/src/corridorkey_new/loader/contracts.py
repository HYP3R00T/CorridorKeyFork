"""Stage 1 contracts."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class ClipManifest(BaseModel):
    """Output contract of stage 1. Input to stage 2 or stage 3.

    Attributes:
        clip_name: Name of the clip, carried through for logging and output naming.
        input_path: Validated path to the input frames directory.
        alpha_path: Validated path to the alpha frames directory. None if absent.
        needs_alpha: True if alpha is missing and stage 2 must run before stage 3.
        frame_count: Number of input frames detected.
        is_linear: True if input frames are in linear light (detected from .exr extension).
    """

    clip_name: str
    input_path: Path
    alpha_path: Path | None
    needs_alpha: bool
    frame_count: int
    is_linear: bool
