"""Postprocessor stage — output contract."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PostprocessedFrame:
    """Output contract of the postprocessor stage. Input to the writer stage.

    All arrays are at original source resolution, float32, numpy.

    Attributes:
        alpha: Alpha matte [H, W, 1], linear, range 0-1.
        fg: Foreground RGB [H, W, 3], sRGB straight, range 0-1.
            In transparent regions the values are undefined — use ``processed``
            for compositing work.
        processed: Premultiplied linear RGBA [H, W, 4], range 0-1.
            This is the primary output for compositing. Transparent regions are
            correctly zeroed out (fg * alpha), so no black-blob artefacts.
        comp: Preview composite over checkerboard [H, W, 3], sRGB, range 0-1.
        frame_index: Frame index carried through from FrameMeta.
        source_h: Original frame height in pixels.
        source_w: Original frame width in pixels.
        stem: Filename stem for output naming (e.g. "frame_000001").
    """

    alpha: np.ndarray
    fg: np.ndarray
    processed: np.ndarray
    comp: np.ndarray
    frame_index: int
    source_h: int
    source_w: int
    stem: str = ""
