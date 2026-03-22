"""Unit tests for corridorkey_new.stages.postprocessor.contracts."""

from __future__ import annotations

import numpy as np
import pytest
from corridorkey_new.stages.postprocessor.contracts import PostprocessedFrame


def _make_frame(h: int = 32, w: int = 32) -> PostprocessedFrame:
    return PostprocessedFrame(
        alpha=np.zeros((h, w, 1), dtype=np.float32),
        fg=np.zeros((h, w, 3), dtype=np.float32),
        processed=np.zeros((h, w, 4), dtype=np.float32),
        comp=np.zeros((h, w, 3), dtype=np.float32),
        frame_index=0,
        source_h=h,
        source_w=w,
        stem="frame_000000",
    )


class TestPostprocessedFrame:
    def test_alpha_shape(self):
        f = _make_frame(32, 32)
        assert f.alpha.shape == (32, 32, 1)

    def test_fg_shape(self):
        f = _make_frame(32, 32)
        assert f.fg.shape == (32, 32, 3)

    def test_comp_shape(self):
        f = _make_frame(32, 32)
        assert f.comp.shape == (32, 32, 3)

    def test_stem_default(self):
        f = PostprocessedFrame(
            alpha=np.zeros((4, 4, 1), dtype=np.float32),
            fg=np.zeros((4, 4, 3), dtype=np.float32),
            processed=np.zeros((4, 4, 4), dtype=np.float32),
            comp=np.zeros((4, 4, 3), dtype=np.float32),
            frame_index=5,
            source_h=4,
            source_w=4,
        )
        assert f.stem == ""

    def test_is_frozen(self):
        f = _make_frame()
        with pytest.raises((AttributeError, TypeError, Exception)):
            f.frame_index = 99  # type: ignore[misc]

    def test_dtypes_float32(self):
        f = _make_frame()
        assert f.alpha.dtype == np.float32
        assert f.fg.dtype == np.float32
        assert f.comp.dtype == np.float32
