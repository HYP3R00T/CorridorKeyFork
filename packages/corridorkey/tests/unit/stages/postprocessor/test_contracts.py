"""Unit tests for corridorkey.stages.postprocessor.contracts."""

from __future__ import annotations

import numpy as np
import pytest
from corridorkey.stages.postprocessor.contracts import ProcessedFrame


def _make_frame(h: int = 32, w: int = 32) -> ProcessedFrame:
    return ProcessedFrame(
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
        """alpha field is [H, W, 1]."""
        f = _make_frame(32, 32)
        assert f.alpha.shape == (32, 32, 1)

    def test_fg_shape(self):
        """fg field is [H, W, 3]."""
        f = _make_frame(32, 32)
        assert f.fg.shape == (32, 32, 3)

    def test_processed_shape(self):
        """processed field is [H, W, 4] — premultiplied RGBA."""
        f = _make_frame(32, 32)
        assert f.processed.shape == (32, 32, 4)

    def test_comp_shape(self):
        """comp field is [H, W, 3]."""
        f = _make_frame(32, 32)
        assert f.comp.shape == (32, 32, 3)

    def test_stem_default(self):
        """stem defaults to empty string when not provided."""
        f = ProcessedFrame(
            alpha=np.zeros((4, 4, 1), dtype=np.float32),
            fg=np.zeros((4, 4, 3), dtype=np.float32),
            processed=np.zeros((4, 4, 4), dtype=np.float32),
            comp=np.zeros((4, 4, 3), dtype=np.float32),
            frame_index=5,
            source_h=4,
            source_w=4,
        )
        assert f.stem == ""

    def test_source_h_and_w_stored(self):
        """source_h and source_w are stored as provided."""
        f = _make_frame(48, 64)
        assert f.source_h == 48
        assert f.source_w == 64

    def test_frame_index_stored(self):
        """frame_index is stored as provided."""
        f = ProcessedFrame(
            alpha=np.zeros((4, 4, 1), dtype=np.float32),
            fg=np.zeros((4, 4, 3), dtype=np.float32),
            processed=np.zeros((4, 4, 4), dtype=np.float32),
            comp=np.zeros((4, 4, 3), dtype=np.float32),
            frame_index=42,
            source_h=4,
            source_w=4,
        )
        assert f.frame_index == 42

    def test_is_frozen(self):
        """ProcessedFrame is frozen — attribute assignment raises."""
        f = _make_frame()
        with pytest.raises((AttributeError, TypeError, Exception)):
            f.frame_index = 99  # type: ignore[misc]

    def test_dtypes_float32(self):
        """All array fields are float32."""
        f = _make_frame()
        assert f.alpha.dtype == np.float32
        assert f.fg.dtype == np.float32
        assert f.comp.dtype == np.float32
