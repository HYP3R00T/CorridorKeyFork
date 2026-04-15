"""Unit tests for corridorkey.stages.inference.contracts."""

from __future__ import annotations

import torch
from corridorkey.stages.inference.contracts import InferenceResult
from corridorkey.stages.preprocessor.contracts import FrameMeta


def _make_result(h: int = 32, w: int = 32) -> InferenceResult:
    meta = FrameMeta(frame_index=0, original_h=64, original_w=64)
    return InferenceResult(
        alpha=torch.zeros(1, 1, h, w),
        fg=torch.zeros(1, 3, h, w),
        meta=meta,
    )


class TestInferenceResult:
    def test_alpha_shape(self):
        r = _make_result(32, 32)
        assert r.alpha.shape == (1, 1, 32, 32)

    def test_fg_shape(self):
        r = _make_result(32, 32)
        assert r.fg.shape == (1, 3, 32, 32)

    def test_meta_carried_through(self):
        r = _make_result()
        assert r.meta.frame_index == 0
        assert r.meta.original_h == 64
        assert r.meta.original_w == 64

    def test_is_frozen(self):
        r = _make_result()
        import pytest

        with pytest.raises((AttributeError, TypeError, Exception)):
            r.alpha = torch.ones(1, 1, 32, 32)  # type: ignore[misc]

    def test_alpha_dtype_float32(self):
        r = _make_result()
        assert r.alpha.dtype == torch.float32

    def test_fg_dtype_float32(self):
        r = _make_result()
        assert r.fg.dtype == torch.float32

    def test_non_square_shapes(self):
        r = _make_result(h=16, w=48)
        assert r.alpha.shape == (1, 1, 16, 48)
        assert r.fg.shape == (1, 3, 16, 48)


class TestInferenceResultValidation:
    def test_bad_alpha_ndim_raises(self):
        import pytest

        meta = FrameMeta(frame_index=0, original_h=32, original_w=32)
        with pytest.raises(ValueError, match="alpha must be"):
            InferenceResult(alpha=torch.zeros(1, 32, 32), fg=torch.zeros(1, 3, 32, 32), meta=meta)

    def test_bad_alpha_channels_raises(self):
        import pytest

        meta = FrameMeta(frame_index=0, original_h=32, original_w=32)
        with pytest.raises(ValueError, match="alpha must be"):
            InferenceResult(alpha=torch.zeros(1, 2, 32, 32), fg=torch.zeros(1, 3, 32, 32), meta=meta)

    def test_bad_fg_channels_raises(self):
        import pytest

        meta = FrameMeta(frame_index=0, original_h=32, original_w=32)
        with pytest.raises(ValueError, match="fg must be"):
            InferenceResult(alpha=torch.zeros(1, 1, 32, 32), fg=torch.zeros(1, 4, 32, 32), meta=meta)

    def test_bad_fg_ndim_raises(self):
        """fg with wrong number of dimensions raises ValueError."""
        import pytest

        meta = FrameMeta(frame_index=0, original_h=32, original_w=32)
        with pytest.raises(ValueError, match="fg must be"):
            InferenceResult(alpha=torch.zeros(1, 1, 32, 32), fg=torch.zeros(1, 32, 32), meta=meta)

    def test_spatial_mismatch_raises(self):
        import pytest

        meta = FrameMeta(frame_index=0, original_h=32, original_w=32)
        with pytest.raises(ValueError, match="spatial size mismatch"):
            InferenceResult(alpha=torch.zeros(1, 1, 32, 32), fg=torch.zeros(1, 3, 16, 16), meta=meta)

    def test_batch_size_mismatch_raises(self):
        import pytest

        meta = FrameMeta(frame_index=0, original_h=32, original_w=32)
        with pytest.raises(ValueError, match="batch size mismatch"):
            InferenceResult(alpha=torch.zeros(1, 1, 32, 32), fg=torch.zeros(2, 3, 32, 32), meta=meta)
