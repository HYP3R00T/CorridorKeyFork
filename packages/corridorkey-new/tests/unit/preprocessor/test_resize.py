"""Unit tests for corridorkey_new.preprocessor.resize."""

from __future__ import annotations

import numpy as np
from corridorkey_new.preprocessor.resize import resize_frame


def _make_image(h: int, w: int) -> np.ndarray:
    return np.random.rand(h, w, 3).astype(np.float32)


def _make_alpha(h: int, w: int) -> np.ndarray:
    return np.random.rand(h, w, 1).astype(np.float32)


class TestResizeFrame:
    def test_squish_output_shape_image(self):
        image, alpha = resize_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512, "squish")
        assert image.shape == (512, 512, 3)

    def test_squish_output_shape_alpha(self):
        image, alpha = resize_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512, "squish")
        assert alpha.shape == (512, 512, 1)

    def test_already_square_unchanged_shape(self):
        image, alpha = resize_frame(_make_image(512, 512), _make_alpha(512, 512), 512, "squish")
        assert image.shape == (512, 512, 3)
        assert alpha.shape == (512, 512, 1)

    def test_output_dtype_float32(self):
        image, alpha = resize_frame(_make_image(64, 64), _make_alpha(64, 64), 32, "squish")
        assert image.dtype == np.float32
        assert alpha.dtype == np.float32

    def test_values_stay_in_range(self):
        image, alpha = resize_frame(_make_image(64, 64), _make_alpha(64, 64), 32, "squish")
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0

    def test_letterbox_falls_back_to_squish(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            image, alpha = resize_frame(_make_image(1080, 1920), _make_alpha(1080, 1920), 512, "letterbox")
        assert image.shape == (512, 512, 3)
        assert "not yet implemented" in caplog.text

    def test_image_and_alpha_same_spatial_size(self):
        image, alpha = resize_frame(_make_image(200, 300), _make_alpha(200, 300), 128, "squish")
        assert image.shape[:2] == alpha.shape[:2]
