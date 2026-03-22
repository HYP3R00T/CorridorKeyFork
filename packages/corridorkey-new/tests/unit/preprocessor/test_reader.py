"""Unit tests for corridorkey_new.stages.preprocessor.reader."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from corridorkey_new.stages.preprocessor.reader import FrameReadError, _read_frame_pair


def _write_png(path: Path, h: int = 64, w: int = 64, channels: int = 3) -> None:
    """Write a random uint8 PNG to path."""
    img = np.zeros((h, w), dtype=np.uint8) if channels == 1 else np.zeros((h, w, channels), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_uint16_png(path: Path, h: int = 64, w: int = 64) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint16)
    cv2.imwrite(str(path), img)


class TestReadFramePair:
    def test_returns_float32_arrays(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, channels=3)
        _write_png(alpha_path, channels=1)
        image, alpha = _read_frame_pair(img_path, alpha_path)
        assert image.dtype == np.float32
        assert alpha.dtype == np.float32

    def test_image_shape_is_hwc3(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, h=32, w=48, channels=3)
        _write_png(alpha_path, h=32, w=48, channels=1)
        image, alpha = _read_frame_pair(img_path, alpha_path)
        assert image.shape == (32, 48, 3)

    def test_alpha_shape_is_hwc1(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, h=32, w=48, channels=3)
        _write_png(alpha_path, h=32, w=48, channels=1)
        image, alpha = _read_frame_pair(img_path, alpha_path)
        assert alpha.shape == (32, 48, 1)

    def test_values_in_range_0_1(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path)
        _write_png(alpha_path, channels=1)
        image, alpha = _read_frame_pair(img_path, alpha_path)
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0

    def test_uint16_png_normalised(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_uint16_png(img_path)
        _write_png(alpha_path, channels=1)
        image, _ = _read_frame_pair(img_path, alpha_path)
        assert image.dtype == np.float32
        assert image.max() <= 1.0

    def test_dimension_mismatch_resizes_alpha_with_warning(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, h=64, w=64, channels=3)
        _write_png(alpha_path, h=32, w=32, channels=1)  # half size
        image, alpha = _read_frame_pair(img_path, alpha_path)
        assert alpha.shape[:2] == image.shape[:2]

    def test_missing_image_raises(self, tmp_path: Path):
        alpha_path = tmp_path / "alpha.png"
        _write_png(alpha_path, channels=1)
        with pytest.raises(FrameReadError, match="cv2.imread returned None"):
            _read_frame_pair(tmp_path / "ghost.png", alpha_path)

    def test_missing_alpha_raises(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        _write_png(img_path)
        with pytest.raises(FrameReadError, match="cv2.imread returned None"):
            _read_frame_pair(img_path, tmp_path / "ghost.png")

    def test_rgb_alpha_hint_collapsed_to_single_channel(self, tmp_path: Path):
        """An RGB alpha hint (3-channel) should be collapsed to 1 channel."""
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, channels=3)
        _write_png(alpha_path, channels=3)  # RGB alpha hint
        _, alpha = _read_frame_pair(img_path, alpha_path)
        assert alpha.shape[2] == 1

    def test_bgr_to_rgb_conversion(self, tmp_path: Path):
        """A pure-red BGR image should come back as pure-red RGB."""
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        # Pure red in BGR is (0, 0, 255)
        red_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        red_bgr[:, :, 2] = 255
        cv2.imwrite(str(img_path), red_bgr)
        _write_png(alpha_path, h=4, w=4, channels=1)
        image, _ = _read_frame_pair(img_path, alpha_path)
        # After BGR->RGB, red channel (index 0) should be 1.0
        assert image[0, 0, 0] == pytest.approx(1.0)
        assert image[0, 0, 1] == pytest.approx(0.0)
        assert image[0, 0, 2] == pytest.approx(0.0)
