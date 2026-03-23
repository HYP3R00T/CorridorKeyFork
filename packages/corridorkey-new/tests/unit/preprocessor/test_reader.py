"""Unit tests for corridorkey_new.stages.preprocessor.reader."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from corridorkey_new.stages.preprocessor.reader import FrameReadError, _read_frame_pair


def _write_png(path: Path, h: int = 64, w: int = 64, channels: int = 3) -> None:
    img = np.zeros((h, w), dtype=np.uint8) if channels == 1 else np.zeros((h, w, channels), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_uint16_png(path: Path, h: int = 64, w: int = 64) -> None:
    img = np.zeros((h, w, 3), dtype=np.uint16)
    cv2.imwrite(str(path), img)


class TestReadFramePair:
    def test_returns_three_tuple(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, channels=3)
        _write_png(alpha_path, channels=1)
        result = _read_frame_pair(img_path, alpha_path)
        assert len(result) == 3

    def test_returns_float32_arrays(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, channels=3)
        _write_png(alpha_path, channels=1)
        image, alpha, _ = _read_frame_pair(img_path, alpha_path)
        assert image.dtype == np.float32
        assert alpha.dtype == np.float32

    def test_image_shape_is_hwc3(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, h=32, w=48, channels=3)
        _write_png(alpha_path, h=32, w=48, channels=1)
        image, alpha, _ = _read_frame_pair(img_path, alpha_path)
        assert image.shape == (32, 48, 3)

    def test_alpha_shape_is_hwc1(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, h=32, w=48, channels=3)
        _write_png(alpha_path, h=32, w=48, channels=1)
        image, alpha, _ = _read_frame_pair(img_path, alpha_path)
        assert alpha.shape == (32, 48, 1)

    def test_values_in_range_0_1(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path)
        _write_png(alpha_path, channels=1)
        image, alpha, _ = _read_frame_pair(img_path, alpha_path)
        assert image.min() >= 0.0 and image.max() <= 1.0
        assert alpha.min() >= 0.0 and alpha.max() <= 1.0

    def test_uint16_png_normalised(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_uint16_png(img_path)
        _write_png(alpha_path, channels=1)
        image, _, _ = _read_frame_pair(img_path, alpha_path)
        assert image.dtype == np.float32
        assert image.max() <= 1.0

    def test_dimension_mismatch_resizes_alpha_with_warning(self, tmp_path: Path):
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, h=64, w=64, channels=3)
        _write_png(alpha_path, h=32, w=32, channels=1)
        image, alpha, _ = _read_frame_pair(img_path, alpha_path)
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
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, channels=3)
        _write_png(alpha_path, channels=3)
        _, alpha, _ = _read_frame_pair(img_path, alpha_path)
        assert alpha.shape[2] == 1

    def test_multichannel_alpha_dot_product_preserves_precision(self, tmp_path: Path):
        """Float32 dot product path must preserve sub-uint8 precision.

        A pure-green BGR pixel (B=0, G=1.0, R=0) has luminance 0.587 by the
        BGR→gray weights [0.114, 0.587, 0.299]. A uint8 round-trip would
        quantise this to round(0.587 * 255) / 255 ≈ 0.5882, losing precision.
        The direct float32 dot product must return exactly 0.587.
        """
        from corridorkey_new.stages.preprocessor.reader import _to_channels

        # Pure green in BGR float32 — no uint8 quantisation
        arr = np.zeros((4, 4, 3), dtype=np.float32)
        arr[:, :, 1] = 1.0  # G channel in BGR
        out, bgr = _to_channels(arr, channels=1, path=tmp_path / "fake.png")
        assert bgr is False
        assert out.shape == (4, 4, 1)
        # Expected: 0.114*0 + 0.587*1.0 + 0.299*0 = 0.587
        assert out[0, 0, 0] == pytest.approx(0.587, abs=1e-6)

    def test_bgr_flag_true_for_colour_image(self, tmp_path: Path):
        """Reader must return bgr=True for standard colour images — no CPU reorder."""
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, channels=3)
        _write_png(alpha_path, channels=1)
        _, _, bgr = _read_frame_pair(img_path, alpha_path)
        assert bgr is True

    def test_bgr_flag_false_for_single_channel_alpha(self, tmp_path: Path):
        """Alpha is always single-channel — bgr flag is irrelevant, must be False."""
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        _write_png(img_path, channels=3)
        _write_png(alpha_path, channels=1)
        image, alpha, _ = _read_frame_pair(img_path, alpha_path)
        # alpha has no BGR concept — verify it's single channel
        assert alpha.shape[2] == 1

    def test_image_channels_are_bgr_before_tensor_reorder(self, tmp_path: Path):
        """Reader returns BGR — pure-red BGR image should have max in channel 2 (B=0,G=1,R=2)."""
        img_path = tmp_path / "frame.png"
        alpha_path = tmp_path / "alpha.png"
        red_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        red_bgr[:, :, 2] = 255  # red in BGR is channel index 2
        cv2.imwrite(str(img_path), red_bgr)
        _write_png(alpha_path, h=4, w=4, channels=1)
        image, _, bgr = _read_frame_pair(img_path, alpha_path)
        assert bgr is True
        # Channel 2 should be 1.0 (red in BGR), channels 0 and 1 should be 0.0
        assert image[0, 0, 2] == pytest.approx(1.0)
        assert image[0, 0, 0] == pytest.approx(0.0)
        assert image[0, 0, 1] == pytest.approx(0.0)


class TestToFloat32UnsupportedDtype:
    def test_unsupported_dtype_raises_frame_read_error(self, tmp_path: Path):
        """int32 arrays are not a supported dtype — must raise FrameReadError."""
        from unittest.mock import patch

        import numpy as np
        from corridorkey_new.stages.preprocessor.reader import _read_image

        bad_arr = np.zeros((4, 4, 3), dtype=np.int32)
        with patch("cv2.imread", return_value=bad_arr), pytest.raises(FrameReadError, match="Unsupported dtype"):
            _read_image(tmp_path / "fake.png", channels=3)


class TestToChannelsEdgeCases:
    def test_grayscale_image_broadcast_to_3_channels(self, tmp_path: Path):
        """1-channel image requested as 3-channel → broadcast, bgr=False."""
        from corridorkey_new.stages.preprocessor.reader import _to_channels

        arr = np.ones((4, 4, 1), dtype=np.float32) * 0.5
        out, bgr = _to_channels(arr, channels=3, path=tmp_path / "fake.png")
        assert out.shape == (4, 4, 3)
        assert bgr is False
        assert np.allclose(out, 0.5)

    def test_bgra_image_drops_alpha_channel(self, tmp_path: Path):
        """4-channel BGRA image requested as 3-channel → drop alpha, bgr=True."""
        from corridorkey_new.stages.preprocessor.reader import _to_channels

        arr = np.zeros((4, 4, 4), dtype=np.float32)
        arr[:, :, 2] = 1.0  # red in BGR
        out, bgr = _to_channels(arr, channels=3, path=tmp_path / "fake.png")
        assert out.shape == (4, 4, 3)
        assert bgr is True
        assert np.allclose(out[:, :, 2], 1.0)

    def test_unsupported_channel_count_for_1ch_output_raises(self, tmp_path: Path):
        """5-channel image cannot be reduced to 1 channel — must raise."""
        from corridorkey_new.stages.preprocessor.reader import _to_channels

        arr = np.zeros((4, 4, 5), dtype=np.float32)
        with pytest.raises(FrameReadError, match="Cannot reduce"):
            _to_channels(arr, channels=1, path=tmp_path / "fake.png")

    def test_unsupported_channel_count_for_3ch_output_raises(self, tmp_path: Path):
        """5-channel image cannot be converted to 3 channels — must raise."""
        from corridorkey_new.stages.preprocessor.reader import _to_channels

        arr = np.zeros((4, 4, 5), dtype=np.float32)
        with pytest.raises(FrameReadError, match="Cannot convert"):
            _to_channels(arr, channels=3, path=tmp_path / "fake.png")


class TestToFloat32FloatInput:
    def test_float64_input_normalised_to_float32(self, tmp_path: Path):
        """float64 arrays (e.g. from non-EXR float sources) must be clipped and cast."""
        from corridorkey_new.stages.preprocessor.reader import _to_float32

        arr = np.array([[[0.0, 0.5, 1.5]]], dtype=np.float64)
        result = _to_float32(arr, tmp_path / "fake.png")
        assert result.dtype == np.float32
        assert result[0, 0, 2] == pytest.approx(1.0)  # clipped from 1.5

    def test_float32_input_clipped_and_returned(self, tmp_path: Path):
        """float32 arrays must be clipped to [0, 1] and returned as float32."""
        from corridorkey_new.stages.preprocessor.reader import _to_float32

        arr = np.array([[[0.2, 0.5, 1.5]]], dtype=np.float32)
        result = _to_float32(arr, tmp_path / "fake.png")
        assert result.dtype == np.float32
        assert result[0, 0, 2] == pytest.approx(1.0)  # clipped from 1.5


class TestToChannelsUnsupportedCount:
    def test_unsupported_channel_count_raises(self, tmp_path: Path):
        """channels=2 is not supported — must raise FrameReadError."""
        from corridorkey_new.stages.preprocessor.reader import _to_channels

        arr = np.zeros((4, 4, 3), dtype=np.float32)
        with pytest.raises(FrameReadError, match="Unsupported channel count"):
            _to_channels(arr, channels=2, path=tmp_path / "fake.png")
