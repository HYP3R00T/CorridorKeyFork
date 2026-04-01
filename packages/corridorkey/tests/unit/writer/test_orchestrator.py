"""Unit tests for corridorkey.stages.writer.orchestrator."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from corridorkey.stages.postprocessor.contracts import PostprocessedFrame
from corridorkey.stages.writer.contracts import WriteConfig
from corridorkey.stages.writer.orchestrator import _alpha_to_bgr, _exr_flags, write_frame


def _make_frame(h: int = 16, w: int = 16, stem: str = "frame_000000") -> PostprocessedFrame:
    return PostprocessedFrame(
        alpha=np.full((h, w, 1), 0.5, dtype=np.float32),
        fg=np.full((h, w, 3), 0.4, dtype=np.float32),
        processed=np.full((h, w, 4), 0.2, dtype=np.float32),
        comp=np.full((h, w, 3), 0.3, dtype=np.float32),
        frame_index=0,
        source_h=h,
        source_w=w,
        stem=stem,
    )


class TestAlphaToBgr:
    def test_shape_hwc1_to_hw3(self):
        alpha = np.zeros((8, 8, 1), dtype=np.float32)
        out = _alpha_to_bgr(alpha)
        assert out.shape == (8, 8, 3)

    def test_shape_hw_to_hw3(self):
        alpha = np.zeros((8, 8), dtype=np.float32)
        out = _alpha_to_bgr(alpha)
        assert out.shape == (8, 8, 3)

    def test_values_replicated_across_channels(self):
        alpha = np.full((4, 4, 1), 0.7, dtype=np.float32)
        out = _alpha_to_bgr(alpha)
        assert np.allclose(out[:, :, 0], 0.7)
        assert np.allclose(out[:, :, 1], 0.7)
        assert np.allclose(out[:, :, 2], 0.7)


class TestExrFlags:
    def test_returns_list_of_ints(self):
        flags = _exr_flags("dwaa")
        assert isinstance(flags, list)
        assert all(isinstance(f, int) for f in flags)

    def test_unknown_compression_falls_back_to_dwaa(self):
        flags_dwaa = _exr_flags("dwaa")
        flags_unknown = _exr_flags("unknown_codec")
        assert flags_dwaa == flags_unknown


class TestWriteFrame:
    def test_writes_alpha_png(self, tmp_path: Path):
        frame = _make_frame(stem="f0")
        cfg = WriteConfig(output_dir=tmp_path, fg_enabled=False, comp_enabled=False, processed_enabled=False)
        write_frame(frame, cfg)
        assert (tmp_path / "alpha" / "f0.png").exists()

    def test_writes_fg_png(self, tmp_path: Path):
        frame = _make_frame(stem="f0")
        cfg = WriteConfig(output_dir=tmp_path, alpha_enabled=False, comp_enabled=False, processed_enabled=False)
        write_frame(frame, cfg)
        assert (tmp_path / "fg" / "f0.png").exists()

    def test_writes_comp_png(self, tmp_path: Path):
        frame = _make_frame(stem="f0")
        cfg = WriteConfig(output_dir=tmp_path, alpha_enabled=False, fg_enabled=False, processed_enabled=False)
        write_frame(frame, cfg)
        assert (tmp_path / "comp" / "f0.png").exists()

    def test_creates_subdirectories(self, tmp_path: Path):
        frame = _make_frame(stem="f0")
        cfg = WriteConfig(output_dir=tmp_path, processed_format="png")
        write_frame(frame, cfg)
        assert (tmp_path / "alpha").is_dir()
        assert (tmp_path / "fg").is_dir()
        assert (tmp_path / "comp").is_dir()
        assert (tmp_path / "processed").is_dir()

    def test_disabled_outputs_not_written(self, tmp_path: Path):
        frame = _make_frame(stem="f0")
        cfg = WriteConfig(
            output_dir=tmp_path,
            alpha_enabled=False,
            fg_enabled=False,
            comp_enabled=False,
            processed_enabled=False,
        )
        write_frame(frame, cfg)
        assert not (tmp_path / "alpha").exists()
        assert not (tmp_path / "fg").exists()
        assert not (tmp_path / "comp").exists()
        assert not (tmp_path / "processed").exists()

    def test_png_pixel_values_correct(self, tmp_path: Path):
        frame = _make_frame(stem="f0")
        cfg = WriteConfig(output_dir=tmp_path, fg_enabled=False, comp_enabled=False, processed_enabled=False)
        write_frame(frame, cfg)
        img = cv2.imread(str(tmp_path / "alpha" / "f0.png"))
        assert img is not None
        # 0.5 float → 127 or 128 uint8
        assert 126 <= img[0, 0, 0] <= 129

    def test_multiple_frames_sequential(self, tmp_path: Path):
        cfg = WriteConfig(output_dir=tmp_path, fg_enabled=False, comp_enabled=False, processed_enabled=False)
        for i in range(3):
            frame = _make_frame(stem=f"frame_{i:06d}")
            write_frame(frame, cfg)
        assert len(list((tmp_path / "alpha").glob("*.png"))) == 3

    def test_raises_on_bad_path(self, tmp_path: Path):
        frame = _make_frame(stem="f0")
        # Make output_dir a file so mkdir fails
        bad_dir = tmp_path / "not_a_dir.txt"
        bad_dir.write_text("x")
        cfg = WriteConfig(output_dir=bad_dir, fg_enabled=False, comp_enabled=False, processed_enabled=False)
        with pytest.raises((OSError, NotADirectoryError, Exception)):
            write_frame(frame, cfg)


class TestWriteInternalPaths:
    def test_sixteen_bit_png_written(self, tmp_path: Path):
        """processed_format=png triggers the sixteen_bit branch in _write."""
        from corridorkey.stages.writer.orchestrator import _write

        img = np.full((8, 8, 4), 0.5, dtype=np.float32)
        path = tmp_path / "out.png"
        _write(img, path, "png", [], sixteen_bit=True)
        assert path.exists()
        loaded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert loaded is not None
        assert loaded.dtype == np.uint16

    def test_write_failure_raises(self, tmp_path: Path):
        """cv2.imwrite returning False raises WriteFailureError."""
        from unittest.mock import patch

        from corridorkey.errors import WriteFailureError
        from corridorkey.stages.writer.orchestrator import _write

        img = np.zeros((4, 4, 3), dtype=np.uint8)
        path = tmp_path / "out.png"
        with patch("cv2.imwrite", return_value=False), pytest.raises(WriteFailureError):
            _write(img, path, "png", [])


class TestProcessedPngColourSpace:
    """The processed PNG must have sRGB-encoded colour channels, not linear.

    A linear mid-grey (0.5) written as-is to PNG would display as ~0.214 sRGB
    (dark). After correct linear-to-sRGB conversion it should be ~0.735.
    """

    def _make_frame(self, rgb_linear: float, alpha: float = 1.0) -> PostprocessedFrame:
        h, w = 8, 8
        # processed is premultiplied linear RGBA
        processed = np.full((h, w, 4), 0.0, dtype=np.float32)
        processed[:, :, :3] = rgb_linear * alpha  # premultiplied
        processed[:, :, 3] = alpha
        return PostprocessedFrame(
            alpha=np.full((h, w, 1), alpha, dtype=np.float32),
            fg=np.full((h, w, 3), rgb_linear, dtype=np.float32),
            processed=processed,
            comp=np.zeros((h, w, 3), dtype=np.float32),
            frame_index=0,
            source_h=h,
            source_w=w,
            stem="frame_000000",
        )

    def test_processed_png_rgb_is_srgb_encoded(self, tmp_path: Path):
        """Mid-grey linear (0.5) must be written as ~0.735 sRGB, not 0.5."""
        frame = self._make_frame(rgb_linear=0.5, alpha=1.0)
        cfg = WriteConfig(
            output_dir=tmp_path,
            alpha_enabled=False,
            fg_enabled=False,
            comp_enabled=False,
            processed_format="png",
        )
        write_frame(frame, cfg)

        path = tmp_path / "processed" / "frame_000000.png"
        assert path.exists()
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert img is not None
        assert img.dtype == np.uint16

        # Channel 0 is B in BGRA. For a fully opaque mid-grey premultiplied frame
        # the RGB channels equal the linear value (0.5). After sRGB encoding
        # 0.5 linear → ~0.735 sRGB → ~48168 out of 65535.
        # A raw linear write would give ~32767. We check it's clearly above that.
        b_value = img[0, 0, 0]
        assert b_value > 40000, (
            f"processed PNG B channel is {b_value} — looks like linear data was "
            f"written without sRGB conversion (expected ~48168 for 0.5 linear)"
        )

    def test_processed_exr_rgb_stays_linear(self, tmp_path: Path):
        """EXR must NOT apply sRGB conversion — compositors expect linear data."""
        import pytest

        pytest.importorskip("cv2")
        import os

        if not os.environ.get("OPENCV_IO_ENABLE_OPENEXR"):
            pytest.skip("OpenEXR not enabled in this OpenCV build (set OPENCV_IO_ENABLE_OPENEXR=1)")
        frame = self._make_frame(rgb_linear=0.5, alpha=1.0)
        cfg = WriteConfig(
            output_dir=tmp_path,
            alpha_enabled=False,
            fg_enabled=False,
            comp_enabled=False,
            processed_format="exr",
        )
        write_frame(frame, cfg)

        path = tmp_path / "processed" / "frame_000000.exr"
        assert path.exists()
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert img is not None

        # EXR is float32. The premultiplied linear value for fully opaque 0.5 is 0.5.
        b_value = float(img[0, 0, 0])
        assert abs(b_value - 0.5) < 0.01, (
            f"processed EXR B channel is {b_value:.4f} — expected ~0.5 (linear, no sRGB conversion)"
        )
