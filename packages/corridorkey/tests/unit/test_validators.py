"""Unit tests for validators.py.

Validators are called on every frame in the inference loop. A wrong result
here either silently corrupts output (wrong dtype, wrong channel count) or
raises the wrong exception type, breaking error recovery in the caller.
Tests cover frame count validation, mask normalisation, and the write
success check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from corridorkey.errors import FrameMismatchError, FrameReadError, MaskChannelError, WriteFailureError
from corridorkey.validators import (
    ensure_output_dirs,
    normalize_mask_channels,
    normalize_mask_dtype,
    validate_frame_counts,
    validate_frame_read,
    validate_write,
)


class TestValidateFrameCounts:
    """validate_frame_counts - frame count agreement between source and mask sequences."""

    def test_equal_counts_returns_count(self):
        """Equal source and mask counts must return that count unchanged."""
        assert validate_frame_counts("clip", 100, 100) == 100

    def test_mismatch_non_strict_returns_minimum(self):
        """A count mismatch in non-strict mode must return the smaller of the two counts."""
        assert validate_frame_counts("clip", 100, 90) == 90

    def test_mismatch_strict_raises(self):
        """A count mismatch in strict mode must raise FrameMismatchError naming the clip."""
        with pytest.raises(FrameMismatchError, match="clip"):
            validate_frame_counts("clip", 100, 90, strict=True)

    def test_zero_counts(self):
        """Both counts being zero must return zero without raising."""
        assert validate_frame_counts("clip", 0, 0) == 0


class TestNormalizeMaskChannels:
    """normalize_mask_channels - channel reduction to a single 2-D float32 mask."""

    def test_2d_passthrough(self):
        """A 2-D mask must be returned as-is without modification."""
        mask = np.ones((4, 4), dtype=np.float32)
        result = normalize_mask_channels(mask)
        assert result.shape == (4, 4)

    def test_3d_extracts_first_channel(self):
        """A 3-D mask must have its first channel extracted as the alpha channel."""
        mask = np.zeros((4, 4, 3), dtype=np.float32)
        mask[:, :, 0] = 0.5
        mask[:, :, 1] = 1.0
        result = normalize_mask_channels(mask)
        assert result.shape == (4, 4)
        assert np.allclose(result, 0.5)

    def test_zero_channels_raises(self):
        """A mask with zero channels must raise MaskChannelError."""
        mask = np.zeros((4, 4, 0), dtype=np.float32)
        with pytest.raises(MaskChannelError):
            normalize_mask_channels(mask, "clip", 0)

    def test_4d_raises(self):
        """A 4-D mask must raise MaskChannelError as the shape is unsupported."""
        mask = np.zeros((4, 4, 3, 1), dtype=np.float32)
        with pytest.raises(MaskChannelError):
            normalize_mask_channels(mask, "clip", 0)

    def test_output_dtype_is_float32(self):
        """The output must always be float32 regardless of the input dtype."""
        mask = np.ones((4, 4), dtype=np.uint8)
        result = normalize_mask_channels(mask)
        assert result.dtype == np.float32


class TestNormalizeMaskDtype:
    """normalize_mask_dtype - dtype normalisation to float32 [0, 1]."""

    def test_uint8_normalized(self):
        """uint8 values must be divided by 255 to produce float32 in [0, 1]."""
        mask = np.array([[255, 0]], dtype=np.uint8)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        assert np.isclose(result[0, 0], 1.0)
        assert np.isclose(result[0, 1], 0.0)

    def test_uint16_normalized(self):
        """uint16 values must be divided by 65535 to produce float32 in [0, 1]."""
        mask = np.array([[65535]], dtype=np.uint16)
        result = normalize_mask_dtype(mask)
        assert np.isclose(result[0, 0], 1.0, atol=1e-5)

    def test_float32_passthrough(self):
        """A float32 mask must be returned as the same object with no copy."""
        mask = np.array([[0.5]], dtype=np.float32)
        result = normalize_mask_dtype(mask)
        assert result is mask  # same object, no copy

    def test_float64_cast(self):
        """A float64 mask must be cast to float32 with values preserved."""
        mask = np.array([[0.75]], dtype=np.float64)
        result = normalize_mask_dtype(mask)
        assert result.dtype == np.float32
        assert np.isclose(result[0, 0], 0.75)


class TestValidateFrameRead:
    """validate_frame_read - guards against None frames from failed disk reads."""

    def test_valid_frame_returned(self):
        """A non-None frame must be returned unchanged."""
        frame = np.zeros((4, 4, 3), dtype=np.float32)
        result = validate_frame_read(frame, "clip", 0, "/path")
        assert result is frame

    def test_none_raises(self):
        """A None frame must raise FrameReadError naming the clip."""
        with pytest.raises(FrameReadError, match="clip"):
            validate_frame_read(None, "clip", 5, "/path/frame.exr")


class TestValidateWrite:
    """validate_write - guards against silent write failures."""

    def test_success_no_raise(self):
        """A True success flag must not raise."""
        validate_write(True, "clip", 0, "/path")

    def test_failure_raises(self):
        """A False success flag must raise WriteFailureError naming the clip."""
        with pytest.raises(WriteFailureError, match="clip"):
            validate_write(False, "clip", 3, "/path/frame.exr")


class TestEnsureOutputDirs:
    """ensure_output_dirs - creates the standard Output subdirectory tree."""

    def test_creates_all_subdirs(self, tmp_path: Path):
        """All expected output subdirectories must be created and returned in the dict."""
        dirs = ensure_output_dirs(str(tmp_path / "clip"))
        for key in ("root", "fg", "matte", "comp", "processed"):
            assert key in dirs
            assert Path(dirs[key]).is_dir()

    def test_idempotent(self, tmp_path: Path):
        """Calling ensure_output_dirs twice on the same path must not raise."""
        clip_root = str(tmp_path / "clip")
        ensure_output_dirs(clip_root)
        ensure_output_dirs(clip_root)  # should not raise

    def test_root_is_output_subdir(self, tmp_path: Path):
        """The root key must point to a directory named Output inside the clip root."""
        dirs = ensure_output_dirs(str(tmp_path / "clip"))
        assert dirs["root"].endswith("Output")
