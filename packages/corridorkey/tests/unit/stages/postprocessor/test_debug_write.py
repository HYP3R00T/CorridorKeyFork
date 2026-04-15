"""Tests for _debug_write silent-failure path and both alpha/RGB branches."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
from corridorkey.stages.postprocessor.orchestrator import _debug_write


class TestDebugWriteSilentFailure:
    def test_exception_is_swallowed(self, tmp_path: Path):
        """Any exception inside _debug_write must not propagate to the caller."""
        arr = np.zeros((8, 8, 1), dtype=np.float32)
        with patch("cv2.imwrite", side_effect=RuntimeError("disk full")):
            _debug_write(tmp_path, "frame_000000", "test_tag", arr, is_alpha=True)
        # No exception raised — test passes

    def test_oserror_is_swallowed(self, tmp_path: Path):
        """OSError (e.g. permission denied on mkdir) is also swallowed."""
        arr = np.zeros((8, 8, 3), dtype=np.float32)
        with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
            _debug_write(tmp_path, "frame_000000", "test_tag", arr)
        # No exception raised


class TestDebugWriteOutput:
    def test_alpha_path_writes_png(self, tmp_path: Path):
        """is_alpha=True writes a greyscale PNG to the debug subdirectory."""
        arr = np.full((8, 8, 1), 0.5, dtype=np.float32)
        _debug_write(tmp_path, "frame_000001", "raw_alpha", arr, is_alpha=True)
        expected = tmp_path / "debug" / "frame_000001__raw_alpha.png"
        assert expected.is_file()

    def test_rgb_path_writes_png(self, tmp_path: Path):
        """is_alpha=False (default) writes an RGB PNG to the debug subdirectory."""
        arr = np.full((8, 8, 3), 0.5, dtype=np.float32)
        _debug_write(tmp_path, "frame_000002", "raw_fg", arr)
        expected = tmp_path / "debug" / "frame_000002__raw_fg.png"
        assert expected.is_file()

    def test_debug_dir_is_created(self, tmp_path: Path):
        """The debug/ subdirectory is created if it does not exist."""
        arr = np.zeros((4, 4, 1), dtype=np.float32)
        _debug_write(tmp_path, "frame_000000", "tag", arr, is_alpha=True)
        assert (tmp_path / "debug").is_dir()

    def test_2d_alpha_array_accepted(self, tmp_path: Path):
        """2D alpha array [H, W] (no channel dim) is handled by the alpha path."""
        arr = np.full((8, 8), 0.5, dtype=np.float32)
        _debug_write(tmp_path, "frame_000003", "alpha_2d", arr, is_alpha=True)
        expected = tmp_path / "debug" / "frame_000003__alpha_2d.png"
        assert expected.is_file()
