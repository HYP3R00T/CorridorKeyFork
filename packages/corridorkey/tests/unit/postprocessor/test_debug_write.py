"""Tests for _debug_write silent-failure path (lines 156-157)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from corridorkey.stages.postprocessor.orchestrator import _debug_write


class TestDebugWriteSilentFailure:
    def test_exception_is_swallowed(self, tmp_path: Path):
        """Any exception inside _debug_write must not propagate."""
        import numpy as np

        arr = np.zeros((8, 8, 1), dtype=np.float32)
        with patch("cv2.imwrite", side_effect=RuntimeError("disk full")):
            _debug_write(tmp_path, "frame_000000", "test_tag", arr, is_alpha=True)
        # No exception raised — test passes

    def test_oserror_is_swallowed(self, tmp_path: Path):
        """OSError (e.g. permission denied on mkdir) is also swallowed."""
        import numpy as np

        arr = np.zeros((8, 8, 3), dtype=np.float32)
        with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
            _debug_write(tmp_path, "frame_000000", "test_tag", arr)
        # No exception raised
