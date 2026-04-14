"""Unit tests for _safe_move — copy-verification failure and delete-failure paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from corridorkey.errors import ClipScanError
from corridorkey.stages.scanner.normaliser import _safe_move


class TestSafeMove:
    def test_moves_file_successfully(self, tmp_path: Path):
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"video data")
        dst = tmp_path / "dest" / "clip.mp4"
        dst.parent.mkdir()

        _safe_move(src, dst)

        assert dst.exists()
        assert dst.read_bytes() == b"video data"
        assert not src.exists()

    def test_raises_on_copy_failure(self, tmp_path: Path):
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"data")
        dst = tmp_path / "dest" / "clip.mp4"
        dst.parent.mkdir()

        with (
            patch("shutil.copy2", side_effect=OSError("disk full")),
            pytest.raises(ClipScanError, match="Failed to copy"),
        ):
            _safe_move(src, dst)

    def test_raises_on_size_mismatch(self, tmp_path: Path):
        """If dst size doesn't match src after copy, raises ClipScanError and removes dst."""
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"original data")
        dst = tmp_path / "dest" / "clip.mp4"
        dst.parent.mkdir()

        def fake_copy(s, d, **kwargs):
            Path(d).write_bytes(b"truncated")

        with (
            patch("shutil.copy2", side_effect=fake_copy),
            pytest.raises(ClipScanError, match="Copy verification failed"),
        ):
            _safe_move(src, dst)

        assert not dst.exists()

    def test_raises_on_delete_failure(self, tmp_path: Path):
        """If src deletion fails after successful copy, raises ClipScanError."""
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"data")
        dst = tmp_path / "dest" / "clip.mp4"
        dst.parent.mkdir()

        original_unlink = Path.unlink

        def fail_unlink(self, missing_ok=False):
            if self == src:
                raise OSError("permission denied")
            original_unlink(self, missing_ok=missing_ok)

        with (
            patch.object(Path, "unlink", fail_unlink),
            pytest.raises(ClipScanError, match="failed to delete source"),
        ):
            _safe_move(src, dst)

        assert dst.exists()
