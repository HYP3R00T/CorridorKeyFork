"""Unit tests for corridorkey.stages.scanner.normaliser."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from corridorkey.stages.scanner.contracts import SkippedClip
from corridorkey.stages.scanner.normaliser import (
    _find_icase,
    _find_videos_in,
    _safe_move,
    find_alpha,
    normalise_video,
    try_build_clip,
)


class TestSafeMove:
    def test_copies_and_deletes_source(self, tmp_path: Path):
        src = tmp_path / "src.mp4"
        src.write_bytes(b"data")
        dst = tmp_path / "dst.mp4"
        _safe_move(src, dst)
        assert dst.read_bytes() == b"data"
        assert not src.exists()

    def test_copy_failure_raises_oserror(self, tmp_path: Path):
        src = tmp_path / "src.mp4"
        src.write_bytes(b"data")
        dst = tmp_path / "dst.mp4"
        with patch("shutil.copy2", side_effect=OSError("disk full")), pytest.raises(OSError, match="Failed to copy"):
            _safe_move(src, dst)

    def test_size_mismatch_raises_oserror(self, tmp_path: Path):
        src = tmp_path / "src.mp4"
        src.write_bytes(b"hello")
        dst = tmp_path / "dst.mp4"
        # Write a different-sized file to dst before the move to simulate mismatch
        dst.write_bytes(b"hi")
        with (
            patch("shutil.copy2"),
            pytest.raises(OSError, match="Copy verification failed"),
        ):  # no-op copy — dst already has wrong size
            _safe_move(src, dst)

    def test_delete_failure_raises_oserror(self, tmp_path: Path):
        src = tmp_path / "src.mp4"
        src.write_bytes(b"data")
        dst = tmp_path / "dst.mp4"
        with (
            patch("shutil.copy2", side_effect=lambda s, d: Path(d).write_bytes(b"data")),
            patch.object(Path, "unlink", side_effect=OSError("locked")),
            pytest.raises(OSError, match="failed to delete source"),
        ):
            _safe_move(src, dst)


class TestFindAlphaAmbiguous:
    def test_multiple_videos_returns_skipped(self, tmp_path: Path):
        alpha_dir = tmp_path / "AlphaHint"
        alpha_dir.mkdir()
        (alpha_dir / "a.mp4").touch()
        (alpha_dir / "b.mp4").touch()
        path, skip = find_alpha(tmp_path)
        assert path is None
        assert isinstance(skip, SkippedClip)
        assert "multiple" in skip.reason.lower()


class TestFindVideosInPermission:
    def test_permission_error_returns_empty_list(self, tmp_path: Path):
        with patch("pathlib.Path.iterdir", side_effect=PermissionError("denied")):
            result = _find_videos_in(tmp_path)
        assert result == []


class TestFindIcasePermission:
    def test_permission_error_raises(self, tmp_path: Path):
        with patch("pathlib.Path.iterdir", side_effect=PermissionError("denied")), pytest.raises(PermissionError):
            _find_icase(tmp_path, "Input")


class TestTryBuildClipErrors:
    def test_permission_error_returns_skipped(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        clip_dir.mkdir()
        with patch(
            "corridorkey.stages.scanner.normaliser._find_icase",
            side_effect=PermissionError("denied"),
        ):
            clip, skip = try_build_clip(clip_dir)
        assert clip is None
        assert isinstance(skip, SkippedClip)
        assert "cannot read directory" in skip.reason

    def test_validation_error_returns_skipped(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)

        import corridorkey.stages.scanner.normaliser as mod

        # Trigger a real ValidationError by passing None for required fields
        original_clip = mod.Clip

        def _bad_clip(**kwargs):
            return original_clip(name=None, root=None, input_path=None, alpha_path=None)  # type: ignore

        with patch.object(mod, "Clip", side_effect=lambda **kw: _bad_clip(**kw)):
            clip, skip = try_build_clip(clip_dir)

        assert clip is None
        assert isinstance(skip, SkippedClip)
        assert "validation failed" in skip.reason


class TestNormaliseVideoErrors:
    def test_mkdir_failure_raises_oserror(self, tmp_path: Path):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"data")
        with (
            patch("pathlib.Path.mkdir", side_effect=OSError("no space")),
            pytest.raises(OSError, match="Failed to create clip structure"),
        ):
            normalise_video(video)


class TestNormaliseVideoIdempotent:
    def test_already_in_place_skips_move(self, tmp_path: Path):
        """If dest already exists with same size, _safe_move is not called."""
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        video = input_dir / "clip.mp4"
        video.write_bytes(b"data")
        # normalise_video expects the video to be at clip_root level, not inside Input/
        # So create a video at tmp_path level that already has a matching dest
        video_src = tmp_path / "clip.mp4"
        video_src.write_bytes(b"data")
        # Pre-create the dest with same content (input_dir / "clip.mp4" already exists)
        # dest already exists from above — normalise_video should skip _safe_move
        with patch("corridorkey.stages.scanner.normaliser._safe_move") as mock_move:
            normalise_video(video_src)
            mock_move.assert_not_called()


class TestTryBuildClipNoInputFolder:
    def test_no_input_folder_returns_none_none(self, tmp_path: Path):
        """Directory with no Input/ folder returns (None, None)."""
        clip_dir = tmp_path / "not_a_clip"
        clip_dir.mkdir()
        clip, skip = try_build_clip(clip_dir)
        assert clip is None
        assert skip is None


class TestNormaliseVideoMove:
    def test_moves_video_when_dest_does_not_exist(self, tmp_path: Path):
        """When dest doesn't exist, normalise_video calls _safe_move."""
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"video_data")
        clip = normalise_video(video)
        assert clip.input_path.exists()
        assert not video.exists()  # source was moved
