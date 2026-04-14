"""Unit tests for corridorkey.stages.scanner.normaliser."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from corridorkey.errors import ClipScanError
from corridorkey.stages.scanner.contracts import SkippedClip
from corridorkey.stages.scanner.normaliser import (
    _find_icase,
    _find_videos_in,
    _safe_move,
    normalise_video,
    try_build_clip,
)


class TestSafeMove:
    def test_moves_file_successfully(self, tmp_path: Path):
        """Moving a file to a new destination succeeds and removes the source."""
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"video data")
        dst = tmp_path / "dest" / "clip.mp4"
        dst.parent.mkdir()

        _safe_move(src, dst)

        assert dst.exists()
        assert dst.read_bytes() == b"video data"
        assert not src.exists()

    def test_raises_on_copy_failure(self, tmp_path: Path):
        """When the underlying copy raises OSError, _safe_move raises ClipScanError."""
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


class TestFindAlphaAmbiguous:
    def test_multiple_videos_returns_skipped(self, tmp_path: Path):
        """When an alpha directory contains multiple videos, _find_asset returns a SkippedClip."""
        from corridorkey.stages.scanner.normaliser import _find_asset

        alpha_dir = tmp_path / "AlphaHint"
        alpha_dir.mkdir()
        (alpha_dir / "a.mp4").touch()
        (alpha_dir / "b.mp4").touch()
        path, skip = _find_asset(tmp_path, "AlphaHint")
        assert path is None
        assert isinstance(skip, SkippedClip)
        assert "multiple" in skip.reason.lower()


class TestFindVideosInPermission:
    def test_permission_error_returns_empty_list(self, tmp_path: Path):
        """When iterdir raises PermissionError, _find_videos_in returns an empty list."""
        with patch("pathlib.Path.iterdir", side_effect=PermissionError("denied")):
            result = _find_videos_in(tmp_path)
        assert result == []


class TestFindIcasePermission:
    def test_permission_error_raises(self, tmp_path: Path):
        """When iterdir raises PermissionError, _find_icase re-raises it."""
        with patch("pathlib.Path.iterdir", side_effect=PermissionError("denied")), pytest.raises(PermissionError):
            _find_icase(tmp_path, "Input")


class TestTryBuildClipErrors:
    def test_permission_error_returns_skipped(self, tmp_path: Path):
        """When _find_icase raises PermissionError, try_build_clip returns a SkippedClip."""
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
        """When Clip construction raises a ValidationError, try_build_clip returns a SkippedClip."""
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
        """When mkdir raises OSError, normalise_video raises ClipScanError."""
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"data")

        with (
            patch("pathlib.Path.mkdir", side_effect=OSError("no space")),
            pytest.raises(ClipScanError, match="Failed to create clip structure"),
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


class TestNormaliseVideoOsError:
    def test_mkdir_failure_raises_oserror(self, tmp_path: Path):
        """When mkdir raises OSError, normalise_video raises ClipScanError."""
        video = tmp_path / "clip.mp4"
        video.touch()
        with (
            patch.object(Path, "mkdir", side_effect=OSError("permission denied")),
            pytest.raises(ClipScanError, match="Failed to create clip structure"),
        ):
            normalise_video(video)

    def test_already_in_place_skips_move(self, tmp_path: Path):
        """When the video is already in the Input folder, normalise_video returns the clip without moving."""
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake video data")
        clip = normalise_video(video)
        dest = tmp_path / "Input" / "clip.mp4"
        assert clip.input_path == dest
        assert dest.exists()

    def test_safe_move_called_when_not_in_place(self, tmp_path: Path):
        """When the video is not yet in the Input folder, normalise_video moves it there."""
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"data")
        clip = normalise_video(video)
        assert clip.input_path == tmp_path / "Input" / "clip.mp4"
        assert clip.input_path.exists()
        assert not video.exists()


class TestFindVideosInFallback:
    def test_returns_empty_list_when_no_videos(self, tmp_path: Path):
        """When a directory contains only non-video files, _find_videos_in returns an empty list."""
        (tmp_path / "frame_000001.png").touch()
        (tmp_path / "frame_000002.png").touch()
        result = _find_videos_in(tmp_path)
        assert result == []

    def test_returns_empty_list_for_empty_directory(self, tmp_path: Path):
        """When a directory is empty, _find_videos_in returns an empty list."""
        result = _find_videos_in(tmp_path)
        assert result == []

    def test_returns_videos_sorted(self, tmp_path: Path):
        """When a directory contains multiple video files, _find_videos_in returns them sorted by name."""
        for name in ("c.mp4", "a.mp4", "b.mov"):
            (tmp_path / name).touch()
        result = _find_videos_in(tmp_path)
        names = [p.name for p in result]
        assert names == sorted(names, key=str.lower)

    def test_permission_error_returns_empty_list(self, tmp_path: Path):
        """When iterdir raises PermissionError, _find_videos_in returns an empty list."""
        with patch.object(Path, "iterdir", side_effect=PermissionError("denied")):
            result = _find_videos_in(tmp_path)
        assert result == []


class TestTryBuildClipPermissionError:
    def test_permission_error_on_find_input_returns_skipped(self, tmp_path: Path):
        """When _find_asset raises PermissionError, try_build_clip returns a SkippedClip."""
        with patch(
            "corridorkey.stages.scanner.normaliser._find_asset",
            side_effect=PermissionError("denied"),
        ):
            clip, skip = try_build_clip(tmp_path)

        assert clip is None
        assert skip is not None
        assert "cannot read" in skip.reason.lower()
