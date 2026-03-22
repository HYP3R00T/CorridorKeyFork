"""Unit tests for corridorkey_new.stages.scanner.orchestrator — scan()."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey_new.errors import ClipScanError
from corridorkey_new.stages.scanner import scan


def _make_clip_dir(root: Path, with_alpha: bool = True) -> Path:
    clip = root / "my_clip"
    (clip / "Input").mkdir(parents=True)
    if with_alpha:
        (clip / "AlphaHint").mkdir()
    return clip


class TestScan:
    def test_returns_empty_for_empty_dir(self, tmp_path: Path):
        assert scan(tmp_path) == []

    def test_finds_single_clip(self, tmp_path: Path):
        _make_clip_dir(tmp_path)
        clips = scan(tmp_path)
        assert len(clips) == 1
        assert clips[0].name == "my_clip"

    def test_finds_multiple_clips(self, tmp_path: Path):
        for name in ("clip_a", "clip_b", "clip_c"):
            (tmp_path / name / "Input").mkdir(parents=True)
        assert len(scan(tmp_path)) == 3

    def test_clip_without_alpha_has_none_alpha_path(self, tmp_path: Path):
        _make_clip_dir(tmp_path, with_alpha=False)
        assert scan(tmp_path)[0].alpha_path is None

    def test_clip_with_alpha_has_alpha_path(self, tmp_path: Path):
        _make_clip_dir(tmp_path, with_alpha=True)
        assert scan(tmp_path)[0].alpha_path is not None

    def test_nonexistent_path_raises(self, tmp_path: Path):
        with pytest.raises(ClipScanError, match="does not exist"):
            scan(tmp_path / "ghost")

    def test_unrecognised_file_extension_raises(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.touch()
        with pytest.raises(ClipScanError, match="not a recognised video format"):
            scan(f)

    def test_skips_non_clip_subdirs(self, tmp_path: Path):
        (tmp_path / "not_a_clip").mkdir()
        assert scan(tmp_path) == []

    def test_directory_is_itself_a_clip(self, tmp_path: Path):
        (tmp_path / "Input").mkdir()
        assert len(scan(tmp_path)) == 1

    def test_case_insensitive_input_folder(self, tmp_path: Path):
        (tmp_path / "my_clip" / "input").mkdir(parents=True)
        assert len(scan(tmp_path)) == 1

    def test_case_insensitive_alphahint_folder(self, tmp_path: Path):
        clip = tmp_path / "my_clip"
        (clip / "Input").mkdir(parents=True)
        (clip / "alphahint").mkdir()
        assert scan(tmp_path)[0].alpha_path is not None

    def test_video_file_reorganised_when_reorganise_true(self, tmp_path: Path):
        (tmp_path / "clip.mp4").touch()
        clips = scan(tmp_path / "clip.mp4", reorganise=True)
        assert len(clips) == 1
        assert (tmp_path / "Input" / "clip.mp4").exists()

    def test_video_file_skipped_when_reorganise_false(self, tmp_path: Path):
        (tmp_path / "clip.mp4").touch()
        assert scan(tmp_path / "clip.mp4", reorganise=False) == []

    def test_loose_video_in_dir_reorganised(self, tmp_path: Path):
        (tmp_path / "clip.mp4").touch()
        clips = scan(tmp_path, reorganise=True)
        assert len(clips) == 1
        assert (tmp_path / "Input" / "clip.mp4").exists()

    def test_loose_video_in_dir_skipped_when_reorganise_false(self, tmp_path: Path):
        (tmp_path / "clip.mp4").touch()
        assert scan(tmp_path, reorganise=False) == []

    def test_video_inside_input_dir_used_as_input_path(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()
        clips = scan(tmp_path)
        assert clips[0].input_path == video

    def test_video_inside_alphahint_dir_used_as_alpha_path(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        (clip_dir / "Input").mkdir(parents=True)
        alpha_dir = clip_dir / "AlphaHint"
        alpha_dir.mkdir()
        (clip_dir / "Input" / "frame.png").touch()
        video = alpha_dir / "alpha.mp4"
        video.touch()
        clips = scan(tmp_path)
        assert clips[0].alpha_path == video
