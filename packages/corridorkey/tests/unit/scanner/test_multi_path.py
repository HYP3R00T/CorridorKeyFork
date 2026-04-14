"""Tests for scan() with multiple paths."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey.errors import ClipScanError
from corridorkey.stages.scanner import scan


def _make_clip_dir(root: Path, name: str = "clip") -> Path:
    clip = root / name
    (clip / "Input").mkdir(parents=True)
    return clip


class TestScanMultiplePaths:
    def test_single_path_in_list(self, tmp_path: Path):
        _make_clip_dir(tmp_path)
        result = scan([tmp_path])
        assert result.clip_count == 1

    def test_two_separate_paths_aggregated(self, tmp_path: Path):
        dir_a = tmp_path / "project_a"
        dir_b = tmp_path / "project_b"
        dir_a.mkdir()
        dir_b.mkdir()
        _make_clip_dir(dir_a, "clip_a")
        _make_clip_dir(dir_b, "clip_b")

        result = scan([dir_a, dir_b])
        assert result.clip_count == 2
        names = {c.name for c in result.clips}
        assert names == {"clip_a", "clip_b"}

    def test_skipped_aggregated_across_paths(self, tmp_path: Path):
        dir_a = tmp_path / "project_a"
        dir_b = tmp_path / "project_b"
        dir_a.mkdir()
        dir_b.mkdir()
        # Both have ambiguous clips
        for d in (dir_a, dir_b):
            input_dir = d / "bad_clip" / "Input"
            input_dir.mkdir(parents=True)
            (input_dir / "a.mp4").touch()
            (input_dir / "b.mp4").touch()

        result = scan([dir_a, dir_b])
        assert result.clip_count == 0
        assert result.skipped_count == 2

    def test_mixed_valid_and_skipped_across_paths(self, tmp_path: Path):
        dir_a = tmp_path / "project_a"
        dir_b = tmp_path / "project_b"
        dir_a.mkdir()
        dir_b.mkdir()
        _make_clip_dir(dir_a, "good_clip")
        input_dir = dir_b / "bad_clip" / "Input"
        input_dir.mkdir(parents=True)
        (input_dir / "a.mp4").touch()
        (input_dir / "b.mp4").touch()

        result = scan([dir_a, dir_b])
        assert result.clip_count == 1
        assert result.skipped_count == 1

    def test_nonexistent_path_in_list_raises(self, tmp_path: Path):
        with pytest.raises(ClipScanError, match="does not exist"):
            scan([tmp_path / "ghost"])

    def test_string_path_still_accepted(self, tmp_path: Path):
        _make_clip_dir(tmp_path)
        result = scan(str(tmp_path))
        assert result.clip_count == 1

    def test_empty_list_returns_empty_result(self, tmp_path: Path):
        result = scan([])
        assert result.clip_count == 0
        assert result.skipped_count == 0
