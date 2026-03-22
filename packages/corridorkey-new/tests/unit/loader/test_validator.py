"""Unit tests for corridorkey_new.stages.loader.validator."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey_new.errors import FrameMismatchError
from corridorkey_new.stages.loader.validator import (
    count_frames,
    detect_is_linear,
    get_frame_files,
    validate,
)


def _touch_frames(directory: Path, names: list[str]) -> None:
    """Create empty files with the given names in directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


class TestGetFrameFiles:
    def test_returns_image_files_only(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame_1.png", "frame_2.png", "readme.txt", "data.json"])
        result = get_frame_files(tmp_path)
        assert len(result) == 2
        assert all(f.suffix == ".png" for f in result)

    def test_natural_sort_order(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame_10.png", "frame_2.png", "frame_1.png"])
        result = get_frame_files(tmp_path)
        assert [f.name for f in result] == ["frame_1.png", "frame_2.png", "frame_10.png"]

    def test_all_image_extensions_recognised(self, tmp_path: Path):
        names = ["a.exr", "b.png", "c.jpg", "d.jpeg", "e.tiff", "f.tif"]
        _touch_frames(tmp_path, names)
        result = get_frame_files(tmp_path)
        assert len(result) == 6

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        assert get_frame_files(tmp_path) == []

    def test_non_directory_returns_empty(self, tmp_path: Path):
        f = tmp_path / "file.png"
        f.touch()
        assert get_frame_files(f) == []

    def test_case_insensitive_extensions(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame.PNG", "frame.EXR"])
        result = get_frame_files(tmp_path)
        assert len(result) == 2


class TestCountFrames:
    def test_counts_image_files(self, tmp_path: Path):
        _touch_frames(tmp_path, ["a.png", "b.png", "c.png"])
        assert count_frames(tmp_path) == 3

    def test_ignores_non_image_files(self, tmp_path: Path):
        _touch_frames(tmp_path, ["a.png", "b.txt"])
        assert count_frames(tmp_path) == 1

    def test_empty_directory_returns_zero(self, tmp_path: Path):
        assert count_frames(tmp_path) == 0


class TestDetectIsLinear:
    def test_exr_frames_are_linear(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame_1.exr", "frame_2.exr"])
        assert detect_is_linear(tmp_path) is True

    def test_png_frames_are_not_linear(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame_1.png", "frame_2.png"])
        assert detect_is_linear(tmp_path) is False

    def test_empty_directory_returns_false(self, tmp_path: Path):
        assert detect_is_linear(tmp_path) is False

    def test_uses_first_frame_extension(self, tmp_path: Path):
        # First frame (natural sort) is .png, second is .exr — should return False
        _touch_frames(tmp_path, ["frame_1.png", "frame_2.exr"])
        assert detect_is_linear(tmp_path) is False


class TestValidate:
    def test_valid_clip_no_alpha(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        _touch_frames(input_dir, ["frame_1.png", "frame_2.png"])
        validate("test_clip", input_dir, None)  # should not raise

    def test_valid_clip_with_matching_alpha(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        alpha_dir = tmp_path / "AlphaHint"
        _touch_frames(input_dir, ["frame_1.png", "frame_2.png"])
        _touch_frames(alpha_dir, ["alpha_1.png", "alpha_2.png"])
        validate("test_clip", input_dir, alpha_dir)  # should not raise

    def test_empty_input_raises(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        with pytest.raises(ValueError, match="no image frames"):
            validate("test_clip", input_dir, None)

    def test_frame_count_mismatch_raises(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        alpha_dir = tmp_path / "AlphaHint"
        _touch_frames(input_dir, ["frame_1.png", "frame_2.png"])
        _touch_frames(alpha_dir, ["alpha_1.png"])
        with pytest.raises(FrameMismatchError, match="frame count mismatch"):
            validate("test_clip", input_dir, alpha_dir)
