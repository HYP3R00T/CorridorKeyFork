"""Unit tests for corridorkey.stages.loader.validator."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey.errors import FrameMismatchError
from corridorkey.stages.loader.validator import (
    FrameScan,
    count_frames,
    detect_is_linear,
    list_frames,
    scan_frames,
    validate,
)


def _touch_frames(directory: Path, names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


class TestScanFrames:
    def test_returns_frame_scan(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame_1.png", "frame_2.png"])
        result = scan_frames(tmp_path)
        assert isinstance(result, FrameScan)

    def test_frame_scan_count_property(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame_1.png", "frame_2.png", "frame_3.png"])
        result = scan_frames(tmp_path)
        assert result.count == 3

    def test_frame_scan_count_zero_when_empty(self, tmp_path: Path):
        result = scan_frames(tmp_path)
        assert result.count == 0

    def test_single_pass_returns_count_and_linearity(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame_1.exr", "frame_2.exr"])
        result = scan_frames(tmp_path)
        assert result.count == 2
        assert result.is_linear is True

    def test_png_not_linear(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame_1.png"])
        assert scan_frames(tmp_path).is_linear is False

    def test_empty_directory(self, tmp_path: Path):
        result = scan_frames(tmp_path)
        assert result.count == 0
        assert result.is_linear is False

    def test_non_directory_returns_empty(self, tmp_path: Path):
        f = tmp_path / "file.png"
        f.touch()
        result = scan_frames(f)
        assert result.count == 0

    def test_files_naturally_sorted(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame_10.png", "frame_2.png", "frame_1.png"])
        result = scan_frames(tmp_path)
        names = [f.name for f in result.files]
        assert names == ["frame_1.png", "frame_2.png", "frame_10.png"]

    def test_ignores_non_image_files(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame_1.png", "readme.txt", "data.json"])
        assert scan_frames(tmp_path).count == 1

    def test_all_image_extensions_recognised(self, tmp_path: Path):
        _touch_frames(tmp_path, ["a.exr", "b.png", "c.jpg", "d.jpeg", "e.tiff", "f.tif"])
        assert scan_frames(tmp_path).count == 6

    def test_case_insensitive_extensions(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame.PNG", "frame.EXR"])
        assert scan_frames(tmp_path).count == 2

    def test_linearity_from_first_frame_only(self, tmp_path: Path):
        # First frame (natural sort) is .png — should be non-linear even if .exr follows
        _touch_frames(tmp_path, ["frame_1.png", "frame_2.exr"])
        assert scan_frames(tmp_path).is_linear is False

    def test_frame_scan_is_frozen(self, tmp_path: Path):
        result = scan_frames(tmp_path)
        with pytest.raises(AttributeError):
            result.is_linear = True  # type: ignore[misc]


class TestGetFrameFiles:
    """Wrapper around scan_frames — verify it returns the file list correctly."""

    def test_returns_image_files_only(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame_1.png", "frame_2.png", "readme.txt"])
        result = list_frames(tmp_path)
        assert len(result) == 2
        assert all(f.suffix == ".png" for f in result)

    def test_natural_sort_order(self, tmp_path: Path):
        _touch_frames(tmp_path, ["frame_10.png", "frame_2.png", "frame_1.png"])
        result = list_frames(tmp_path)
        assert [f.name for f in result] == ["frame_1.png", "frame_2.png", "frame_10.png"]

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        assert list_frames(tmp_path) == []


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


class TestValidate:
    def test_valid_clip_no_alpha_returns_scans(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        _touch_frames(input_dir, ["frame_1.png", "frame_2.png"])
        input_scan, alpha_scan = validate("test_clip", input_dir, None)
        assert input_scan.count == 2
        assert alpha_scan is None

    def test_valid_clip_with_matching_alpha(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        alpha_dir = tmp_path / "AlphaHint"
        _touch_frames(input_dir, ["frame_1.png", "frame_2.png"])
        _touch_frames(alpha_dir, ["alpha_1.png", "alpha_2.png"])
        input_scan, alpha_scan = validate("test_clip", input_dir, alpha_dir)
        assert input_scan.count == 2
        assert alpha_scan is not None
        assert alpha_scan.count == 2

    def test_empty_input_raises(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        from corridorkey.errors import ClipLoadError

        with pytest.raises(ClipLoadError, match="no image frames"):
            validate("test_clip", input_dir, None)

    def test_frame_count_mismatch_raises(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        alpha_dir = tmp_path / "AlphaHint"
        _touch_frames(input_dir, ["frame_1.png", "frame_2.png"])
        _touch_frames(alpha_dir, ["alpha_1.png"])
        with pytest.raises(FrameMismatchError, match="frame count mismatch"):
            validate("test_clip", input_dir, alpha_dir)

    def test_expected_frame_count_skips_input_rescan(self, tmp_path: Path):
        """When expected_frame_count is provided, input dir is not re-scanned."""
        input_dir = tmp_path / "Input"
        alpha_dir = tmp_path / "AlphaHint"
        _touch_frames(input_dir, ["frame_1.png", "frame_2.png"])
        _touch_frames(alpha_dir, ["alpha_1.png", "alpha_2.png"])
        # Pass expected_frame_count=2 — input_dir scan is skipped
        input_scan, alpha_scan = validate("test_clip", input_dir, alpha_dir, expected_frame_count=2)
        # input_scan is a placeholder when expected_frame_count is used
        assert alpha_scan is not None
        assert alpha_scan.count == 2

    def test_expected_frame_count_mismatch_raises(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        alpha_dir = tmp_path / "AlphaHint"
        _touch_frames(input_dir, ["frame_1.png", "frame_2.png"])
        _touch_frames(alpha_dir, ["alpha_1.png"])  # only 1, expected 2
        with pytest.raises(FrameMismatchError):
            validate("test_clip", input_dir, alpha_dir, expected_frame_count=2)
