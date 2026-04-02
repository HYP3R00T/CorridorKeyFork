"""Additional ClipManifest tests — covering the start >= end validation branch."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey.stages.loader.contracts import ClipManifest


def _make_manifest(tmp_path: Path, **overrides) -> ClipManifest:
    frames = tmp_path / "Frames"
    frames.mkdir(exist_ok=True)
    output = tmp_path / "Output"
    output.mkdir(exist_ok=True)
    return ClipManifest(
        clip_name=str(overrides.get("clip_name", "test")),
        clip_root=overrides.get("clip_root", tmp_path),
        frames_dir=overrides.get("frames_dir", frames),
        alpha_frames_dir=overrides.get("alpha_frames_dir"),
        output_dir=overrides.get("output_dir", output),
        needs_alpha=bool(overrides.get("needs_alpha", False)),
        frame_count=int(overrides.get("frame_count", 10)),
        frame_range=overrides.get("frame_range", (0, 10)),
        is_linear=bool(overrides.get("is_linear", False)),
    )


class TestClipManifestFrameRangeValidation:
    def test_start_equal_to_end_raises(self, tmp_path: Path):
        """start == end is invalid — range must contain at least one frame."""
        with pytest.raises(Exception, match="must be less than end"):
            _make_manifest(tmp_path, frame_count=10, frame_range=(5, 5))

    def test_start_greater_than_end_raises(self, tmp_path: Path):
        with pytest.raises(Exception, match="must be less than end"):
            _make_manifest(tmp_path, frame_count=10, frame_range=(7, 3))

    def test_start_negative_raises(self, tmp_path: Path):
        with pytest.raises(Exception, match="frame_range start"):
            _make_manifest(tmp_path, frame_count=10, frame_range=(-1, 5))

    def test_end_exceeds_frame_count_raises(self, tmp_path: Path):
        with pytest.raises(Exception, match="frame_range end"):
            _make_manifest(tmp_path, frame_count=10, frame_range=(0, 11))

    def test_valid_partial_range(self, tmp_path: Path):
        m = _make_manifest(tmp_path, frame_count=10, frame_range=(2, 8))
        assert m.frame_range == (2, 8)

    def test_single_frame_range_valid(self, tmp_path: Path):
        m = _make_manifest(tmp_path, frame_count=10, frame_range=(0, 1))
        assert m.frame_range == (0, 1)

    def test_full_range_valid(self, tmp_path: Path):
        m = _make_manifest(tmp_path, frame_count=10, frame_range=(0, 10))
        assert m.frame_range == (0, 10)
