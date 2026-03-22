"""Unit tests for corridorkey_new.stages.loader.contracts — ClipManifest."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey_new.stages.loader.contracts import ClipManifest


def _make_manifest(tmp_path: Path, **overrides: object) -> ClipManifest:
    frames = tmp_path / "Frames"
    frames.mkdir(exist_ok=True)
    output = tmp_path / "Output"
    output.mkdir(exist_ok=True)
    return ClipManifest(
        clip_name=str(overrides.get("clip_name", "test")),
        clip_root=overrides.get("clip_root", tmp_path),  # type: ignore[arg-type]
        frames_dir=overrides.get("frames_dir", frames),  # type: ignore[arg-type]
        alpha_frames_dir=overrides.get("alpha_frames_dir"),  # type: ignore[arg-type]
        output_dir=overrides.get("output_dir", output),  # type: ignore[arg-type]
        needs_alpha=bool(overrides.get("needs_alpha", False)),
        frame_count=int(overrides.get("frame_count", 10)),  # type: ignore[arg-type]
        frame_range=overrides.get("frame_range", (0, 10)),  # type: ignore[arg-type]
        is_linear=bool(overrides.get("is_linear", False)),
        video_meta_path=overrides.get("video_meta_path"),  # type: ignore[arg-type]
    )


class TestClipManifest:
    def test_valid_manifest(self, tmp_path: Path):
        m = _make_manifest(tmp_path)
        assert m.clip_name == "test"
        assert m.video_meta_path is None

    def test_frame_range_start_negative_raises(self, tmp_path: Path):
        with pytest.raises(Exception, match="frame_range start"):
            _make_manifest(tmp_path, frame_range=(-1, 10))

    def test_frame_range_end_exceeds_count_raises(self, tmp_path: Path):
        with pytest.raises(Exception, match="frame_range end"):
            _make_manifest(tmp_path, frame_range=(0, 11))

    def test_frame_range_start_gte_end_raises(self, tmp_path: Path):
        with pytest.raises(Exception, match="must be less than end"):
            _make_manifest(tmp_path, frame_range=(5, 5))

    def test_model_dump_json_roundtrip(self, tmp_path: Path):
        m = _make_manifest(tmp_path, frame_count=5, frame_range=(0, 5))
        restored = ClipManifest.model_validate_json(m.model_dump_json())
        assert restored.clip_name == m.clip_name
        assert restored.frame_count == m.frame_count
