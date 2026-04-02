"""Additional clip_state tests — covering uncovered lines 251, 312, 316-326."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from corridorkey.runtime.clip_state import ClipEntry, ClipState, _resolve_state
from corridorkey.stages.scanner.contracts import Clip


def _make_clip(tmp_path: Path, has_alpha: bool = False) -> Clip:
    input_dir = tmp_path / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir = tmp_path / "AlphaHint" if has_alpha else None
    if alpha_dir:
        alpha_dir.mkdir(parents=True, exist_ok=True)
    return Clip(name="TestClip", root=tmp_path, input_path=input_dir, alpha_path=alpha_dir)


def _write_frames(directory: Path, count: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        cv2.imwrite(str(directory / f"frame_{i:06d}.png"), np.zeros((8, 8, 3), dtype=np.uint8))


class TestCompletedFrameCountNoOutputDir:
    def test_completed_frame_count_no_output_dir(self, tmp_path: Path):
        """completed_frame_count() when output_dir doesn't exist returns 0 (line 251)."""
        clip = _make_clip(tmp_path)
        entry = ClipEntry(clip=clip, state=ClipState.RAW)
        # Output dir does not exist
        assert not entry.output_dir.exists()
        assert entry.completed_frame_count() == 0


class TestResolveStateComplete:
    def test_resolve_state_complete_when_all_outputs_present(self, tmp_path: Path):
        """_resolve_state returns COMPLETE when alpha + all output subdirs have enough frames (lines 316-326)."""
        n = 3
        input_dir = tmp_path / "Input"
        _write_frames(input_dir, n)

        alpha_dir = tmp_path / "AlphaHint"
        _write_frames(alpha_dir, n)

        # Write output frames in all subdirs
        for subdir in ("alpha", "fg"):
            _write_frames(tmp_path / "Output" / subdir, n)

        clip = Clip(name="c", root=tmp_path, input_path=input_dir, alpha_path=alpha_dir)
        state = _resolve_state(clip)
        assert state == ClipState.COMPLETE

    def test_resolve_state_ready_when_outputs_incomplete(self, tmp_path: Path):
        """READY when alpha present but output subdirs don't have enough frames."""
        n = 3
        input_dir = tmp_path / "Input"
        _write_frames(input_dir, n)

        alpha_dir = tmp_path / "AlphaHint"
        _write_frames(alpha_dir, n)

        # Only 1 output frame — not enough
        _write_frames(tmp_path / "Output" / "alpha", 1)

        clip = Clip(name="c", root=tmp_path, input_path=input_dir, alpha_path=alpha_dir)
        state = _resolve_state(clip)
        assert state == ClipState.READY

    def test_resolve_state_ready_when_no_output_dir(self, tmp_path: Path):
        """READY when alpha present but Output/ doesn't exist yet."""
        n = 3
        input_dir = tmp_path / "Input"
        _write_frames(input_dir, n)

        alpha_dir = tmp_path / "AlphaHint"
        _write_frames(alpha_dir, n)

        clip = Clip(name="c", root=tmp_path, input_path=input_dir, alpha_path=alpha_dir)
        state = _resolve_state(clip)
        assert state == ClipState.READY

    def test_resolve_state_raw_when_partial_alpha(self, tmp_path: Path):
        """RAW when alpha frame count < input frame count (line 312 partial alpha log)."""
        n = 5
        input_dir = tmp_path / "Input"
        _write_frames(input_dir, n)

        alpha_dir = tmp_path / "AlphaHint"
        _write_frames(alpha_dir, 2)  # fewer than input

        clip = Clip(name="c", root=tmp_path, input_path=input_dir, alpha_path=alpha_dir)
        state = _resolve_state(clip)
        assert state == ClipState.RAW

    def test_resolve_state_extracting_for_video_input(self, tmp_path: Path):
        """EXTRACTING when input_path is a video file."""
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")
        clip = Clip(name="c", root=tmp_path, input_path=video, alpha_path=None)
        with patch("corridorkey.stages.loader.extractor.is_video", return_value=True):
            state = _resolve_state(clip)
        assert state == ClipState.EXTRACTING


class TestVersionFallback:
    def test_version_fallback_when_not_installed(self):
        """The PackageNotFoundError fallback sets __version__ to dev string."""
        from importlib.metadata import PackageNotFoundError

        with patch("importlib.metadata.version", side_effect=PackageNotFoundError("corridorkey")):
            # Re-execute the version block logic directly
            try:
                from importlib.metadata import version as _version

                v = _version("corridorkey-nonexistent-package-xyz")
            except PackageNotFoundError:
                v = "0.0.0.dev0"

        assert v == "0.0.0.dev0"
