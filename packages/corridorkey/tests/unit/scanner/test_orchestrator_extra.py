"""Cover scanner orchestrator lines 71, 102, 117 — events in directory loop."""

from __future__ import annotations

from pathlib import Path

from corridorkey.events import PipelineEvents
from corridorkey.stages.scanner import scan


class TestScanDirectoryLoopVideoEvents:
    def test_clip_found_fires_for_reorganised_video_in_dir_loop(self, tmp_path: Path):
        """Line 71: on_clip_found fires when a loose video in a dir loop is reorganised."""
        (tmp_path / "clip.mp4").touch()
        found: list[str] = []
        events = PipelineEvents(on_clip_found=lambda name, root: found.append(name))
        scan(tmp_path, reorganise=True, events=events)
        assert len(found) == 1

    def test_clip_skipped_fires_for_loose_video_reorganise_false_in_dir_loop(self, tmp_path: Path):
        """Line 102: on_clip_skipped fires for loose video when reorganise=False in dir loop."""
        (tmp_path / "clip.mp4").touch()
        skipped: list[str] = []
        events = PipelineEvents(on_clip_skipped=lambda reason, path: skipped.append(reason))
        scan(tmp_path, reorganise=False, events=events)
        assert len(skipped) == 1
        assert "reorganise" in skipped[0]

    def test_clip_skipped_fires_for_ambiguous_clip_in_dir_loop(self, tmp_path: Path):
        """Line 117: on_clip_skipped fires for ambiguous clip (multiple videos) in dir loop."""
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        (input_dir / "a.mp4").touch()
        (input_dir / "b.mp4").touch()
        skipped: list[str] = []
        events = PipelineEvents(on_clip_skipped=lambda reason, path: skipped.append(reason))
        scan(tmp_path, events=events)
        assert len(skipped) == 1
