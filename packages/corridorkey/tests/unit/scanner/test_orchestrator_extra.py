"""Cover scanner orchestrator branches not hit by the main test file."""

from __future__ import annotations

from pathlib import Path

from corridorkey.events import PipelineEvents
from corridorkey.stages.scanner import scan


class TestScanDirectoryLoopVideoEvents:
    def test_clip_found_fires_for_reorganised_video_in_dir_loop(self, tmp_path: Path):
        """on_clip_found fires when a loose video in a dir loop is reorganised."""
        (tmp_path / "clip.mp4").touch()
        found: list[str] = []
        events = PipelineEvents(on_clip_found=lambda name, root: found.append(name))
        scan(tmp_path, reorganise=True, events=events)
        assert len(found) == 1

    def test_clip_skipped_fires_for_loose_video_reorganise_false_in_dir_loop(self, tmp_path: Path):
        """on_clip_skipped fires for loose video when reorganise=False in dir loop."""
        (tmp_path / "clip.mp4").touch()
        skipped: list[str] = []
        events = PipelineEvents(on_clip_skipped=lambda reason, path: skipped.append(reason))
        scan(tmp_path, reorganise=False, events=events)
        assert len(skipped) == 1
        assert "reorganise" in skipped[0]

    def test_clip_skipped_fires_for_ambiguous_clip_in_dir_loop(self, tmp_path: Path):
        """on_clip_skipped fires for ambiguous clip (multiple videos) in dir loop."""
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        (input_dir / "a.mp4").touch()
        (input_dir / "b.mp4").touch()
        skipped: list[str] = []
        events = PipelineEvents(on_clip_skipped=lambda reason, path: skipped.append(reason))
        scan(tmp_path, events=events)
        assert len(skipped) == 1


class TestScanVideoFileWithEvents:
    def test_clip_found_fires_when_scanning_video_file_directly(self, tmp_path: Path):
        """on_clip_found fires when a video file is passed directly and reorganised."""
        video = tmp_path / "clip.mp4"
        video.touch()
        found: list[str] = []
        events = PipelineEvents(on_clip_found=lambda name, root: found.append(name))
        scan(video, reorganise=True, events=events)
        assert len(found) == 1

    def test_clip_skipped_fires_when_scanning_video_file_directly_no_reorganise(self, tmp_path: Path):
        """on_clip_skipped fires when a video file is passed directly with reorganise=False."""
        video = tmp_path / "clip.mp4"
        video.touch()
        skipped_paths: list[Path] = []
        events = PipelineEvents(on_clip_skipped=lambda reason, path: skipped_paths.append(path))
        scan(video, reorganise=False, events=events)
        assert len(skipped_paths) == 1
        assert skipped_paths[0] == video


class TestScanDirectoryLoopNonVideoFiles:
    def test_non_video_file_in_directory_is_silently_ignored(self, tmp_path: Path):
        """A non-video file (e.g. .txt) inside a clips directory is skipped silently."""
        (tmp_path / "readme.txt").touch()
        result = scan(tmp_path)
        assert result.clip_count == 0
        assert result.skipped_count == 0

    def test_non_video_file_alongside_clip_does_not_affect_result(self, tmp_path: Path):
        """A .txt file next to a valid clip folder does not appear in skipped."""
        (tmp_path / "notes.txt").touch()
        (tmp_path / "my_clip" / "Input").mkdir(parents=True)
        result = scan(tmp_path)
        assert result.clip_count == 1
        assert result.skipped_count == 0


class TestScanDirectoryLoopNoneNoneFolder:
    def test_plain_subdir_with_no_input_folder_is_silently_ignored(self, tmp_path: Path):
        """A subdirectory with no Input/ folder returns (None, None) from try_build_clip
        and must not appear in clips or skipped."""
        (tmp_path / "not_a_clip").mkdir()
        result = scan(tmp_path)
        assert result.clip_count == 0
        assert result.skipped_count == 0

    def test_plain_subdir_does_not_fire_any_events(self, tmp_path: Path):
        """No events fire for a subdirectory that is not a clip."""
        (tmp_path / "not_a_clip").mkdir()
        found: list[str] = []
        skipped: list[str] = []
        events = PipelineEvents(
            on_clip_found=lambda name, root: found.append(name),
            on_clip_skipped=lambda reason, path: skipped.append(reason),
        )
        scan(tmp_path, events=events)
        assert found == []
        assert skipped == []
