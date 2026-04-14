"""Unit tests for corridorkey.stages.scanner.orchestrator — scan()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from corridorkey.errors import ClipScanError
from corridorkey.events import PipelineEvents
from corridorkey.stages.scanner import ScanResult, scan


def _make_clip_dir(root: Path, name: str = "my_clip", with_alpha: bool = True) -> Path:
    clip = root / name
    (clip / "Input").mkdir(parents=True)
    if with_alpha:
        (clip / "AlphaHint").mkdir()
    return clip


class TestScanReturnType:
    def test_returns_scan_result(self, tmp_path: Path):
        result = scan(tmp_path)
        assert isinstance(result, ScanResult)

    def test_empty_dir_returns_empty_result(self, tmp_path: Path):
        result = scan(tmp_path)
        assert result.clip_count == 0
        assert result.skipped_count == 0


class TestScanClipDiscovery:
    def test_finds_single_clip(self, tmp_path: Path):
        _make_clip_dir(tmp_path)
        result = scan(tmp_path)
        assert result.clip_count == 1
        assert result.clips[0].name == "my_clip"

    def test_finds_multiple_clips(self, tmp_path: Path):
        for name in ("clip_a", "clip_b", "clip_c"):
            (tmp_path / name / "Input").mkdir(parents=True)
        assert scan(tmp_path).clip_count == 3

    def test_clips_sorted_by_name(self, tmp_path: Path):
        for name in ("clip_c", "clip_a", "clip_b"):
            (tmp_path / name / "Input").mkdir(parents=True)
        names = [c.name for c in scan(tmp_path).clips]
        assert names == sorted(names)

    def test_clip_without_alpha_has_none_alpha_path(self, tmp_path: Path):
        _make_clip_dir(tmp_path, with_alpha=False)
        assert scan(tmp_path).clips[0].alpha_path is None

    def test_clip_with_alpha_has_alpha_path(self, tmp_path: Path):
        _make_clip_dir(tmp_path, with_alpha=True)
        assert scan(tmp_path).clips[0].alpha_path is not None

    def test_directory_is_itself_a_clip(self, tmp_path: Path):
        (tmp_path / "Input").mkdir()
        assert scan(tmp_path).clip_count == 1

    def test_case_insensitive_input_folder(self, tmp_path: Path):
        (tmp_path / "my_clip" / "input").mkdir(parents=True)
        assert scan(tmp_path).clip_count == 1

    def test_case_insensitive_alphahint_folder(self, tmp_path: Path):
        clip = tmp_path / "my_clip"
        (clip / "Input").mkdir(parents=True)
        (clip / "alphahint").mkdir()
        assert scan(tmp_path).clips[0].alpha_path is not None

    def test_skips_non_clip_subdirs_silently(self, tmp_path: Path):
        (tmp_path / "not_a_clip").mkdir()
        result = scan(tmp_path)
        assert result.clip_count == 0
        assert result.skipped_count == 0  # no Input/ folder = not a clip, not an error

    def test_video_inside_input_dir_used_as_input_path(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()
        clips = scan(tmp_path).clips
        assert clips[0].input_path == video

    def test_video_inside_alphahint_dir_used_as_alpha_path(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        (clip_dir / "Input").mkdir(parents=True)
        alpha_dir = clip_dir / "AlphaHint"
        alpha_dir.mkdir()
        video = alpha_dir / "alpha.mp4"
        video.touch()
        clips = scan(tmp_path).clips
        assert clips[0].alpha_path == video


class TestScanErrors:
    def test_nonexistent_path_raises(self, tmp_path: Path):
        with pytest.raises(ClipScanError, match="does not exist"):
            scan(tmp_path / "ghost")

    def test_unrecognised_file_extension_raises(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.touch()
        with pytest.raises(ClipScanError, match="not a recognised video format"):
            scan(f)


class TestScanVideoFile:
    def test_video_file_reorganised(self, tmp_path: Path):
        (tmp_path / "clip.mp4").touch()
        result = scan(tmp_path / "clip.mp4")
        assert result.clip_count == 1
        assert (tmp_path / "Input" / "clip.mp4").exists()

    def test_loose_video_in_dir_reorganised(self, tmp_path: Path):
        (tmp_path / "clip.mp4").touch()
        result = scan(tmp_path)
        assert result.clip_count == 1
        assert (tmp_path / "Input" / "clip.mp4").exists()

    def test_reorganise_idempotent(self, tmp_path: Path):
        """Scanning the same video twice should not fail or duplicate."""
        (tmp_path / "clip.mp4").touch()
        scan(tmp_path / "clip.mp4")
        result = scan(tmp_path / "Input" / "clip.mp4")
        assert result.clip_count == 1


class TestScanAmbiguousVideos:
    def test_multiple_videos_in_input_reported_as_skipped(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        (input_dir / "a.mp4").touch()
        (input_dir / "b.mp4").touch()
        result = scan(tmp_path)
        assert result.clip_count == 0
        assert result.skipped_count == 1
        assert "multiple" in result.skipped[0].reason.lower()


class TestScanEvents:
    def test_on_clip_found_fires_for_each_clip(self, tmp_path: Path):
        for name in ("clip_a", "clip_b"):
            (tmp_path / name / "Input").mkdir(parents=True)

        found: list[str] = []
        events = PipelineEvents(on_clip_found=lambda name, root: found.append(name))
        scan(tmp_path, events=events)
        assert sorted(found) == ["clip_a", "clip_b"]

    def test_on_clip_skipped_fires_for_skipped_paths(self, tmp_path: Path):
        clip_dir = tmp_path / "bad_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        (input_dir / "a.mp4").touch()
        (input_dir / "b.mp4").touch()

        skipped: list[str] = []
        events = PipelineEvents(on_clip_skipped=lambda reason, path: skipped.append(reason))
        scan(tmp_path, events=events)
        assert len(skipped) == 1

    def test_events_none_does_not_raise(self, tmp_path: Path):
        _make_clip_dir(tmp_path)
        scan(tmp_path, events=None)  # should not raise


class TestScanPermissionErrors:
    def test_permission_error_on_iterdir_raises(self, tmp_path: Path):
        """PermissionError on top-level iterdir must propagate as ClipScanError."""
        with (
            patch(
                "corridorkey.stages.scanner.orchestrator.try_build_clip",
                return_value=(None, None),
            ),
            patch.object(Path, "iterdir", side_effect=PermissionError("denied")),
            pytest.raises(ClipScanError, match="Cannot read directory"),
        ):
            scan(tmp_path)


class TestScanEventsExtended:
    def test_clip_found_event_fires_for_single_clip_folder(self, tmp_path: Path):
        """When path is itself a clip folder, on_clip_found must fire."""
        (tmp_path / "Input").mkdir()
        found: list[str] = []
        events = PipelineEvents(on_clip_found=lambda name, root: found.append(name))
        scan(tmp_path, events=events)
        assert len(found) == 1

    def test_clip_skipped_event_fires_for_single_clip_folder_skip(self, tmp_path: Path):
        """When path is itself a clip folder but ambiguous, on_clip_skipped must fire."""
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        (input_dir / "a.mp4").touch()
        (input_dir / "b.mp4").touch()
        skipped: list[str] = []
        events = PipelineEvents(on_clip_skipped=lambda reason, path: skipped.append(reason))
        scan(tmp_path, events=events)
        assert len(skipped) == 1

    def test_clip_skipped_event_fires_in_directory_loop(self, tmp_path: Path):
        """Ambiguous clip inside a clips directory fires on_clip_skipped."""
        clip_dir = tmp_path / "bad_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        (input_dir / "a.mp4").touch()
        (input_dir / "b.mp4").touch()
        skipped: list[str] = []
        events = PipelineEvents(on_clip_skipped=lambda reason, path: skipped.append(reason))
        scan(tmp_path, events=events)
        assert len(skipped) == 1

    def test_warning_logged_when_no_clips_found(self, tmp_path: Path, caplog):
        """Empty directory with no clips or skipped paths logs a warning."""
        import logging

        with caplog.at_level(logging.WARNING, logger="corridorkey.stages.scanner.orchestrator"):
            scan(tmp_path)
        assert any("No clips found" in r.message for r in caplog.records)


class TestScanDirectoryLoopEvents:
    def test_clip_found_event_fires_in_directory_loop(self, tmp_path: Path):
        """on_clip_found fires for clips discovered in the directory loop."""
        _make_clip_dir(tmp_path, name="clip_a")
        _make_clip_dir(tmp_path, name="clip_b")
        found: list[str] = []
        events = PipelineEvents(on_clip_found=lambda name, root: found.append(name))
        scan(tmp_path, events=events)
        assert sorted(found) == ["clip_a", "clip_b"]

    def test_clip_skipped_event_fires_in_directory_loop(self, tmp_path: Path):
        """on_clip_skipped fires for ambiguous clips in the directory loop."""
        clip_dir = tmp_path / "bad_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        (input_dir / "a.mp4").touch()
        (input_dir / "b.mp4").touch()
        skipped_reasons: list[str] = []
        events = PipelineEvents(on_clip_skipped=lambda reason, path: skipped_reasons.append(reason))
        scan(tmp_path, events=events)
        assert len(skipped_reasons) == 1

    def test_loose_video_reorganised_event_fires_in_directory_loop(self, tmp_path: Path):
        """on_clip_found fires when a loose video in a directory is reorganised."""
        (tmp_path / "clip.mp4").touch()
        found: list[str] = []
        events = PipelineEvents(on_clip_found=lambda name, root: found.append(name))
        scan(tmp_path, events=events)
        assert len(found) == 1


class TestScanVideoFileWithEvents:
    def test_clip_found_fires_when_scanning_video_file_directly(self, tmp_path: Path):
        """on_clip_found fires when a video file is passed directly and reorganised."""
        video = tmp_path / "clip.mp4"
        video.touch()
        found: list[str] = []
        events = PipelineEvents(on_clip_found=lambda name, root: found.append(name))
        scan(video, events=events)
        assert len(found) == 1


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

    def test_string_path_accepted(self, tmp_path: Path):
        _make_clip_dir(tmp_path)
        result = scan(str(tmp_path))
        assert result.clip_count == 1

    def test_empty_list_returns_empty_result(self, tmp_path: Path):
        result = scan([])
        assert result.clip_count == 0
        assert result.skipped_count == 0
