"""Unit tests for project.py.

project.py manages the on-disk project structure that all other components
read from. Bugs here corrupt project metadata or create unreadable folder
layouts. Tests cover sanitization, JSON atomic writes, display name
persistence, in/out range roundtrips, and the create_project/add_clips
workflows using tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey.models import InOutRange
from corridorkey.project import (
    add_clips_to_project,
    create_project,
    get_clip_dirs,
    get_display_name,
    is_image_file,
    is_v2_project,
    is_video_file,
    load_in_out_range,
    read_clip_json,
    read_project_json,
    sanitize_stem,
    save_in_out_range,
    set_display_name,
    write_clip_json,
    write_project_json,
)


class TestSanitizeStem:
    """sanitize_stem - filesystem-safe name derivation from raw filenames."""

    def test_strips_extension(self):
        """The file extension must be removed from the stem."""
        assert sanitize_stem("my_clip.mp4") == "my_clip"

    def test_replaces_spaces(self):
        """Spaces must be replaced with underscores."""
        assert sanitize_stem("my clip.mp4") == "my_clip"

    def test_collapses_underscores(self):
        """Consecutive underscores must be collapsed to a single underscore."""
        assert sanitize_stem("my__clip.mp4") == "my_clip"

    def test_truncates_to_max_len(self):
        """Names longer than max_len must be truncated."""
        long_name = "a" * 100 + ".mp4"
        result = sanitize_stem(long_name, max_len=60)
        assert len(result) <= 60

    def test_no_extension(self):
        """A name with no extension must be returned as-is (after sanitization)."""
        assert sanitize_stem("my_clip") == "my_clip"

    def test_special_chars_replaced(self):
        """Parentheses, hyphens, and other special chars must be replaced."""
        result = sanitize_stem("my-clip (1).mp4")
        assert " " not in result
        assert "(" not in result


class TestIsVideoFile:
    """is_video_file - extension-based video format detection."""

    def test_mp4(self):
        """mp4 files must be recognised as video."""
        assert is_video_file("clip.mp4")

    def test_mov(self):
        """MOV files (case-insensitive) must be recognised as video."""
        assert is_video_file("clip.MOV")

    def test_not_video(self):
        """Image and JSON files must not be recognised as video."""
        assert not is_video_file("frame.png")
        assert not is_video_file("data.json")


class TestIsImageFile:
    """is_image_file - extension-based image format detection."""

    def test_png(self):
        """png files must be recognised as image."""
        assert is_image_file("frame.png")

    def test_exr(self):
        """EXR files (case-insensitive) must be recognised as image."""
        assert is_image_file("frame.EXR")

    def test_not_image(self):
        """Video and JSON files must not be recognised as image."""
        assert not is_image_file("clip.mp4")
        assert not is_image_file("data.json")


class TestWriteReadProjectJson:
    """write_project_json / read_project_json - atomic JSON persistence."""

    def test_roundtrip(self, tmp_path: Path):
        """Data written by write_project_json must be returned unchanged by read_project_json."""
        data = {"version": 2, "display_name": "Test Project"}
        write_project_json(str(tmp_path), data)
        result = read_project_json(str(tmp_path))
        assert result == data

    def test_missing_returns_none(self, tmp_path: Path):
        """read_project_json must return None when no project.json exists."""
        assert read_project_json(str(tmp_path)) is None

    def test_corrupt_returns_none(self, tmp_path: Path):
        """read_project_json must return None when project.json contains invalid JSON."""
        (tmp_path / "project.json").write_text("not json")
        assert read_project_json(str(tmp_path)) is None


class TestWriteReadClipJson:
    """write_clip_json / read_clip_json - atomic JSON persistence for clip metadata."""

    def test_roundtrip(self, tmp_path: Path):
        """Data written by write_clip_json must be returned unchanged by read_clip_json."""
        data = {"source": {"filename": "clip.mp4"}}
        write_clip_json(str(tmp_path), data)
        result = read_clip_json(str(tmp_path))
        assert result == data

    def test_missing_returns_none(self, tmp_path: Path):
        """read_clip_json must return None when no clip.json exists."""
        assert read_clip_json(str(tmp_path)) is None

    def test_corrupt_returns_none(self, tmp_path: Path):
        """read_clip_json must return None when clip.json contains invalid JSON."""
        (tmp_path / "clip.json").write_text("{bad json")
        assert read_clip_json(str(tmp_path)) is None


class TestGetSetDisplayName:
    """get_display_name / set_display_name - human-readable name persistence."""

    def test_falls_back_to_folder_name(self, tmp_path: Path):
        """When no JSON exists, get_display_name must fall back to the folder name."""
        clip_dir = tmp_path / "MyClip"
        clip_dir.mkdir()
        assert get_display_name(str(clip_dir)) == "MyClip"

    def test_reads_from_clip_json(self, tmp_path: Path):
        """display_name stored in clip.json must take priority over the folder name."""
        write_clip_json(str(tmp_path), {"display_name": "Pretty Name"})
        assert get_display_name(str(tmp_path)) == "Pretty Name"

    def test_reads_from_project_json_fallback(self, tmp_path: Path):
        """display_name stored in project.json must be used when clip.json is absent."""
        write_project_json(str(tmp_path), {"display_name": "Project Name"})
        assert get_display_name(str(tmp_path)) == "Project Name"

    def test_set_writes_to_clip_json(self, tmp_path: Path):
        """set_display_name must persist the name into clip.json when it exists."""
        write_clip_json(str(tmp_path), {})
        set_display_name(str(tmp_path), "New Name")
        assert get_display_name(str(tmp_path)) == "New Name"

    def test_set_writes_to_project_json_when_no_clip_json(self, tmp_path: Path):
        """set_display_name must create project.json when no clip.json is present."""
        set_display_name(str(tmp_path), "Project Name")
        data = read_project_json(str(tmp_path))
        assert data is not None
        assert data["display_name"] == "Project Name"


class TestSaveLoadInOutRange:
    """save_in_out_range / load_in_out_range - in/out point persistence via clip.json."""

    def test_roundtrip_via_clip_json(self, tmp_path: Path):
        """An InOutRange saved and reloaded must have identical in_point and out_point."""
        write_clip_json(str(tmp_path), {})
        r = InOutRange(in_point=10, out_point=50)
        save_in_out_range(str(tmp_path), r)
        loaded = load_in_out_range(str(tmp_path))
        assert loaded is not None
        assert loaded.in_point == 10
        assert loaded.out_point == 50

    def test_clear_range(self, tmp_path: Path):
        """Saving None must remove the in/out range so load returns None."""
        write_clip_json(str(tmp_path), {})
        save_in_out_range(str(tmp_path), InOutRange(0, 10))
        save_in_out_range(str(tmp_path), None)
        assert load_in_out_range(str(tmp_path)) is None

    def test_missing_returns_none(self, tmp_path: Path):
        """load_in_out_range must return None when no clip.json exists."""
        assert load_in_out_range(str(tmp_path)) is None


class TestIsV2Project:
    """is_v2_project - v2 layout detection via presence of a clips/ subdirectory."""

    def test_v2_has_clips_subdir(self, tmp_path: Path):
        """A directory containing a clips/ subdirectory must be identified as v2."""
        (tmp_path / "clips").mkdir()
        assert is_v2_project(str(tmp_path))

    def test_v1_no_clips_subdir(self, tmp_path: Path):
        """A directory without a clips/ subdirectory must not be identified as v2."""
        assert not is_v2_project(str(tmp_path))


class TestGetClipDirs:
    """get_clip_dirs - enumerates per-clip directories for both v1 and v2 layouts."""

    def test_v2_returns_clip_subdirs(self, tmp_path: Path):
        """A v2 project must return each subdirectory inside clips/."""
        clips = tmp_path / "clips"
        (clips / "shot1").mkdir(parents=True)
        (clips / "shot2").mkdir(parents=True)
        result = get_clip_dirs(str(tmp_path))
        names = [Path(p).name for p in result]
        assert sorted(names) == ["shot1", "shot2"]

    def test_v1_returns_project_dir(self, tmp_path: Path):
        """A v1 project (no clips/ subdir) must return the project directory itself."""
        result = get_clip_dirs(str(tmp_path))
        assert result == [str(tmp_path)]

    def test_hidden_dirs_excluded(self, tmp_path: Path):
        """Hidden directories (dot-prefixed) inside clips/ must be excluded."""
        clips = tmp_path / "clips"
        (clips / "shot1").mkdir(parents=True)
        (clips / ".hidden").mkdir()
        result = get_clip_dirs(str(tmp_path))
        names = [Path(p).name for p in result]
        assert ".hidden" not in names


class TestCreateProject:
    """create_project - project folder scaffolding from source video paths."""

    def test_creates_project_folder(self, tmp_path: Path):
        """create_project must create the top-level project directory."""
        video = tmp_path / "source.mp4"
        video.touch()
        projects_dir = tmp_path / "Projects"
        project_dir = create_project(str(video), str(projects_dir), copy_source=False)
        assert Path(project_dir).is_dir()

    def test_creates_clips_subdir(self, tmp_path: Path):
        """create_project must create a clips/ subdirectory inside the project folder."""
        video = tmp_path / "source.mp4"
        video.touch()
        projects_dir = tmp_path / "Projects"
        project_dir = create_project(str(video), str(projects_dir), copy_source=False)
        assert (Path(project_dir) / "clips").is_dir()

    def test_creates_project_json(self, tmp_path: Path):
        """create_project must write a project.json with version=2."""
        video = tmp_path / "source.mp4"
        video.touch()
        projects_dir = tmp_path / "Projects"
        project_dir = create_project(str(video), str(projects_dir), copy_source=False)
        data = read_project_json(project_dir)
        assert data is not None
        assert data["version"] == 2

    def test_custom_display_name(self, tmp_path: Path):
        """A custom display_name must be written into project.json."""
        video = tmp_path / "source.mp4"
        video.touch()
        projects_dir = tmp_path / "Projects"
        project_dir = create_project(str(video), str(projects_dir), copy_source=False, display_name="My Project")
        data = read_project_json(project_dir)
        assert data is not None
        assert data["display_name"] == "My Project"

    def test_empty_paths_raises(self, tmp_path: Path):
        """Passing an empty paths list must raise ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            create_project([], str(tmp_path / "Projects"))

    def test_copy_source_copies_file(self, tmp_path: Path):
        """copy_source=True must copy the video file into the clip's Source/ directory."""
        video = tmp_path / "source.mp4"
        video.write_bytes(b"fake video data")
        projects_dir = tmp_path / "Projects"
        project_dir = create_project(str(video), str(projects_dir), copy_source=True)
        clips_dir = Path(project_dir) / "clips"
        clip_dirs = [d for d in clips_dir.iterdir() if d.is_dir()]
        assert len(clip_dirs) == 1
        source_dir = clip_dirs[0] / "Source"
        assert (source_dir / "source.mp4").exists()


class TestAddClipsToProject:
    """add_clips_to_project - appending new clips to an existing project."""

    def test_adds_clip_to_existing_project(self, tmp_path: Path):
        """A new clip path must result in a new clip directory inside clips/."""
        video1 = tmp_path / "clip1.mp4"
        video1.touch()
        video2 = tmp_path / "clip2.mp4"
        video2.touch()
        projects_dir = tmp_path / "Projects"

        project_dir = create_project(str(video1), str(projects_dir), copy_source=False)
        new_paths = add_clips_to_project(project_dir, [str(video2)], copy_source=False)

        assert len(new_paths) == 1
        assert Path(new_paths[0]).is_dir()

    def test_project_json_updated(self, tmp_path: Path):
        """project.json must list all clips after add_clips_to_project is called."""
        video1 = tmp_path / "clip1.mp4"
        video1.touch()
        video2 = tmp_path / "clip2.mp4"
        video2.touch()
        projects_dir = tmp_path / "Projects"

        project_dir = create_project(str(video1), str(projects_dir), copy_source=False)
        add_clips_to_project(project_dir, [str(video2)], copy_source=False)

        data = read_project_json(project_dir)
        assert data is not None
        assert len(data["clips"]) == 2
