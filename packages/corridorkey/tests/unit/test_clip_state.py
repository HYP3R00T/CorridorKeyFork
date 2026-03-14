"""Unit tests for clip_state.py.

ClipEntry is the central data object passed through the entire pipeline.
Its state machine controls what operations are allowed at each stage, and
its asset discovery logic determines what the pipeline can process. Bugs
here affect every downstream consumer (service, pipeline, GUI, CLI).

Tests cover state transitions, asset discovery from real directory
structures, the processing lock, and the scan helpers. All tests use
tmp_path - no GPU or model files required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey.clip_state import ClipAsset, ClipEntry, ClipState, scan_clips_dir, scan_project_clips
from corridorkey.errors import ClipScanError, InvalidStateTransitionError


def _make_sequence(path: Path, count: int = 5) -> None:
    """Write count PNG stubs into path."""
    path.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (path / f"frame_{i:06d}.png").touch()


def _make_clip_dir(root: Path, with_alpha: bool = False, with_mask: bool = False) -> Path:
    """Create a minimal clip directory structure."""
    frames = root / "Frames"
    _make_sequence(frames)
    if with_alpha:
        _make_sequence(root / "AlphaHint")
    if with_mask:
        _make_sequence(root / "VideoMamaMaskHint")
    return root


class TestClipState:
    """Enum completeness - all expected states must be present."""

    def test_all_members_present(self):
        """All six lifecycle states must be defined - adding or removing one is a breaking change."""
        states = {s.value for s in ClipState}
        assert states == {"EXTRACTING", "RAW", "MASKED", "READY", "COMPLETE", "ERROR"}


class TestClipEntry:
    """ClipEntry state machine - valid transitions, error handling, and flags."""

    def test_transition_raw_to_ready(self, tmp_path: Path):
        """RAW -> READY is a valid forward transition."""
        clip = ClipEntry(name="shot1", root_path=str(tmp_path), state=ClipState.RAW)
        clip.transition_to(ClipState.READY)
        assert clip.state == ClipState.READY

    def test_transition_raw_to_complete_raises(self, tmp_path: Path):
        """RAW -> COMPLETE must be rejected - inference hasn't run yet."""
        clip = ClipEntry(name="shot1", root_path=str(tmp_path), state=ClipState.RAW)
        with pytest.raises(InvalidStateTransitionError):
            clip.transition_to(ClipState.COMPLETE)

    def test_set_error_sets_message(self, tmp_path: Path):
        """set_error must transition to ERROR and store the message."""
        clip = ClipEntry(name="shot1", root_path=str(tmp_path), state=ClipState.RAW)
        clip.set_error("something broke")
        assert clip.state == ClipState.ERROR
        assert clip.error_message == "something broke"

    def test_transition_clears_error_message(self, tmp_path: Path):
        """Transitioning out of ERROR must clear the stale error message."""
        clip = ClipEntry(name="shot1", root_path=str(tmp_path), state=ClipState.ERROR)
        clip.error_message = "old error"
        clip.transition_to(ClipState.RAW)
        assert clip.error_message is None

    def test_output_dir_property(self, tmp_path: Path):
        """output_dir must point to the Output subdirectory of root_path."""
        clip = ClipEntry(name="shot1", root_path=str(tmp_path))
        assert clip.output_dir == str(tmp_path / "Output")

    def test_has_outputs_false_when_empty(self, tmp_path: Path):
        """has_outputs must be False when no Output subdirectory exists."""
        clip = ClipEntry(name="shot1", root_path=str(tmp_path))
        assert not clip.has_outputs

    def test_has_outputs_true_when_fg_exists(self, tmp_path: Path):
        """has_outputs must be True when at least one FG frame exists."""
        fg_dir = tmp_path / "Output" / "FG"
        fg_dir.mkdir(parents=True)
        (fg_dir / "frame_000000.exr").touch()
        clip = ClipEntry(name="shot1", root_path=str(tmp_path))
        assert clip.has_outputs

    def test_processing_lock(self, tmp_path: Path):
        """is_processing must reflect the value set by set_processing."""
        clip = ClipEntry(name="shot1", root_path=str(tmp_path))
        assert not clip.is_processing
        clip.set_processing(True)
        assert clip.is_processing
        clip.set_processing(False)
        assert not clip.is_processing

    def test_error_to_raw_transition_allowed(self, tmp_path: Path):
        """ERROR -> RAW must be allowed so the user can retry after fixing the source."""
        clip = ClipEntry(name="shot1", root_path=str(tmp_path), state=ClipState.ERROR)
        clip.transition_to(ClipState.RAW)
        assert clip.state == ClipState.RAW

    def test_complete_to_ready_allowed(self, tmp_path: Path):
        """COMPLETE -> READY must be allowed to support reprocessing with new params."""
        clip = ClipEntry(name="shot1", root_path=str(tmp_path), state=ClipState.COMPLETE)
        clip.transition_to(ClipState.READY)
        assert clip.state == ClipState.READY


class TestClipAsset:
    """ClipAsset frame counting and file listing - used by the inference loop."""

    def test_sequence_frame_count(self, tmp_path: Path):
        """frame_count must equal the number of image files in the directory."""
        seq = tmp_path / "Frames"
        _make_sequence(seq, count=10)
        asset = ClipAsset(str(seq), "sequence")
        assert asset.frame_count == 10

    def test_sequence_get_frame_files_naturally_sorted(self, tmp_path: Path):
        """get_frame_files must return files in natural sort order, not lexicographic."""
        seq = tmp_path / "Frames"
        seq.mkdir()
        for name in ["frame_10.png", "frame_2.png", "frame_1.png"]:
            (seq / name).touch()
        asset = ClipAsset(str(seq), "sequence")
        files = asset.get_frame_files()
        assert files == ["frame_1.png", "frame_2.png", "frame_10.png"]

    def test_video_asset_get_frame_files_empty(self, tmp_path: Path):
        """Video assets have no frame files - the caller uses VideoCapture instead."""
        video = tmp_path / "clip.mp4"
        video.touch()
        asset = ClipAsset(str(video), "video")
        assert asset.get_frame_files() == []

    def test_missing_sequence_dir_frame_count_zero(self, tmp_path: Path):
        """A missing directory must return 0 frames rather than raising."""
        asset = ClipAsset(str(tmp_path / "nonexistent"), "sequence")
        assert asset.frame_count == 0


class TestFindAssets:
    """find_assets() - asset discovery and state inference from directory contents.

    find_assets() is called once per clip at scan time. It determines what
    the clip contains and sets the initial state. Getting this wrong means
    the pipeline either skips processable clips or tries to process ones
    that aren't ready.
    """

    def test_finds_frames_sequence(self, tmp_path: Path):
        """A Frames/ subdirectory must be detected as a sequence input asset."""
        clip_dir = tmp_path / "shot1"
        _make_clip_dir(clip_dir)
        clip = ClipEntry(name="shot1", root_path=str(clip_dir))
        clip.find_assets()
        assert clip.input_asset is not None
        assert clip.input_asset.asset_type == "sequence"

    def test_state_is_raw_without_alpha(self, tmp_path: Path):
        """Frames only, no AlphaHint -> RAW state (needs alpha generation)."""
        clip_dir = tmp_path / "shot1"
        _make_clip_dir(clip_dir)
        clip = ClipEntry(name="shot1", root_path=str(clip_dir))
        clip.find_assets()
        assert clip.state == ClipState.RAW

    def test_state_is_ready_with_alpha(self, tmp_path: Path):
        """Frames + AlphaHint -> READY state (can run inference immediately)."""
        clip_dir = tmp_path / "shot1"
        _make_clip_dir(clip_dir, with_alpha=True)
        clip = ClipEntry(name="shot1", root_path=str(clip_dir))
        clip.find_assets()
        assert clip.state == ClipState.READY

    def test_state_is_masked_with_mask_no_alpha(self, tmp_path: Path):
        """Frames + VideoMamaMaskHint but no AlphaHint -> MASKED (needs alpha gen from mask)."""
        clip_dir = tmp_path / "shot1"
        _make_clip_dir(clip_dir, with_mask=True)
        clip = ClipEntry(name="shot1", root_path=str(clip_dir))
        clip.find_assets()
        assert clip.state == ClipState.MASKED

    def test_no_input_raises(self, tmp_path: Path):
        """An empty clip directory must raise ClipScanError, not return a broken clip."""
        clip_dir = tmp_path / "empty"
        clip_dir.mkdir()
        clip = ClipEntry(name="empty", root_path=str(clip_dir))
        with pytest.raises(ClipScanError):
            clip.find_assets()

    def test_loads_in_out_range_from_clip_json(self, tmp_path: Path):
        """An in/out range stored in clip.json must be loaded into clip.in_out_range."""
        from corridorkey.project import write_clip_json

        clip_dir = tmp_path / "shot1"
        _make_clip_dir(clip_dir)
        write_clip_json(str(clip_dir), {"in_out_range": {"in_point": 5, "out_point": 20}})
        clip = ClipEntry(name="shot1", root_path=str(clip_dir))
        clip.find_assets()
        assert clip.in_out_range is not None
        assert clip.in_out_range.in_point == 5
        assert clip.in_out_range.out_point == 20

    def test_state_complete_when_all_outputs_present(self, tmp_path: Path):
        """A clip with a manifest and matching output frames must be detected as COMPLETE."""
        clip_dir = tmp_path / "shot1"
        _make_clip_dir(clip_dir, with_alpha=True)

        import json

        out_dir = clip_dir / "Output"
        for subdir in ("FG", "Matte"):
            d = out_dir / subdir
            d.mkdir(parents=True)
            for i in range(5):
                (d / f"frame_{i:06d}.exr").touch()
        manifest = {"version": 1, "enabled_outputs": ["fg", "matte"], "formats": {}, "params": {}}
        (out_dir / ".corridorkey_manifest.json").write_text(json.dumps(manifest))

        clip = ClipEntry(name="shot1", root_path=str(clip_dir))
        clip.find_assets()
        assert clip.state == ClipState.COMPLETE


class TestScanClipsDir:
    """scan_clips_dir() - directory scanning and v2 project layout detection."""

    def test_scans_subdirectories(self, tmp_path: Path):
        """Each subdirectory with a Frames/ folder must produce one ClipEntry."""
        for name in ("shot1", "shot2"):
            _make_clip_dir(tmp_path / name)
        clips = scan_clips_dir(str(tmp_path))
        assert len(clips) == 2

    def test_skips_hidden_dirs(self, tmp_path: Path):
        """Directories starting with '.' must be ignored."""
        _make_clip_dir(tmp_path / "shot1")
        (tmp_path / ".hidden").mkdir()
        clips = scan_clips_dir(str(tmp_path))
        assert len(clips) == 1

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        """A non-existent directory must return an empty list, not raise."""
        clips = scan_clips_dir(str(tmp_path / "nonexistent"))
        assert clips == []

    def test_v2_project_scanned_correctly(self, tmp_path: Path):
        """A v2 project with a clips/ subdirectory must be unwrapped correctly."""
        clips_dir = tmp_path / "clips"
        _make_clip_dir(clips_dir / "shot1")
        clips = scan_clips_dir(str(tmp_path))
        assert len(clips) == 1
        assert clips[0].name == "shot1"


class TestScanProjectClips:
    """scan_project_clips() - v1 and v2 project layout compatibility."""

    def test_v2_project(self, tmp_path: Path):
        """v2 projects must return one ClipEntry per clips/ subdirectory."""
        clips_dir = tmp_path / "clips"
        _make_clip_dir(clips_dir / "shot1")
        _make_clip_dir(clips_dir / "shot2")
        clips = scan_project_clips(str(tmp_path))
        assert len(clips) == 2

    def test_v1_project_single_clip(self, tmp_path: Path):
        """v1 projects (no clips/ subdir) must return the project dir itself as one clip."""
        _make_clip_dir(tmp_path)
        clips = scan_project_clips(str(tmp_path))
        assert len(clips) == 1
