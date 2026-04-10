"""Unit tests for corridorkey.runtime.clip_state."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from corridorkey.errors import InvalidStateTransitionError
from corridorkey.runtime.clip_state import ClipEntry, ClipState, InOutRange
from corridorkey.stages.scanner.contracts import Clip

# Helpers


def _make_clip(tmp_path: Path, has_alpha: bool = False) -> Clip:
    input_dir = tmp_path / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir = tmp_path / "AlphaHint" if has_alpha else None
    if alpha_dir:
        alpha_dir.mkdir(parents=True, exist_ok=True)
    return Clip(name="TestClip", root=tmp_path, input_path=input_dir, alpha_path=alpha_dir)


def _make_entry(tmp_path: Path, state: ClipState = ClipState.RAW) -> ClipEntry:
    clip = _make_clip(tmp_path)
    entry = ClipEntry(clip=clip, state=state)
    return entry


# InOutRange


class TestInOutRange:
    def test_frame_count(self):
        r = InOutRange(in_point=0, out_point=9)
        assert r.frame_count == 10

    def test_contains_in_range(self):
        r = InOutRange(in_point=5, out_point=10)
        assert r.contains(5)
        assert r.contains(7)
        assert r.contains(10)

    def test_contains_out_of_range(self):
        r = InOutRange(in_point=5, out_point=10)
        assert not r.contains(4)
        assert not r.contains(11)

    def test_to_frame_range(self):
        r = InOutRange(in_point=2, out_point=7)
        assert r.to_frame_range() == (2, 8)


# ClipEntry properties


class TestClipEntryProperties:
    def test_name(self, tmp_path):
        entry = _make_entry(tmp_path)
        assert entry.name == "TestClip"

    def test_root(self, tmp_path):
        entry = _make_entry(tmp_path)
        assert entry.root == tmp_path

    def test_output_dir(self, tmp_path):
        entry = _make_entry(tmp_path)
        assert entry.output_dir == tmp_path / "Output"

    def test_is_processing_default_false(self, tmp_path):
        entry = _make_entry(tmp_path)
        assert entry.is_processing is False

    def test_set_processing(self, tmp_path):
        entry = _make_entry(tmp_path)
        entry.set_processing(True)
        assert entry.is_processing is True
        entry.set_processing(False)
        assert entry.is_processing is False


# State transitions


class TestTransitions:
    def test_raw_to_ready(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.RAW)
        entry.transition_to(ClipState.READY)
        assert entry.state == ClipState.READY

    def test_raw_to_error(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.RAW)
        entry.transition_to(ClipState.ERROR)
        assert entry.state == ClipState.ERROR

    def test_ready_to_complete(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.READY)
        entry.transition_to(ClipState.COMPLETE)
        assert entry.state == ClipState.COMPLETE

    def test_ready_to_error(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.READY)
        entry.transition_to(ClipState.ERROR)
        assert entry.state == ClipState.ERROR

    def test_complete_to_ready(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.COMPLETE)
        entry.transition_to(ClipState.READY)
        assert entry.state == ClipState.READY

    def test_error_to_raw(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.ERROR)
        entry.transition_to(ClipState.RAW)
        assert entry.state == ClipState.RAW

    def test_error_to_ready(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.ERROR)
        entry.transition_to(ClipState.READY)
        assert entry.state == ClipState.READY

    def test_error_to_extracting(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.ERROR)
        entry.transition_to(ClipState.EXTRACTING)
        assert entry.state == ClipState.EXTRACTING

    def test_extracting_to_raw(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.EXTRACTING)
        entry.transition_to(ClipState.RAW)
        assert entry.state == ClipState.RAW

    def test_invalid_transition_raises(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.RAW)
        with pytest.raises(InvalidStateTransitionError):
            entry.transition_to(ClipState.COMPLETE)

    def test_invalid_transition_raw_to_extracting_raises(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.RAW)
        with pytest.raises(InvalidStateTransitionError):
            entry.transition_to(ClipState.EXTRACTING)

    def test_error_message_cleared_on_non_error_transition(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.RAW)
        entry.set_error("something broke")
        assert entry.error_message == "something broke"
        entry.transition_to(ClipState.READY)
        assert entry.error_message is None

    def test_set_error_sets_message_and_state(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.RAW)
        entry.set_error("boom")
        assert entry.state == ClipState.ERROR
        assert entry.error_message == "boom"


# has_outputs / completed_stems


class TestOutputInspection:
    def test_has_outputs_false_when_no_output_dir(self, tmp_path):
        entry = _make_entry(tmp_path)
        assert entry.has_outputs is False

    def test_has_outputs_true_when_alpha_written(self, tmp_path):
        alpha_dir = tmp_path / "Output" / "alpha"
        alpha_dir.mkdir(parents=True)
        (alpha_dir / "frame_000001.png").write_bytes(b"")
        entry = _make_entry(tmp_path)
        assert entry.has_outputs is True

    def test_completed_stems_empty_when_no_output(self, tmp_path):
        entry = _make_entry(tmp_path)
        assert entry.completed_stems() == set()

    def test_completed_stems_intersection(self, tmp_path):
        alpha_dir = tmp_path / "Output" / "alpha"
        fg_dir = tmp_path / "Output" / "fg"
        alpha_dir.mkdir(parents=True)
        fg_dir.mkdir(parents=True)
        (alpha_dir / "frame_000001.png").write_bytes(b"")
        (alpha_dir / "frame_000002.png").write_bytes(b"")
        (fg_dir / "frame_000001.png").write_bytes(b"")
        # frame_000002 missing from fg → not in intersection
        entry = _make_entry(tmp_path)
        stems = entry.completed_stems()
        assert "frame_000001" in stems
        assert "frame_000002" not in stems

    def test_completed_frame_count(self, tmp_path):
        alpha_dir = tmp_path / "Output" / "alpha"
        alpha_dir.mkdir(parents=True)
        for i in range(3):
            (alpha_dir / f"frame_{i:06d}.png").write_bytes(b"")
        entry = _make_entry(tmp_path)
        assert entry.completed_frame_count() == 3


# refresh_state


class TestRefreshState:
    def test_refresh_state_skipped_when_processing(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.READY)
        entry.set_processing(True)
        entry.refresh_state()
        # State should not change while processing
        assert entry.state == ClipState.READY

    def test_refresh_state_updates_from_disk(self, tmp_path):
        entry = _make_entry(tmp_path, ClipState.COMPLETE)
        # Patch _resolve_state to return RAW
        with patch("corridorkey.runtime.clip_state._resolve_state", return_value=ClipState.RAW):
            entry.refresh_state()
        assert entry.state == ClipState.RAW


# from_clip


class TestFromClip:
    def test_from_clip_resolves_raw_for_image_sequence(self, tmp_path):
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        import cv2
        import numpy as np

        cv2.imwrite(str(input_dir / "frame_000001.png"), np.zeros((8, 8, 3), dtype=np.uint8))
        clip = Clip(name="c", root=tmp_path, input_path=input_dir, alpha_path=None)
        entry = ClipEntry.from_clip(clip)
        assert entry.state == ClipState.RAW

    def test_from_clip_resolves_extracting_for_video(self, tmp_path):
        video_path = tmp_path / "input.mp4"
        video_path.write_bytes(b"")
        clip = Clip(name="c", root=tmp_path, input_path=video_path, alpha_path=None)
        with patch("corridorkey.stages.loader.extractor.is_video", return_value=True):
            entry = ClipEntry.from_clip(clip)
        assert entry.state == ClipState.EXTRACTING


class TestClipEntryWarningsAndManifest:
    def test_warnings_empty_by_default(self, tmp_path):
        entry = _make_entry(tmp_path)
        assert entry.warnings == []

    def test_warnings_can_be_appended(self, tmp_path):
        entry = _make_entry(tmp_path)
        entry.warnings.append("partial alpha detected")
        assert len(entry.warnings) == 1
        assert "partial alpha" in entry.warnings[0]

    def test_manifest_none_by_default(self, tmp_path):
        entry = _make_entry(tmp_path)
        assert entry.manifest is None

    def test_manifest_can_be_set(self, tmp_path):
        entry = _make_entry(tmp_path)
        fake_manifest = object()
        entry.manifest = fake_manifest
        assert entry.manifest is fake_manifest

    def test_error_message_none_by_default(self, tmp_path):
        entry = _make_entry(tmp_path)
        assert entry.error_message is None
