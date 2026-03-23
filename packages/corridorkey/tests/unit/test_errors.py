"""Unit tests for corridorkey.errors."""

from __future__ import annotations

import pytest
from corridorkey.errors import (
    ClipScanError,
    CorridorKeyError,
    DeviceError,
    ExtractionError,
    FrameMismatchError,
    FrameReadError,
    InvalidStateTransitionError,
    JobCancelledError,
    ModelError,
    VRAMInsufficientError,
    WriteFailureError,
)


class TestErrorHierarchy:
    def test_all_errors_inherit_from_base(self):
        for cls in (
            ClipScanError,
            ExtractionError,
            FrameMismatchError,
            FrameReadError,
            WriteFailureError,
            VRAMInsufficientError,
            DeviceError,
            ModelError,
            InvalidStateTransitionError,
            JobCancelledError,
        ):
            assert issubclass(cls, CorridorKeyError)

    def test_base_is_exception(self):
        assert issubclass(CorridorKeyError, Exception)


class TestExtractionError:
    def test_message_contains_clip_name_and_detail(self):
        err = ExtractionError("MyClip", "ffmpeg failed")
        assert "MyClip" in str(err)
        assert "ffmpeg failed" in str(err)

    def test_attributes(self):
        err = ExtractionError("MyClip", "ffmpeg failed")
        assert err.clip_name == "MyClip"
        assert err.detail == "ffmpeg failed"

    def test_is_catchable_as_base(self):
        with pytest.raises(CorridorKeyError):
            raise ExtractionError("clip", "detail")


class TestFrameMismatchError:
    def test_message_contains_counts(self):
        err = FrameMismatchError("clip", 100, 99)
        assert "100" in str(err)
        assert "99" in str(err)
        assert "clip" in str(err)

    def test_attributes(self):
        err = FrameMismatchError("clip", 100, 99)
        assert err.clip_name == "clip"
        assert err.input_count == 100
        assert err.alpha_count == 99


class TestWriteFailureError:
    def test_message_contains_path(self):
        err = WriteFailureError("/some/path/frame.png")
        assert "/some/path/frame.png" in str(err)

    def test_path_attribute(self):
        err = WriteFailureError("/some/path/frame.png")
        assert err.path == "/some/path/frame.png"


class TestVRAMInsufficientError:
    def test_message_contains_values(self):
        err = VRAMInsufficientError(12.0, 8.5)
        assert "12.0" in str(err)
        assert "8.5" in str(err)

    def test_attributes(self):
        err = VRAMInsufficientError(12.0, 8.5)
        assert err.required_gb == pytest.approx(12.0)
        assert err.available_gb == pytest.approx(8.5)


class TestInvalidStateTransitionError:
    def test_message_contains_states(self):
        err = InvalidStateTransitionError("clip", "RAW", "COMPLETE")
        assert "RAW" in str(err)
        assert "COMPLETE" in str(err)
        assert "clip" in str(err)

    def test_attributes(self):
        err = InvalidStateTransitionError("clip", "RAW", "COMPLETE")
        assert err.clip_name == "clip"
        assert err.current_state == "RAW"
        assert err.target_state == "COMPLETE"


class TestJobCancelledError:
    def test_message_without_frame_index(self):
        err = JobCancelledError("clip")
        assert "clip" in str(err)

    def test_message_with_frame_index(self):
        err = JobCancelledError("clip", frame_index=42)
        assert "42" in str(err)
        assert "clip" in str(err)

    def test_attributes(self):
        err = JobCancelledError("clip", frame_index=5)
        assert err.clip_name == "clip"
        assert err.frame_index == 5

    def test_frame_index_none_by_default(self):
        err = JobCancelledError("clip")
        assert err.frame_index is None
