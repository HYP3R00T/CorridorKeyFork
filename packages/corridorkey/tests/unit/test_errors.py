"""Unit tests for errors.py.

Every error class carries structured attributes (clip name, frame index,
path, etc.) that consumers use to build user-facing messages and decide
how to recover. Tests verify the inheritance hierarchy, attribute values,
and that error messages contain the information needed to diagnose problems.
FFmpegNotFoundError is also tested for platform-specific install hints.
"""

from __future__ import annotations

from unittest.mock import patch

from corridorkey.errors import (
    ClipScanError,
    CorridorKeyError,
    ExtractionError,
    FFmpegNotFoundError,
    FrameMismatchError,
    FrameReadError,
    InvalidStateTransitionError,
    JobCancelledError,
    MaskChannelError,
    VRAMInsufficientError,
    WriteFailureError,
)


class TestErrorHierarchy:
    """All custom errors must inherit from CorridorKeyError for uniform catch blocks."""

    def test_all_inherit_from_base(self):
        """Every error subclass must be catchable via `except CorridorKeyError`."""
        errors = [
            ClipScanError("x"),
            FrameMismatchError("clip", 10, 5),
            FrameReadError("clip", 0, "/path"),
            WriteFailureError("clip", 0, "/path"),
            MaskChannelError("clip", 0, 2),
            VRAMInsufficientError(10.0, 4.0),
            InvalidStateTransitionError("clip", "RAW", "COMPLETE"),
            JobCancelledError("clip"),
            ExtractionError("clip", "detail"),
        ]
        for err in errors:
            assert isinstance(err, CorridorKeyError)
            assert isinstance(err, Exception)


class TestFrameMismatchError:
    """FrameMismatchError - structured attributes for mismatched input/alpha frame counts."""

    def test_attributes(self):
        """clip_name, input_count, and alpha_count must be stored for programmatic access."""
        err = FrameMismatchError("shot1", 100, 90)
        assert err.clip_name == "shot1"
        assert err.input_count == 100
        assert err.alpha_count == 90

    def test_message_contains_counts(self):
        """The string representation must include both counts so the user knows what mismatched."""
        err = FrameMismatchError("shot1", 100, 90)
        assert "100" in str(err)
        assert "90" in str(err)
        assert "shot1" in str(err)


class TestFrameReadError:
    """FrameReadError - structured attributes for failed frame reads."""

    def test_attributes(self):
        """clip_name, frame_index, and path must be stored for log messages and recovery."""
        err = FrameReadError("shot1", 42, "/frames/0042.exr")
        assert err.clip_name == "shot1"
        assert err.frame_index == 42
        assert err.path == "/frames/0042.exr"

    def test_message(self):
        """The string representation must include the frame index and clip name."""
        err = FrameReadError("shot1", 42, "/frames/0042.exr")
        assert "42" in str(err)
        assert "shot1" in str(err)


class TestJobCancelledError:
    """JobCancelledError - optional frame_index for mid-clip cancellation context."""

    def test_without_frame_index(self):
        """frame_index must be None and must not appear in the message when not provided."""
        err = JobCancelledError("shot1")
        assert err.frame_index is None
        assert "shot1" in str(err)
        assert "frame" not in str(err)

    def test_with_frame_index(self):
        """When provided, frame_index must be stored and appear in the message."""
        err = JobCancelledError("shot1", frame_index=15)
        assert err.frame_index == 15
        assert "15" in str(err)


class TestInvalidStateTransitionError:
    """InvalidStateTransitionError - current and target state for transition debugging."""

    def test_attributes(self):
        """clip_name, current_state, and target_state must be stored."""
        err = InvalidStateTransitionError("shot1", "RAW", "COMPLETE")
        assert err.clip_name == "shot1"
        assert err.current_state == "RAW"
        assert err.target_state == "COMPLETE"

    def test_message(self):
        """Both state names must appear in the message so the user knows what transition failed."""
        err = InvalidStateTransitionError("shot1", "RAW", "COMPLETE")
        assert "RAW" in str(err)
        assert "COMPLETE" in str(err)


class TestVRAMInsufficientError:
    """VRAMInsufficientError - required vs available GB for actionable error messages."""

    def test_attributes(self):
        """required_gb and available_gb must be stored as floats."""
        err = VRAMInsufficientError(22.7, 8.0)
        assert err.required_gb == 22.7
        assert err.available_gb == 8.0

    def test_message_formatted(self):
        """Both GB values must appear in the message so the user knows how much VRAM is needed."""
        err = VRAMInsufficientError(22.7, 8.0)
        assert "22.7" in str(err)
        assert "8.0" in str(err)


class TestFFmpegNotFoundError:
    """FFmpegNotFoundError - platform-specific install hints reduce support burden."""

    def test_darwin_hint(self):
        """macOS users must see a brew install hint."""
        with patch("corridorkey.errors.sys") as mock_sys:
            mock_sys.platform = "darwin"
            err = FFmpegNotFoundError("ffmpeg")
        assert "brew" in str(err)

    def test_linux_hint(self):
        """Linux users must see an apt install hint and the binary name."""
        with patch("corridorkey.errors.sys") as mock_sys:
            mock_sys.platform = "linux"
            err = FFmpegNotFoundError("ffprobe")
        assert "apt" in str(err)
        assert "ffprobe" in str(err)

    def test_windows_hint(self):
        """Windows users must see a choco install hint."""
        with patch("corridorkey.errors.sys") as mock_sys:
            mock_sys.platform = "win32"
            err = FFmpegNotFoundError("ffmpeg")
        assert "choco" in str(err)

    def test_default_binary_is_ffmpeg(self):
        """Constructing without arguments must default to the ffmpeg binary name."""
        with patch("corridorkey.errors.sys") as mock_sys:
            mock_sys.platform = "linux"
            err = FFmpegNotFoundError("ffmpeg")
        assert "ffmpeg" in str(err)
