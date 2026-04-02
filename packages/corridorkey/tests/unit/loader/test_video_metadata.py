"""Unit tests for VideoMetadata computed properties — fps and estimated_frame_count."""

from __future__ import annotations

import pytest
from corridorkey.stages.loader.extractor import VideoMetadata


def _make_meta(**kwargs) -> VideoMetadata:
    return VideoMetadata(
        **{
            "filename": "clip.mp4",
            "width": 1920,
            "height": 1080,
            "fps_num": 24,
            "fps_den": 1,
            "pix_fmt": "yuv420p",
            "codec_name": "h264",
        }
        | kwargs
    )


class TestVideoMetadataFps:
    def test_fps_integer_rate(self):
        meta = _make_meta(fps_num=24, fps_den=1)
        assert meta.fps == pytest.approx(24.0)

    def test_fps_fractional_rate(self):
        meta = _make_meta(fps_num=30000, fps_den=1001)
        assert meta.fps == pytest.approx(29.97, abs=0.01)

    def test_fps_zero_denominator_returns_zero(self):
        meta = _make_meta(fps_num=24, fps_den=0)
        assert meta.fps == 0.0

    def test_fps_25(self):
        meta = _make_meta(fps_num=25, fps_den=1)
        assert meta.fps == pytest.approx(25.0)

    def test_fps_60(self):
        meta = _make_meta(fps_num=60, fps_den=1)
        assert meta.fps == pytest.approx(60.0)


class TestVideoMetadataEstimatedFrameCount:
    def test_uses_frame_count_when_positive(self):
        meta = _make_meta(frame_count=100, duration_s=None)
        assert meta.estimated_frame_count == 100

    def test_falls_back_to_duration_times_fps(self):
        meta = _make_meta(frame_count=0, fps_num=24, fps_den=1, duration_s=5.0)
        assert meta.estimated_frame_count == 120  # 24 * 5

    def test_rounds_fractional_result(self):
        # 29.97 fps * 10s = 299.7 → rounds to 300
        meta = _make_meta(frame_count=0, fps_num=30000, fps_den=1001, duration_s=10.0)
        assert meta.estimated_frame_count == 300

    def test_returns_zero_when_no_frame_count_and_no_duration(self):
        meta = _make_meta(frame_count=0, duration_s=None)
        assert meta.estimated_frame_count == 0

    def test_returns_zero_when_fps_is_zero(self):
        meta = _make_meta(frame_count=0, fps_num=0, fps_den=0, duration_s=5.0)
        assert meta.estimated_frame_count == 0

    def test_frame_count_takes_priority_over_duration(self):
        # Even if duration would give a different answer, frame_count wins
        meta = _make_meta(frame_count=50, fps_num=24, fps_den=1, duration_s=10.0)
        assert meta.estimated_frame_count == 50
