"""Unit tests for corridorkey.stages.loader.extractor (non-video parts)."""

from __future__ import annotations

from pathlib import Path

from corridorkey.stages.loader.extractor import (
    DEFAULT_PNG_COMPRESSION,
    VideoMetadata,
    is_video,
    load_video_metadata,
    save_video_metadata,
)


def _make_meta(
    filename: str = "clip.mp4",
    width: int = 1920,
    height: int = 1080,
    fps_num: int = 24,
    fps_den: int = 1,
    pix_fmt: str = "yuv420p",
    codec_name: str = "h264",
    **kwargs: object,
) -> VideoMetadata:
    return VideoMetadata(
        filename=filename,
        width=width,
        height=height,
        fps_num=fps_num,
        fps_den=fps_den,
        pix_fmt=pix_fmt,
        codec_name=codec_name,
        **kwargs,  # type: ignore[arg-type]
    )


class TestIsVideo:
    def test_mp4_is_video(self, tmp_path: Path):
        f = tmp_path / "clip.mp4"
        f.touch()
        assert is_video(f) is True

    def test_mov_is_video(self, tmp_path: Path):
        f = tmp_path / "clip.mov"
        f.touch()
        assert is_video(f) is True

    def test_png_is_not_video(self, tmp_path: Path):
        f = tmp_path / "frame.png"
        f.touch()
        assert is_video(f) is False

    def test_directory_is_not_video(self, tmp_path: Path):
        assert is_video(tmp_path) is False

    def test_nonexistent_is_not_video(self, tmp_path: Path):
        assert is_video(tmp_path / "ghost.mp4") is False

    def test_case_insensitive_extension(self, tmp_path: Path):
        f = tmp_path / "clip.MP4"
        f.touch()
        assert is_video(f) is True


class TestVideoMetadata:
    def test_fps_property(self):
        meta = _make_meta(fps_num=30000, fps_den=1001)
        assert abs(meta.fps - 29.97) < 0.01

    def test_fps_zero_denominator(self):
        meta = _make_meta(fps_num=24, fps_den=0)
        assert meta.fps == 0.0

    def test_optional_fields_default_none(self):
        meta = _make_meta()
        assert meta.duration_s is None
        assert meta.color_space is None
        assert meta.color_transfer is None
        assert meta.color_primaries is None

    def test_has_audio_defaults_false(self):
        assert _make_meta().has_audio is False

    def test_frame_count_defaults_zero(self):
        assert _make_meta().frame_count == 0

    def test_estimated_frame_count_from_container(self):
        meta = _make_meta(frame_count=120)
        assert meta.estimated_frame_count == 120

    def test_estimated_frame_count_from_duration(self):
        # frame_count=0, fall back to duration * fps
        meta = _make_meta(fps_num=24, fps_den=1, duration_s=5.0, frame_count=0)
        assert meta.estimated_frame_count == 120

    def test_estimated_frame_count_zero_when_no_info(self):
        meta = _make_meta(frame_count=0, duration_s=None)
        assert meta.estimated_frame_count == 0

    def test_model_dump_json_roundtrip(self):
        meta = _make_meta(duration_s=12.5, has_audio=True, color_space="bt709", frame_count=300)
        restored = VideoMetadata.model_validate_json(meta.model_dump_json())
        assert restored.duration_s == 12.5
        assert restored.has_audio is True
        assert restored.color_space == "bt709"
        assert restored.frame_count == 300


class TestDefaultPngCompression:
    def test_default_is_reasonable(self):
        # Must be in valid PNG range and fast enough for intermediate frames.
        assert 0 <= DEFAULT_PNG_COMPRESSION <= 3


class TestSaveLoadVideoMetadata:
    def test_save_creates_json_file(self, tmp_path: Path):
        meta = _make_meta()
        path = save_video_metadata(meta, tmp_path)
        assert path == tmp_path / "video_meta.json"
        assert path.exists()

    def test_load_returns_none_when_missing(self, tmp_path: Path):
        assert load_video_metadata(tmp_path) is None

    def test_roundtrip(self, tmp_path: Path):
        meta = _make_meta(width=1280, height=720, fps_num=30000, fps_den=1001, has_audio=True)
        save_video_metadata(meta, tmp_path)
        loaded = load_video_metadata(tmp_path)
        assert loaded is not None
        assert loaded.width == 1280
        assert loaded.fps_num == 30000
        assert loaded.has_audio is True

    def test_overwrite_existing(self, tmp_path: Path):
        save_video_metadata(_make_meta(width=1920), tmp_path)
        save_video_metadata(_make_meta(width=3840), tmp_path)
        loaded = load_video_metadata(tmp_path)
        assert loaded is not None
        assert loaded.width == 3840


class TestVideoMetadataFpsEdgeCases:
    def test_fps_fractional_ntsc(self):
        """29.97 fps (30000/1001) is computed correctly."""
        meta = _make_meta(fps_num=30000, fps_den=1001)
        assert abs(meta.fps - 29.97) < 0.01

    def test_fps_25(self):
        """25 fps is returned exactly."""
        meta = _make_meta(fps_num=25, fps_den=1)
        assert meta.fps == 25.0

    def test_fps_60(self):
        """60 fps is returned exactly."""
        meta = _make_meta(fps_num=60, fps_den=1)
        assert meta.fps == 60.0


class TestVideoMetadataEstimatedFrameCountEdgeCases:
    def test_rounds_fractional_result(self):
        """29.97 fps * 10s = 299.7 rounds to 300."""
        meta = _make_meta(frame_count=0, fps_num=30000, fps_den=1001, duration_s=10.0)
        assert meta.estimated_frame_count == 300

    def test_returns_zero_when_fps_is_zero(self):
        """Zero fps with a duration produces zero estimated frames."""
        meta = _make_meta(frame_count=0, fps_num=0, fps_den=0, duration_s=5.0)
        assert meta.estimated_frame_count == 0

    def test_frame_count_takes_priority_over_duration(self):
        """frame_count wins over duration * fps when both are present."""
        meta = _make_meta(frame_count=50, fps_num=24, fps_den=1, duration_s=10.0)
        assert meta.estimated_frame_count == 50
