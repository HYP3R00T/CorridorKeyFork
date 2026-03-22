"""Unit tests for corridorkey_new.stages.loader.orchestrator — load(), resolve_alpha()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from corridorkey_new.errors import FrameMismatchError
from corridorkey_new.stages.loader.contracts import ClipManifest
from corridorkey_new.stages.loader.orchestrator import load, resolve_alpha
from corridorkey_new.stages.scanner.contracts import Clip


def _make_frames(directory: Path, count: int = 3, ext: str = ".png") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        (directory / f"frame_{i:06d}{ext}").touch()


def _make_clip(root: Path, with_alpha: bool = False, ext: str = ".png") -> Clip:
    input_dir = root / "Input"
    _make_frames(input_dir, ext=ext)
    alpha_path = None
    if with_alpha:
        alpha_dir = root / "AlphaHint"
        _make_frames(alpha_dir, ext=ext)
        alpha_path = alpha_dir
    return Clip(name=root.name, root=root, input_path=input_dir, alpha_path=alpha_path)


class TestLoad:
    def test_image_sequence_no_alpha(self, tmp_path: Path):
        clip = _make_clip(tmp_path / "my_clip")
        manifest = load(clip)
        assert manifest.clip_name == "my_clip"
        assert manifest.needs_alpha is True
        assert manifest.alpha_frames_dir is None
        assert manifest.frame_count == 3
        assert manifest.frame_range == (0, 3)
        assert manifest.is_linear is False
        assert manifest.video_meta_path is None
        assert (tmp_path / "my_clip" / "Output").is_dir()

    def test_image_sequence_with_alpha(self, tmp_path: Path):
        clip = _make_clip(tmp_path / "my_clip", with_alpha=True)
        manifest = load(clip)
        assert manifest.needs_alpha is False
        assert manifest.alpha_frames_dir is not None
        assert manifest.frame_count == 3

    def test_linear_exr_detected(self, tmp_path: Path):
        clip = _make_clip(tmp_path / "my_clip", ext=".exr")
        assert load(clip).is_linear is True

    def test_output_dir_created(self, tmp_path: Path):
        clip = _make_clip(tmp_path / "my_clip")
        assert load(clip).output_dir.is_dir()

    def test_empty_input_raises(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        clip = Clip(name="my_clip", root=clip_dir, input_path=input_dir, alpha_path=None)
        with pytest.raises(ValueError, match="no image frames"):
            load(clip)

    def test_alpha_frame_count_mismatch_raises(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        alpha_dir = clip_dir / "AlphaHint"
        _make_frames(input_dir, count=3)
        _make_frames(alpha_dir, count=2)
        clip = Clip(name="my_clip", root=clip_dir, input_path=input_dir, alpha_path=alpha_dir)
        with pytest.raises(FrameMismatchError, match="frame count mismatch"):
            load(clip)

    def test_video_input_triggers_extraction(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()

        from corridorkey_new.stages.loader.extractor import VideoMetadata

        fake_meta = VideoMetadata(
            filename="clip.mp4",
            width=1920,
            height=1080,
            fps_num=24,
            fps_den=1,
            pix_fmt="yuv420p",
            codec_name="h264",
        )

        def fake_extract(video_path, output_dir, pattern="frame_{:06d}.png", on_frame=None):
            _make_frames(output_dir, count=5)
            return fake_meta

        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)
        with patch("corridorkey_new.stages.loader.orchestrator.extract_video", side_effect=fake_extract):
            manifest = load(clip)

        assert manifest.frame_count == 5
        assert manifest.video_meta_path == clip_dir / "video_meta.json"
        assert manifest.video_meta_path is not None and manifest.video_meta_path.exists()

    def test_video_meta_written_on_first_run(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()

        from corridorkey_new.stages.loader.extractor import VideoMetadata, load_video_metadata

        fake_meta = VideoMetadata(
            filename="clip.mp4",
            width=1280,
            height=720,
            fps_num=30000,
            fps_den=1001,
            pix_fmt="yuv420p",
            codec_name="h264",
            has_audio=True,
        )

        def fake_extract(video_path, output_dir, pattern="frame_{:06d}.png", on_frame=None):
            _make_frames(output_dir, count=2)
            return fake_meta

        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)
        with patch("corridorkey_new.stages.loader.orchestrator.extract_video", side_effect=fake_extract):
            load(clip)

        saved = load_video_metadata(clip_dir)
        assert saved is not None
        assert saved.width == 1280
        assert saved.has_audio is True

    def test_video_meta_read_on_second_run(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()

        frames_dir = clip_dir / "Frames"
        _make_frames(frames_dir, count=4)

        from corridorkey_new.stages.loader.extractor import VideoMetadata, save_video_metadata

        save_video_metadata(
            VideoMetadata(
                filename="clip.mp4",
                width=1920,
                height=1080,
                fps_num=24,
                fps_den=1,
                pix_fmt="yuv420p",
                codec_name="h264",
            ),
            clip_dir,
        )

        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)
        with patch("corridorkey_new.stages.loader.orchestrator.extract_video") as mock_extract:
            manifest = load(clip)
            mock_extract.assert_not_called()

        assert manifest.frame_count == 4


class TestResolveAlpha:
    def _base_manifest(self, tmp_path: Path) -> ClipManifest:
        return load(_make_clip(tmp_path / "my_clip"))

    def test_sets_alpha_frames_dir(self, tmp_path: Path):
        manifest = self._base_manifest(tmp_path)
        alpha_dir = tmp_path / "my_clip" / "AlphaFrames"
        _make_frames(alpha_dir, count=3)
        updated = resolve_alpha(manifest, alpha_dir)
        assert updated.alpha_frames_dir == alpha_dir
        assert updated.needs_alpha is False

    def test_original_manifest_unchanged(self, tmp_path: Path):
        manifest = self._base_manifest(tmp_path)
        alpha_dir = tmp_path / "my_clip" / "AlphaFrames"
        _make_frames(alpha_dir, count=3)
        resolve_alpha(manifest, alpha_dir)
        assert manifest.needs_alpha is True

    def test_raises_if_already_has_alpha(self, tmp_path: Path):
        clip = _make_clip(tmp_path / "my_clip", with_alpha=True)
        manifest = load(clip)
        alpha_dir = tmp_path / "my_clip" / "AlphaFrames"
        _make_frames(alpha_dir, count=3)
        with pytest.raises(ValueError, match="already has alpha"):
            resolve_alpha(manifest, alpha_dir)

    def test_raises_on_frame_count_mismatch(self, tmp_path: Path):
        manifest = self._base_manifest(tmp_path)
        alpha_dir = tmp_path / "my_clip" / "AlphaFrames"
        _make_frames(alpha_dir, count=2)
        with pytest.raises(FrameMismatchError, match="frame count mismatch"):
            resolve_alpha(manifest, alpha_dir)

    def test_raises_on_empty_alpha_dir(self, tmp_path: Path):
        manifest = self._base_manifest(tmp_path)
        alpha_dir = tmp_path / "my_clip" / "AlphaFrames"
        alpha_dir.mkdir()
        with pytest.raises(FrameMismatchError, match="frame count mismatch"):
            resolve_alpha(manifest, alpha_dir)
