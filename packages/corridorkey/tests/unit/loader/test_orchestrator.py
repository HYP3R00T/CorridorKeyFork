"""Unit tests for corridorkey.stages.loader.orchestrator — load(), resolve_alpha()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from corridorkey.errors import ExtractionError, FrameMismatchError
from corridorkey.events import PipelineEvents
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.loader.extractor import DEFAULT_PNG_COMPRESSION, VideoMetadata
from corridorkey.stages.loader.orchestrator import attach_alpha, load
from corridorkey.stages.scanner.contracts import Clip
from pydantic import ValidationError


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


def _fake_meta(frame_count: int = 0, **kwargs: object) -> VideoMetadata:
    return VideoMetadata(
        filename="clip.mp4",
        width=1920,
        height=1080,
        fps_num=24,
        fps_den=1,
        pix_fmt="yuv420p",
        codec_name="h264",
        frame_count=frame_count,
        **kwargs,  # type: ignore[arg-type]
    )


def _patch_extract(output_dir_ref: list[Path], frame_count: int = 5):
    """Return a fake extract_video that writes frames and returns metadata."""

    def _fake(
        video_path, output_dir, pattern="frame_{:06d}.png", png_compression=DEFAULT_PNG_COMPRESSION, on_frame=None
    ):
        _make_frames(output_dir, count=frame_count)
        output_dir_ref.append(output_dir)
        return _fake_meta(frame_count=frame_count)

    return _fake


class TestLoadImageSequence:
    def test_no_alpha(self, tmp_path: Path):
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

    def test_with_alpha(self, tmp_path: Path):
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

    def test_png_compression_default(self, tmp_path: Path):
        clip = _make_clip(tmp_path / "my_clip")
        assert load(clip).png_compression == DEFAULT_PNG_COMPRESSION

    def test_png_compression_custom(self, tmp_path: Path):
        clip = _make_clip(tmp_path / "my_clip")
        assert load(clip, png_compression=0).png_compression == 0

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

    def test_manifest_is_frozen(self, tmp_path: Path):
        clip = _make_clip(tmp_path / "my_clip")
        manifest = load(clip)
        with pytest.raises(ValidationError):
            manifest.frame_count = 99


class TestLoadVideoInput:
    def test_extraction_called_on_first_run(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()
        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)

        extracted: list[Path] = []
        with (
            patch(
                "corridorkey.stages.loader.orchestrator.extract_video",
                side_effect=_patch_extract(extracted, frame_count=5),
            ),
            patch("corridorkey.stages.loader.orchestrator.read_video_metadata", return_value=_fake_meta(frame_count=5)),
        ):
            manifest = load(clip)

        assert manifest.frame_count == 5
        assert len(extracted) == 1

    def test_video_meta_written_on_first_run(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()
        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)

        from corridorkey.stages.loader.extractor import load_video_metadata

        with (
            patch(
                "corridorkey.stages.loader.orchestrator.extract_video",
                side_effect=_patch_extract([], frame_count=3),
            ),
            patch("corridorkey.stages.loader.orchestrator.read_video_metadata", return_value=_fake_meta(frame_count=3)),
        ):
            load(clip)

        saved = load_video_metadata(clip_dir)
        assert saved is not None
        assert saved.width == 1920

    def test_extraction_skipped_on_second_run_when_count_matches(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()

        # Pre-populate Frames/ with 4 frames
        frames_dir = clip_dir / "Frames"
        _make_frames(frames_dir, count=4)

        from corridorkey.stages.loader.extractor import save_video_metadata

        save_video_metadata(_fake_meta(frame_count=4), clip_dir)

        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)

        with (
            patch("corridorkey.stages.loader.orchestrator.extract_video") as mock_extract,
            patch("corridorkey.stages.loader.orchestrator.read_video_metadata", return_value=_fake_meta(frame_count=4)),
        ):
            manifest = load(clip)
            mock_extract.assert_not_called()

        assert manifest.frame_count == 4

    def test_re_extraction_on_incomplete_cache(self, tmp_path: Path):
        """If cached frame count doesn't match container, re-extract."""
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()

        # Only 2 frames on disk, but container says 5
        frames_dir = clip_dir / "Frames"
        _make_frames(frames_dir, count=2)

        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)

        extracted: list[Path] = []
        with (
            patch(
                "corridorkey.stages.loader.orchestrator.extract_video",
                side_effect=_patch_extract(extracted, frame_count=5),
            ),
            patch("corridorkey.stages.loader.orchestrator.read_video_metadata", return_value=_fake_meta(frame_count=5)),
        ):
            manifest = load(clip)

        assert len(extracted) == 1
        assert manifest.frame_count == 5

    def test_extraction_error_raises_extraction_error(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()
        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)

        with (
            patch("corridorkey.stages.loader.orchestrator.extract_video", side_effect=RuntimeError("codec error")),
            patch("corridorkey.stages.loader.orchestrator.read_video_metadata", return_value=_fake_meta()),
            pytest.raises(ExtractionError, match="extraction failed"),
        ):
            load(clip)

    def test_png_compression_passed_to_extract(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()
        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)

        mock_extract = MagicMock(return_value=_fake_meta(frame_count=2))
        mock_extract.side_effect = lambda *a, **kw: (
            _make_frames(clip_dir / "Frames", count=2) or _fake_meta(frame_count=2)
        )

        with (
            patch(
                "corridorkey.stages.loader.orchestrator.extract_video",
                side_effect=lambda vp, od, pattern="frame_{:06d}.png", png_compression=DEFAULT_PNG_COMPRESSION, on_frame=None: (
                    _make_frames(od, count=2) or _fake_meta(frame_count=2)
                ),
            ) as m,
            patch("corridorkey.stages.loader.orchestrator.read_video_metadata", return_value=_fake_meta()),
        ):
            load(clip, png_compression=0)
            # Verify png_compression=0 was forwarded
            call_kwargs = m.call_args
            assert (
                call_kwargs.kwargs.get(
                    "png_compression", call_kwargs.args[3] if len(call_kwargs.args) > 3 else DEFAULT_PNG_COMPRESSION
                )
                == 0
            )


class TestLoadEvents:
    def test_stage_start_fired_before_extraction(self, tmp_path: Path):
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()
        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)

        stage_starts: list[tuple[str, int]] = []
        events = PipelineEvents(on_stage_start=lambda s, t: stage_starts.append((s, t)))

        with (
            patch(
                "corridorkey.stages.loader.orchestrator.extract_video",
                side_effect=_patch_extract([], frame_count=3),
            ),
            patch("corridorkey.stages.loader.orchestrator.read_video_metadata", return_value=_fake_meta(frame_count=3)),
        ):
            load(clip, events=events)

        assert any(s == "extract" for s, _ in stage_starts)

    def test_stage_start_total_from_metadata(self, tmp_path: Path):
        """stage_start total should come from read_video_metadata, not 0."""
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()
        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)

        totals: list[int] = []
        events = PipelineEvents(on_stage_start=lambda s, t: totals.append(t))

        with (
            patch(
                "corridorkey.stages.loader.orchestrator.extract_video",
                side_effect=_patch_extract([], frame_count=10),
            ),
            patch(
                "corridorkey.stages.loader.orchestrator.read_video_metadata",
                return_value=_fake_meta(frame_count=10),
            ),
        ):
            load(clip, events=events)

        assert totals[0] == 10  # accurate total, not 0


class TestResolveAlpha:
    def _base_manifest(self, tmp_path: Path) -> ClipManifest:
        return load(_make_clip(tmp_path / "my_clip"))

    def test_sets_alpha_frames_dir(self, tmp_path: Path):
        manifest = self._base_manifest(tmp_path)
        alpha_dir = tmp_path / "my_clip" / "AlphaFrames"
        _make_frames(alpha_dir, count=3)
        updated = attach_alpha(manifest, alpha_dir)
        assert updated.alpha_frames_dir == alpha_dir
        assert updated.needs_alpha is False

    def test_original_manifest_unchanged(self, tmp_path: Path):
        """resolve_alpha must return a new manifest — original is frozen."""
        manifest = self._base_manifest(tmp_path)
        alpha_dir = tmp_path / "my_clip" / "AlphaFrames"
        _make_frames(alpha_dir, count=3)
        attach_alpha(manifest, alpha_dir)
        assert manifest.needs_alpha is True
        assert manifest.alpha_frames_dir is None

    def test_raises_if_already_has_alpha(self, tmp_path: Path):
        clip = _make_clip(tmp_path / "my_clip", with_alpha=True)
        manifest = load(clip)
        alpha_dir = tmp_path / "my_clip" / "AlphaFrames"
        _make_frames(alpha_dir, count=3)
        with pytest.raises(ValueError, match="already has alpha"):
            attach_alpha(manifest, alpha_dir)

    def test_raises_on_frame_count_mismatch(self, tmp_path: Path):
        manifest = self._base_manifest(tmp_path)
        alpha_dir = tmp_path / "my_clip" / "AlphaFrames"
        _make_frames(alpha_dir, count=2)  # manifest has 3 frames
        with pytest.raises(FrameMismatchError, match="frame count mismatch"):
            attach_alpha(manifest, alpha_dir)

    def test_raises_on_empty_alpha_dir(self, tmp_path: Path):
        manifest = self._base_manifest(tmp_path)
        alpha_dir = tmp_path / "my_clip" / "AlphaFrames"
        alpha_dir.mkdir()
        with pytest.raises(FrameMismatchError):
            attach_alpha(manifest, alpha_dir)

    def test_raises_if_alpha_dir_does_not_exist(self, tmp_path: Path):
        manifest = self._base_manifest(tmp_path)
        with pytest.raises(ValueError, match="does not exist"):
            attach_alpha(manifest, tmp_path / "ghost")

    def test_does_not_rescan_input_dir(self, tmp_path: Path):
        """resolve_alpha should use manifest.frame_count, not re-scan frames_dir."""
        manifest = self._base_manifest(tmp_path)
        alpha_dir = tmp_path / "my_clip" / "AlphaFrames"
        _make_frames(alpha_dir, count=3)

        with patch(
            "corridorkey.stages.loader.orchestrator.scan_frames",
            wraps=__import__("corridorkey.stages.loader.validator", fromlist=["scan_frames"]).scan_frames,
        ) as mock_scan:
            attach_alpha(manifest, alpha_dir)
            # scan_frames should only be called for alpha_dir, not frames_dir
            called_paths = [call.args[0] for call in mock_scan.call_args_list]
            assert manifest.frames_dir not in called_paths


class TestLoadVideoMetaPath:
    def test_video_meta_path_set_when_json_exists(self, tmp_path: Path):
        """video_meta_path is set when video_meta.json already exists in clip root."""
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()
        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)

        # Pre-create video_meta.json so the candidate.exists() branch is hit
        from corridorkey.stages.loader.extractor import save_video_metadata

        save_video_metadata(_fake_meta(frame_count=3), clip_dir)

        with (
            patch(
                "corridorkey.stages.loader.orchestrator.extract_video",
                side_effect=_patch_extract([], frame_count=3),
            ),
            patch(
                "corridorkey.stages.loader.orchestrator.read_video_metadata",
                return_value=_fake_meta(frame_count=3),
            ),
        ):
            manifest = load(clip)

        assert manifest.video_meta_path is not None
        assert manifest.video_meta_path.name == "video_meta.json"

    def test_read_video_metadata_runtime_error_during_cache_check(self, tmp_path: Path):
        """RuntimeError from read_video_metadata during cache check sets expected=0 (skip re-extract)."""
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()

        # Pre-populate Frames/ so the cache check branch is entered
        frames_dir = clip_dir / "Frames"
        _make_frames(frames_dir, count=3)

        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)

        with (
            patch(
                "corridorkey.stages.loader.orchestrator.read_video_metadata",
                side_effect=RuntimeError("no container"),
            ),
        ):
            # expected=0 means existing.count == 0 is False but expected == 0 → skip re-extract
            manifest = load(clip)

        assert manifest.frame_count == 3

    def test_read_video_metadata_runtime_error_before_extraction(self, tmp_path: Path):
        """RuntimeError from read_video_metadata before extraction sets total_frames=0."""
        clip_dir = tmp_path / "my_clip"
        input_dir = clip_dir / "Input"
        input_dir.mkdir(parents=True)
        video = input_dir / "clip.mp4"
        video.touch()
        clip = Clip(name="my_clip", root=clip_dir, input_path=video, alpha_path=None)

        totals: list[int] = []
        events = PipelineEvents(on_stage_start=lambda s, t: totals.append(t))

        with (
            patch(
                "corridorkey.stages.loader.orchestrator.extract_video",
                side_effect=_patch_extract([], frame_count=2),
            ),
            patch(
                "corridorkey.stages.loader.orchestrator.read_video_metadata",
                side_effect=RuntimeError("no container"),
            ),
        ):
            load(clip, events=events)

        # total_frames falls back to 0 when metadata read fails
        assert totals[0] == 0
