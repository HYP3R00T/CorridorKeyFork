"""Unit tests for corridorkey.stages.preprocessor.orchestrator — preprocess_frame()."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.loader.validator import get_frame_files
from corridorkey.stages.preprocessor import PreprocessConfig, PreprocessedFrame, preprocess_frame


def _write_png(path: Path, h: int = 64, w: int = 64, channels: int = 3) -> None:
    img = np.zeros((h, w), dtype=np.uint8) if channels == 1 else np.zeros((h, w, channels), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _make_manifest(
    tmp_path: Path,
    frame_count: int = 3,
    is_linear: bool = False,
    with_alpha: bool = True,
    img_h: int = 64,
    img_w: int = 64,
) -> ClipManifest:
    frames_dir = tmp_path / "Frames"
    frames_dir.mkdir(parents=True)
    alpha_dir = tmp_path / "AlphaFrames"
    output_dir = tmp_path / "Output"
    output_dir.mkdir()

    for i in range(frame_count):
        _write_png(frames_dir / f"frame_{i:06d}.png", h=img_h, w=img_w, channels=3)

    if with_alpha:
        alpha_dir.mkdir()
        for i in range(frame_count):
            _write_png(alpha_dir / f"frame_{i:06d}.png", h=img_h, w=img_w, channels=1)

    return ClipManifest(
        clip_name="test_clip",
        clip_root=tmp_path,
        frames_dir=frames_dir,
        alpha_frames_dir=alpha_dir if with_alpha else None,
        output_dir=output_dir,
        needs_alpha=not with_alpha,
        frame_count=frame_count,
        frame_range=(0, frame_count),
        is_linear=is_linear,
    )


class TestPreprocessFrame:
    def test_returns_preprocessed_frame(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        config = PreprocessConfig(img_size=64, device="cpu")
        result = preprocess_frame(manifest, 0, config)
        assert isinstance(result, PreprocessedFrame)

    def test_tensor_shape(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        config = PreprocessConfig(img_size=64, device="cpu")
        result = preprocess_frame(manifest, 0, config)
        assert result.tensor.shape == (1, 4, 64, 64)

    def test_tensor_dtype_float32(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        config = PreprocessConfig(img_size=64, device="cpu")
        result = preprocess_frame(manifest, 0, config)
        assert result.tensor.dtype == torch.float32

    def test_tensor_dtype_float16_when_half_precision(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        config = PreprocessConfig(img_size=64, device="cpu", half_precision=True)
        result = preprocess_frame(manifest, 0, config)
        assert result.tensor.dtype == torch.float16

    def test_tensor_on_cpu(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        config = PreprocessConfig(img_size=64, device="cpu")
        result = preprocess_frame(manifest, 0, config)
        assert result.tensor.device.type == "cpu"

    def test_meta_original_dimensions(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, img_h=48, img_w=80)
        config = PreprocessConfig(img_size=64, device="cpu")
        result = preprocess_frame(manifest, 0, config)
        assert result.meta.original_h == 48
        assert result.meta.original_w == 80

    def test_meta_frame_index(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=3)
        config = PreprocessConfig(img_size=64, device="cpu")
        result = preprocess_frame(manifest, 2, config)
        assert result.meta.frame_index == 2

    def test_needs_alpha_raises(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, with_alpha=False)
        config = PreprocessConfig(img_size=64, device="cpu")
        with pytest.raises(ValueError, match="still needs alpha"):
            preprocess_frame(manifest, 0, config)

    def test_out_of_range_index_raises(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=3)
        config = PreprocessConfig(img_size=64, device="cpu")
        with pytest.raises(ValueError, match="out of range"):
            preprocess_frame(manifest, 5, config)

    def test_negative_index_raises(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=3)
        config = PreprocessConfig(img_size=64, device="cpu")
        with pytest.raises(ValueError, match="out of range"):
            preprocess_frame(manifest, -1, config)

    def test_all_frames_in_range(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path, frame_count=3)
        config = PreprocessConfig(img_size=64, device="cpu")
        imgs = get_frame_files(manifest.frames_dir)
        alps = get_frame_files(manifest.alpha_frames_dir)  # type: ignore[arg-type]
        for i in range(*manifest.frame_range):
            result = preprocess_frame(manifest, i, config, image_files=imgs, alpha_files=alps)
            assert result.meta.frame_index == i

    def test_prebuilt_file_lists_used(self, tmp_path: Path):
        """Passing pre-built file lists should produce the same result as not passing them."""
        manifest = _make_manifest(tmp_path, frame_count=2)
        config = PreprocessConfig(img_size=64, device="cpu")
        imgs = get_frame_files(manifest.frames_dir)
        alps = get_frame_files(manifest.alpha_frames_dir)  # type: ignore[arg-type]
        r1 = preprocess_frame(manifest, 0, config)
        r2 = preprocess_frame(manifest, 0, config, image_files=imgs, alpha_files=alps)
        assert r1.tensor.shape == r2.tensor.shape
        assert r1.meta.frame_index == r2.meta.frame_index
        assert r1.meta.original_h == r2.meta.original_h
        assert r1.meta.original_w == r2.meta.original_w

    def test_linear_input_converted_to_srgb(self, tmp_path: Path):
        """is_linear=True should produce a different tensor than is_linear=False."""
        # Use mid-grey (128) so linear->sRGB conversion produces a visible difference.
        clip_dir = tmp_path / "srgb"
        frames_dir = clip_dir / "Frames"
        frames_dir.mkdir(parents=True)
        alpha_dir = clip_dir / "AlphaFrames"
        alpha_dir.mkdir()
        output_dir = clip_dir / "Output"
        output_dir.mkdir()
        grey = np.full((64, 64, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "frame_000000.png"), grey)
        alpha = np.full((64, 64), 255, dtype=np.uint8)
        cv2.imwrite(str(alpha_dir / "frame_000000.png"), alpha)

        manifest_srgb = ClipManifest(
            clip_name="test_clip",
            clip_root=clip_dir,
            frames_dir=frames_dir,
            alpha_frames_dir=alpha_dir,
            output_dir=output_dir,
            needs_alpha=False,
            frame_count=1,
            frame_range=(0, 1),
            is_linear=False,
        )
        manifest_linear = ClipManifest(
            clip_name=manifest_srgb.clip_name,
            clip_root=manifest_srgb.clip_root,
            frames_dir=manifest_srgb.frames_dir,
            alpha_frames_dir=manifest_srgb.alpha_frames_dir,
            output_dir=manifest_srgb.output_dir,
            needs_alpha=False,
            frame_count=manifest_srgb.frame_count,
            frame_range=manifest_srgb.frame_range,
            is_linear=True,
        )
        config = PreprocessConfig(img_size=64, device="cpu")
        r_srgb = preprocess_frame(manifest_srgb, 0, config)
        r_linear = preprocess_frame(manifest_linear, 0, config)
        # Tensors should differ because color space conversion was applied
        assert not torch.allclose(r_srgb.tensor, r_linear.tensor)

    def test_source_passthrough_true_produces_numpy_array(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        config = PreprocessConfig(img_size=64, device="cpu", source_passthrough=True)
        result = preprocess_frame(manifest, 0, config)
        assert result.meta.source_image is not None
        assert isinstance(result.meta.source_image, np.ndarray)

    def test_source_passthrough_false_produces_none(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        config = PreprocessConfig(img_size=64, device="cpu", source_passthrough=False)
        result = preprocess_frame(manifest, 0, config)
        assert result.meta.source_image is None

    def test_source_image_shape_is_original_resolution(self, tmp_path: Path):
        """source_image must be at original resolution, not model resolution."""
        manifest = _make_manifest(tmp_path, img_h=48, img_w=80)
        config = PreprocessConfig(img_size=64, device="cpu", source_passthrough=True)
        result = preprocess_frame(manifest, 0, config)
        assert result.meta.source_image is not None
        assert result.meta.source_image.shape == (48, 80, 3)

    def test_source_image_is_srgb_for_linear_input(self, tmp_path: Path):
        """For linear inputs, source_image must be sRGB-converted (same as model input)."""
        clip_dir = tmp_path / "linear_clip"
        frames_dir = clip_dir / "Frames"
        frames_dir.mkdir(parents=True)
        alpha_dir = clip_dir / "AlphaFrames"
        alpha_dir.mkdir()
        output_dir = clip_dir / "Output"
        output_dir.mkdir()
        # Mid-grey linear — sRGB conversion will change the values
        grey = np.full((8, 8, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / "frame_000000.png"), grey)
        cv2.imwrite(str(alpha_dir / "frame_000000.png"), np.full((8, 8), 255, dtype=np.uint8))

        manifest = ClipManifest(
            clip_name="linear",
            clip_root=clip_dir,
            frames_dir=frames_dir,
            alpha_frames_dir=alpha_dir,
            output_dir=output_dir,
            needs_alpha=False,
            frame_count=1,
            frame_range=(0, 1),
            is_linear=True,
        )
        config = PreprocessConfig(img_size=8, device="cpu", source_passthrough=True)
        result = preprocess_frame(manifest, 0, config)
        # Raw pixel value 128/255 ≈ 0.502 linear; after sRGB conversion it should be ~0.735
        assert result.meta.source_image is not None
        assert result.meta.source_image[0, 0, 0] > 0.6  # clearly converted, not raw linear

    def test_source_image_not_affected_by_resize(self, tmp_path: Path):
        """source_image is captured before resize — tensor is img_size, source is original."""
        manifest = _make_manifest(tmp_path, img_h=32, img_w=32)
        config = PreprocessConfig(img_size=16, device="cpu", source_passthrough=True)
        result = preprocess_frame(manifest, 0, config)
        assert result.tensor.shape == (1, 4, 16, 16)
        assert result.meta.source_image is not None
        assert result.meta.source_image.shape == (32, 32, 3)

    def test_source_image_is_rgb_not_bgr(self, tmp_path: Path):
        """source_image must be in RGB order — the postprocessor contract requires it."""
        frames_dir = tmp_path / "Frames"
        frames_dir.mkdir(parents=True)
        alpha_dir = tmp_path / "AlphaFrames"
        alpha_dir.mkdir()
        output_dir = tmp_path / "Output"
        output_dir.mkdir()
        # Pure-red image in BGR (OpenCV native): B=0, G=0, R=255
        red_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
        red_bgr[:, :, 2] = 255
        cv2.imwrite(str(frames_dir / "frame_000000.png"), red_bgr)
        cv2.imwrite(str(alpha_dir / "frame_000000.png"), np.full((8, 8), 255, dtype=np.uint8))
        manifest = ClipManifest(
            clip_name="rgb_test",
            clip_root=tmp_path,
            frames_dir=frames_dir,
            alpha_frames_dir=alpha_dir,
            output_dir=output_dir,
            needs_alpha=False,
            frame_count=1,
            frame_range=(0, 1),
            is_linear=False,
        )
        config = PreprocessConfig(img_size=8, device="cpu", source_passthrough=True)
        result = preprocess_frame(manifest, 0, config)
        assert result.meta.source_image is not None
        # In RGB order: channel 0 = R = 1.0, channels 1 and 2 = 0.0
        assert result.meta.source_image[0, 0, 0] == pytest.approx(1.0)
        assert result.meta.source_image[0, 0, 1] == pytest.approx(0.0)
        assert result.meta.source_image[0, 0, 2] == pytest.approx(0.0)

    @pytest.mark.gpu
    def test_tensor_on_cuda(self, tmp_path: Path):
        manifest = _make_manifest(tmp_path)
        config = PreprocessConfig(img_size=64, device="cuda")
        result = preprocess_frame(manifest, 0, config)
        assert result.tensor.device.type == "cuda"

    @pytest.mark.gpu
    def test_source_image_on_cpu_even_when_device_is_cuda(self, tmp_path: Path):
        """source_image must always be a CPU numpy array regardless of device."""
        manifest = _make_manifest(tmp_path)
        config = PreprocessConfig(img_size=64, device="cuda", source_passthrough=True)
        result = preprocess_frame(manifest, 0, config)
        assert result.meta.source_image is not None
        assert isinstance(result.meta.source_image, np.ndarray)  # never a tensor


class TestPreprocessConfigValidation:
    def test_img_size_zero_raises(self):
        with pytest.raises(ValueError, match="img_size must be > 0"):
            PreprocessConfig(img_size=0)

    def test_img_size_negative_raises(self):
        with pytest.raises(ValueError, match="img_size must be > 0"):
            PreprocessConfig(img_size=-1)

    def test_img_size_positive_is_valid(self):
        cfg = PreprocessConfig(img_size=1)
        assert cfg.img_size == 1


class TestPreprocessorPackageExports:
    def test_image_upsample_mode_importable(self):
        from corridorkey.stages.preprocessor import ImageUpsampleMode

        assert ImageUpsampleMode is not None

    def test_default_image_upsample_mode_importable(self):
        from corridorkey.stages.preprocessor import DEFAULT_IMAGE_UPSAMPLE_MODE

        assert isinstance(DEFAULT_IMAGE_UPSAMPLE_MODE, str)
