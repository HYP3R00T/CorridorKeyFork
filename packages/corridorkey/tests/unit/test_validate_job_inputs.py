"""Unit tests for validators.validate_job_inputs -- Tier 1 and Tier 2 checks.

All tests use tmp_path and mock ClipEntry objects. No GPU, no model files,
no network access required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from corridorkey.validators import ValidationResult, validate_job_inputs


def _asset(path: str, asset_type: str = "sequence", frame_count: int = 10) -> MagicMock:
    asset = MagicMock()
    asset.path = path
    asset.asset_type = asset_type
    asset.frame_count = frame_count
    return asset


def _clip(
    tmp_path: Path,
    name: str = "shot1",
    input_exists: bool = True,
    alpha_exists: bool = True,
    input_type: str = "sequence",
    alpha_type: str = "sequence",
    input_frame_count: int = 5,
    alpha_frame_count: int = 5,
) -> MagicMock:
    input_dir = tmp_path / "Frames"
    alpha_dir = tmp_path / "AlphaHint"

    if input_exists:
        input_dir.mkdir(parents=True, exist_ok=True)
    if alpha_exists:
        alpha_dir.mkdir(parents=True, exist_ok=True)

    clip = MagicMock()
    clip.name = name
    clip.root_path = str(tmp_path)

    clip.input_asset = _asset(str(input_dir), input_type, input_frame_count)
    clip.alpha_asset = _asset(str(alpha_dir), alpha_type, alpha_frame_count)

    # get_frame_files returns a list of filenames
    clip.input_asset.get_frame_files.return_value = [f"{i:05d}.png" for i in range(input_frame_count)]
    clip.alpha_asset.get_frame_files.return_value = [f"{i:05d}.png" for i in range(alpha_frame_count)]

    return clip


class TestTier1NoInputAsset:
    """Tier 1: missing input asset is a fatal error."""

    def test_no_input_asset_returns_error(self, tmp_path: Path):
        clip = MagicMock()
        clip.name = "shot1"
        clip.root_path = str(tmp_path)
        clip.input_asset = None
        result = validate_job_inputs(clip)
        assert result.ok is False
        assert any("no input asset" in e for e in result.errors)


class TestTier1PathChecks:
    """Tier 1: input and alpha path existence checks."""

    def test_ok_when_both_paths_exist(self, tmp_path: Path):
        clip = _clip(tmp_path)
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.disk_usage") as mock_du,
            patch("cv2.imread", return_value=fake_frame),
        ):
            mock_du.return_value = MagicMock(free=100 * 1024**3)
            result = validate_job_inputs(clip)
        assert result.ok is True

    def test_error_when_input_path_missing(self, tmp_path: Path):
        clip = _clip(tmp_path, input_exists=False)
        result = validate_job_inputs(clip)
        assert result.ok is False
        assert any("input path does not exist" in e for e in result.errors)

    def test_error_when_alpha_path_missing(self, tmp_path: Path):
        clip = _clip(tmp_path, alpha_exists=False)
        result = validate_job_inputs(clip)
        assert result.ok is False
        assert any("alpha path does not exist" in e for e in result.errors)


class TestTier1DiskSpace:
    """Tier 1: disk space check."""

    def test_error_when_insufficient_disk_space(self, tmp_path: Path):
        clip = _clip(tmp_path)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.disk_usage") as mock_du,
        ):
            mock_du.return_value = MagicMock(free=0)  # 0 bytes free
            result = validate_job_inputs(clip, expected_output_gb=2.0)
        assert result.ok is False
        assert any("insufficient disk space" in e for e in result.errors)

    def test_ok_when_sufficient_disk_space(self, tmp_path: Path):
        clip = _clip(tmp_path)
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.disk_usage") as mock_du,
            patch("cv2.imread", return_value=fake_frame),
        ):
            mock_du.return_value = MagicMock(free=100 * 1024**3)
            result = validate_job_inputs(clip, expected_output_gb=2.0)
        assert result.ok is True


class TestTier1VramCheck:
    """Tier 1: VRAM check when CUDA is available."""

    def test_error_when_insufficient_vram(self, tmp_path: Path):
        clip = _clip(tmp_path)
        mock_props = MagicMock()
        mock_props.total_mem = 4 * 1024**3  # 4 GB total
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch("torch.cuda.memory_reserved", return_value=3 * 1024**3),  # 3 GB reserved -> 1 GB free
            patch("shutil.disk_usage") as mock_du,
        ):
            mock_du.return_value = MagicMock(free=100 * 1024**3)
            result = validate_job_inputs(clip, min_vram_gb=6.0)
        assert result.ok is False
        assert any("insufficient VRAM" in e for e in result.errors)

    def test_ok_when_sufficient_vram(self, tmp_path: Path):
        clip = _clip(tmp_path)
        mock_props = MagicMock()
        mock_props.total_mem = 16 * 1024**3
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
            patch("torch.cuda.memory_reserved", return_value=1 * 1024**3),
            patch("shutil.disk_usage") as mock_du,
            patch("cv2.imread", return_value=fake_frame),
        ):
            mock_du.return_value = MagicMock(free=100 * 1024**3)
            result = validate_job_inputs(clip, min_vram_gb=6.0)
        assert result.ok is True


class TestTier1FrameCountMismatch:
    """Tier 1: frame count mismatch between input and alpha sequences."""

    def test_error_when_frame_counts_differ(self, tmp_path: Path):
        clip = _clip(tmp_path, input_frame_count=10, alpha_frame_count=8)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.disk_usage") as mock_du,
        ):
            mock_du.return_value = MagicMock(free=100 * 1024**3)
            result = validate_job_inputs(clip)
        assert result.ok is False
        assert any("frame count mismatch" in e for e in result.errors)

    def test_ok_when_frame_counts_match(self, tmp_path: Path):
        clip = _clip(tmp_path, input_frame_count=5, alpha_frame_count=5)
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.disk_usage") as mock_du,
            patch("cv2.imread", return_value=fake_frame),
        ):
            mock_du.return_value = MagicMock(free=100 * 1024**3)
            result = validate_job_inputs(clip)
        assert result.ok is True


class TestTier2SampleDecode:
    """Tier 2: sample decode checks on sequence assets."""

    def test_error_when_frame_unreadable(self, tmp_path: Path):
        """A frame that cv2.imread returns None for must produce an error."""
        clip = _clip(tmp_path)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.disk_usage") as mock_du,
            patch("cv2.imread", return_value=None),
        ):
            mock_du.return_value = MagicMock(free=100 * 1024**3)
            result = validate_job_inputs(clip)
        assert result.ok is False
        assert any("could not decode" in e for e in result.errors)

    def test_ok_when_frames_readable_and_consistent(self, tmp_path: Path):
        """Consistent frames must produce ok=True."""
        clip = _clip(tmp_path)
        fake_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.disk_usage") as mock_du,
            patch("cv2.imread", return_value=fake_frame),
        ):
            mock_du.return_value = MagicMock(free=100 * 1024**3)
            result = validate_job_inputs(clip)
        assert result.ok is True

    def test_warning_when_mixed_dtypes(self, tmp_path: Path):
        """Inconsistent dtypes across sample frames must produce a warning (not error)."""
        clip = _clip(tmp_path, input_frame_count=3, alpha_frame_count=3)
        frames = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint16),
            np.zeros((100, 100, 3), dtype=np.uint8),
        ]
        call_count = [0]

        def fake_imread(path, *args, **kwargs):
            idx = call_count[0] % len(frames)
            call_count[0] += 1
            return frames[idx]

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.disk_usage") as mock_du,
            patch("cv2.imread", side_effect=fake_imread),
        ):
            mock_du.return_value = MagicMock(free=100 * 1024**3)
            result = validate_job_inputs(clip)
        assert any("mixed dtypes" in w for w in result.warnings)

    def test_error_when_inconsistent_resolution(self, tmp_path: Path):
        """Inconsistent frame resolution across sample frames must produce an error."""
        clip = _clip(tmp_path, input_frame_count=3, alpha_frame_count=3)
        frames = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((200, 200, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
        ]
        call_count = [0]

        def fake_imread(path, *args, **kwargs):
            idx = call_count[0] % len(frames)
            call_count[0] += 1
            return frames[idx]

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.disk_usage") as mock_du,
            patch("cv2.imread", side_effect=fake_imread),
        ):
            mock_du.return_value = MagicMock(free=100 * 1024**3)
            result = validate_job_inputs(clip)
        assert result.ok is False
        assert any("inconsistent frame resolution" in e for e in result.errors)

    def test_video_asset_skips_tier2(self, tmp_path: Path):
        """Video assets must skip Tier 2 sample decode (no frame files to decode)."""
        clip = _clip(tmp_path, input_type="video", alpha_type="video")
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("shutil.disk_usage") as mock_du,
        ):
            mock_du.return_value = MagicMock(free=100 * 1024**3)
            result = validate_job_inputs(clip)
        # Video assets skip Tier 1 frame count check and Tier 2 -- should be ok
        assert result.ok is True


class TestValidationResult:
    """ValidationResult -- ok flag semantics."""

    def test_ok_true_when_no_errors(self):
        r = ValidationResult(ok=True, errors=[], warnings=[])
        assert r.ok is True

    def test_ok_false_when_errors(self):
        r = ValidationResult(ok=False, errors=["something wrong"], warnings=[])
        assert r.ok is False
