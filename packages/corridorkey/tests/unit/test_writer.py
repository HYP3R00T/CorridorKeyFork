"""Unit tests for processing/writer.py - write_outputs and generate_masks.

write_outputs writes all enabled output images for one processed frame.
Tests use tmp_path and synthetic numpy arrays - no GPU, no model files.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from corridorkey.contracts import WriteConfig
from corridorkey.writer import generate_masks, write_outputs
from corridorkey_core.contracts import ProcessedFrame


def _frame(h: int = 16, w: int = 16, stem: str = "frame_00001") -> ProcessedFrame:
    return ProcessedFrame(
        alpha=np.random.rand(h, w, 1).astype(np.float32),
        fg=np.random.rand(h, w, 3).astype(np.float32),
        comp=np.random.rand(h, w, 3).astype(np.float32),
        processed=np.random.rand(h, w, 4).astype(np.float32),
        source_h=h,
        source_w=w,
        stem=stem,
    )


def _dirs(tmp_path: Path) -> dict[str, str]:
    dirs = {}
    for name in ("fg", "matte", "comp", "processed"):
        d = tmp_path / name
        d.mkdir()
        dirs[name] = str(d)
    return dirs


def _cfg(
    tmp_path: Path,
    fg_enabled: bool = True,
    fg_format: str = "png",
    matte_enabled: bool = True,
    matte_format: str = "png",
    comp_enabled: bool = True,
    comp_format: str = "png",
    processed_enabled: bool = True,
    processed_format: str = "png",
    exr_compression: str = "dwaa",
    dirs: dict[str, str] | None = None,
) -> WriteConfig:
    return WriteConfig(
        fg_enabled=fg_enabled,
        fg_format=fg_format,
        matte_enabled=matte_enabled,
        matte_format=matte_format,
        comp_enabled=comp_enabled,
        comp_format=comp_format,
        processed_enabled=processed_enabled,
        processed_format=processed_format,
        exr_compression=exr_compression,
        dirs=dirs if dirs is not None else _dirs(tmp_path),
    )


class TestWriteOutputsPng:
    """write_outputs with PNG format - all four outputs written to disk."""

    def test_fg_written(self, tmp_path: Path):
        """FG PNG must be written to the fg directory."""
        cfg = _cfg(tmp_path)
        write_outputs(_frame(), cfg)
        assert (Path(cfg.dirs["fg"]) / "frame_00001.png").exists()

    def test_matte_written(self, tmp_path: Path):
        """Matte PNG must be written to the matte directory."""
        cfg = _cfg(tmp_path)
        write_outputs(_frame(), cfg)
        assert (Path(cfg.dirs["matte"]) / "frame_00001.png").exists()

    def test_comp_written(self, tmp_path: Path):
        """Comp PNG must be written to the comp directory."""
        cfg = _cfg(tmp_path)
        write_outputs(_frame(), cfg)
        assert (Path(cfg.dirs["comp"]) / "frame_00001.png").exists()

    def test_processed_written(self, tmp_path: Path):
        """Processed PNG must be written to the processed directory."""
        cfg = _cfg(tmp_path)
        write_outputs(_frame(), cfg)
        assert (Path(cfg.dirs["processed"]) / "frame_00001.png").exists()

    def test_stem_used_as_filename(self, tmp_path: Path):
        """The frame stem must be used as the output filename."""
        cfg = _cfg(tmp_path)
        write_outputs(_frame(stem="shot_042"), cfg)
        assert (Path(cfg.dirs["fg"]) / "shot_042.png").exists()

    def test_fg_disabled_not_written(self, tmp_path: Path):
        """When fg_enabled=False, no FG file must be written."""
        cfg = _cfg(tmp_path, fg_enabled=False)
        write_outputs(_frame(), cfg)
        assert not list(Path(cfg.dirs["fg"]).iterdir())

    def test_matte_disabled_not_written(self, tmp_path: Path):
        """When matte_enabled=False, no matte file must be written."""
        cfg = _cfg(tmp_path, matte_enabled=False)
        write_outputs(_frame(), cfg)
        assert not list(Path(cfg.dirs["matte"]).iterdir())

    def test_comp_disabled_not_written(self, tmp_path: Path):
        """When comp_enabled=False, no comp file must be written."""
        cfg = _cfg(tmp_path, comp_enabled=False)
        write_outputs(_frame(), cfg)
        assert not list(Path(cfg.dirs["comp"]).iterdir())

    def test_processed_disabled_not_written(self, tmp_path: Path):
        """When processed_enabled=False, no processed file must be written."""
        cfg = _cfg(tmp_path, processed_enabled=False)
        write_outputs(_frame(), cfg)
        assert not list(Path(cfg.dirs["processed"]).iterdir())

    def test_all_disabled_writes_nothing(self, tmp_path: Path):
        """With all outputs disabled, no files must be written."""
        cfg = _cfg(
            tmp_path,
            fg_enabled=False,
            matte_enabled=False,
            comp_enabled=False,
            processed_enabled=False,
        )
        write_outputs(_frame(), cfg)
        for d in cfg.dirs.values():
            assert not list(Path(d).iterdir())

    def test_fg_png_is_readable(self, tmp_path: Path):
        """Written FG PNG must be readable by cv2 with correct shape."""
        cfg = _cfg(tmp_path)
        frame = _frame(32, 32)
        write_outputs(frame, cfg)
        img = cv2.imread(str(Path(cfg.dirs["fg"]) / "frame_00001.png"))
        assert img is not None
        assert img.shape == (32, 32, 3)

    def test_matte_png_is_single_channel(self, tmp_path: Path):
        """Written matte PNG must be a single-channel grayscale image."""
        cfg = _cfg(tmp_path)
        write_outputs(_frame(32, 32), cfg)
        img = cv2.imread(str(Path(cfg.dirs["matte"]) / "frame_00001.png"), cv2.IMREAD_GRAYSCALE)
        assert img is not None
        assert img.ndim == 2


class TestWriteOutputsExr:
    """write_outputs with EXR format - verifies the write path is exercised.

    EXR support requires OPENCV_IO_ENABLE_OPENEXR=1 at import time. In CI
    environments where that env var is not set we mock cv2.imwrite so the
    code path is still covered without needing a real EXR codec.
    """

    def _write_with_mock(self, cfg):
        from unittest.mock import patch

        with patch("cv2.imwrite", return_value=True):
            write_outputs(_frame(), cfg)

    def test_fg_exr_path_exercised(self, tmp_path: Path):
        """FG EXR write path must be exercised without raising."""
        cfg = _cfg(tmp_path, fg_format="exr")
        self._write_with_mock(cfg)

    def test_matte_exr_path_exercised(self, tmp_path: Path):
        """Matte EXR write path must be exercised without raising."""
        cfg = _cfg(tmp_path, matte_format="exr")
        self._write_with_mock(cfg)

    def test_processed_exr_path_exercised(self, tmp_path: Path):
        """Processed EXR write path must be exercised without raising."""
        cfg = _cfg(tmp_path, processed_format="exr")
        self._write_with_mock(cfg)


class TestWriteOutputsMissingDir:
    """write_outputs with a missing dir key must skip that output silently."""

    def test_missing_fg_dir_skips_fg(self, tmp_path: Path):
        """If 'fg' is not in dirs, fg output must be silently skipped."""
        dirs = _dirs(tmp_path)
        del dirs["fg"]
        cfg = WriteConfig(
            fg_enabled=True,
            fg_format="png",
            matte_enabled=False,
            comp_enabled=False,
            processed_enabled=False,
            dirs=dirs,
        )
        write_outputs(_frame(), cfg)  # must not raise


class TestGenerateMasks:
    """generate_masks - always raises NotImplementedError (stage 2 placeholder)."""

    def test_raises_without_generator(self, tmp_path: Path):
        """Calling generate_masks with no generator must raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="no generator"):
            generate_masks(str(tmp_path), str(tmp_path))

    def test_raises_with_generator(self, tmp_path: Path):
        """Calling generate_masks with a generator must still raise NotImplementedError."""
        from unittest.mock import MagicMock

        gen = MagicMock()
        with pytest.raises(NotImplementedError, match="not yet wired"):
            generate_masks(str(tmp_path), str(tmp_path), generator=gen)
