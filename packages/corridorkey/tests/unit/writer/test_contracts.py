"""Unit tests for corridorkey.stages.writer.contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey.stages.writer.contracts import WriteConfig


class TestWriteConfigDefaults:
    def test_alpha_enabled(self, tmp_path: Path):
        cfg = WriteConfig(output_dir=tmp_path)
        assert cfg.alpha_enabled is True

    def test_fg_enabled(self, tmp_path: Path):
        cfg = WriteConfig(output_dir=tmp_path)
        assert cfg.fg_enabled is True

    def test_comp_enabled(self, tmp_path: Path):
        cfg = WriteConfig(output_dir=tmp_path)
        assert cfg.comp_enabled is True

    def test_default_formats_png(self, tmp_path: Path):
        cfg = WriteConfig(output_dir=tmp_path)
        assert cfg.alpha_format == "png"
        assert cfg.fg_format == "png"
        assert cfg.comp_format == "png"

    def test_default_exr_compression(self, tmp_path: Path):
        cfg = WriteConfig(output_dir=tmp_path)
        assert cfg.exr_compression == "dwaa"

    def test_output_dir_stored(self, tmp_path: Path):
        cfg = WriteConfig(output_dir=tmp_path)
        assert cfg.output_dir == tmp_path

    def test_is_frozen(self, tmp_path: Path):
        cfg = WriteConfig(output_dir=tmp_path)
        with pytest.raises((AttributeError, TypeError, Exception)):
            cfg.alpha_enabled = False  # type: ignore[misc]
