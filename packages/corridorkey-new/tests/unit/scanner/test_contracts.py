"""Unit tests for corridorkey_new.stages.scanner.contracts — Clip."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey_new.stages.scanner.contracts import Clip
from pydantic import ValidationError


class TestClip:
    def test_valid_clip_without_alpha(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        assert clip.alpha_path is None

    def test_valid_clip_with_alpha(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        alpha_dir = tmp_path / "AlphaHint"
        input_dir.mkdir()
        alpha_dir.mkdir()
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=alpha_dir)
        assert clip.alpha_path == alpha_dir

    def test_nonexistent_root_raises(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        with pytest.raises(ValidationError, match="does not exist"):
            Clip(name="test", root=tmp_path / "ghost", input_path=input_dir, alpha_path=None)

    def test_nonexistent_input_raises(self, tmp_path: Path):
        with pytest.raises(ValidationError, match="does not exist"):
            Clip(name="test", root=tmp_path, input_path=tmp_path / "Input", alpha_path=None)

    def test_root_must_be_directory(self, tmp_path: Path):
        f = tmp_path / "file.txt"
        f.touch()
        with pytest.raises(ValidationError):
            Clip(name="test", root=f, input_path=f, alpha_path=None)

    def test_repr_contains_name(self, tmp_path: Path):
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        clip = Clip(name="my_clip", root=tmp_path, input_path=input_dir, alpha_path=None)
        assert "my_clip" in repr(clip)
