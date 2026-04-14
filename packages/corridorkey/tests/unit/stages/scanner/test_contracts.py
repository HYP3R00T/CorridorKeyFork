"""Unit tests for corridorkey.stages.scanner.contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey.stages.scanner.contracts import Clip, ScanResult, SkippedClip
from pydantic import ValidationError


class TestClip:
    def test_valid_clip_without_alpha(self, tmp_path: Path):
        """Creating a Clip with no alpha path succeeds and alpha_path is None."""
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        assert clip.alpha_path is None

    def test_valid_clip_with_alpha(self, tmp_path: Path):
        """Creating a Clip with an existing alpha directory succeeds and stores the path."""
        input_dir = tmp_path / "Input"
        alpha_dir = tmp_path / "AlphaHint"
        input_dir.mkdir()
        alpha_dir.mkdir()
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=alpha_dir)
        assert clip.alpha_path == alpha_dir

    def test_nonexistent_root_raises(self, tmp_path: Path):
        """Creating a Clip with a root path that does not exist raises ValidationError."""
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        with pytest.raises(ValidationError, match="does not exist"):
            Clip(name="test", root=tmp_path / "ghost", input_path=input_dir, alpha_path=None)

    def test_nonexistent_input_raises(self, tmp_path: Path):
        """Creating a Clip with an input_path that does not exist raises ValidationError."""
        with pytest.raises(ValidationError, match="does not exist"):
            Clip(name="test", root=tmp_path, input_path=tmp_path / "Input", alpha_path=None)

    def test_root_must_be_directory(self, tmp_path: Path):
        """Creating a Clip with a file as root raises ValidationError."""
        f = tmp_path / "file.txt"
        f.touch()
        with pytest.raises(ValidationError):
            Clip(name="test", root=f, input_path=f, alpha_path=None)

    def test_repr_contains_name(self, tmp_path: Path):
        """The repr of a Clip includes the clip name."""
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        clip = Clip(name="my_clip", root=tmp_path, input_path=input_dir, alpha_path=None)
        assert "my_clip" in repr(clip)

    def test_clip_is_frozen(self, tmp_path: Path):
        """Clip must be immutable — mutation should raise."""
        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        with pytest.raises(ValidationError):
            clip.name = "other"


class TestSkippedPath:
    def test_basic_construction(self, tmp_path: Path):
        """Creating a SkippedClip with a path and reason stores both values correctly."""
        s = SkippedClip(path=tmp_path, reason="test reason")
        assert s.path == tmp_path
        assert s.reason == "test reason"

    def test_repr_contains_reason(self, tmp_path: Path):
        """The repr of a SkippedClip includes the skip reason."""
        s = SkippedClip(path=tmp_path, reason="multiple videos")
        assert "multiple videos" in repr(s)

    def test_is_frozen(self, tmp_path: Path):
        """SkippedClip is immutable — attempting to mutate it raises ValidationError."""
        s = SkippedClip(path=tmp_path, reason="x")
        with pytest.raises(ValidationError):
            s.reason = "y"


class TestScanResult:
    def _make_clip(self, root: Path) -> Clip:
        (root / "Input").mkdir(parents=True, exist_ok=True)
        return Clip(name=root.name, root=root, input_path=root / "Input", alpha_path=None)

    def test_empty_result(self):
        """Creating a ScanResult with no clips or skipped entries reports zero counts."""
        result = ScanResult(clips=(), skipped=())
        assert result.clip_count == 0
        assert result.skipped_count == 0

    def test_clip_count(self, tmp_path: Path):
        """clip_count returns the number of clips in the ScanResult."""
        clips = tuple(self._make_clip(tmp_path / f"clip_{i}") for i in range(3))
        result = ScanResult(clips=clips, skipped=())
        assert result.clip_count == 3

    def test_skipped_count(self, tmp_path: Path):
        """skipped_count returns the number of skipped entries in the ScanResult."""
        skipped = (
            SkippedClip(path=tmp_path / "a", reason="x"),
            SkippedClip(path=tmp_path / "b", reason="y"),
        )
        result = ScanResult(clips=(), skipped=skipped)
        assert result.skipped_count == 2

    def test_repr(self, tmp_path: Path):
        """The repr of a ScanResult includes the clip and skipped counts."""
        clip = self._make_clip(tmp_path / "c")
        result = ScanResult(clips=(clip,), skipped=())
        assert "clips=1" in repr(result)
        assert "skipped=0" in repr(result)

    def test_is_frozen(self, tmp_path: Path):
        """ScanResult is immutable — attempting to mutate it raises ValidationError."""
        result = ScanResult(clips=(), skipped=())
        with pytest.raises(ValidationError):
            result.clips = ()
