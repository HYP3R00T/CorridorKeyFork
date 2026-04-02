"""Unit tests for corridorkey.stages.inference.factory — discover_checkpoint and backend resolution."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from corridorkey.stages.inference.factory import (
    _auto_detect,
    _mlx_importable,
    _resolve_backend,
    discover_checkpoint,
)


class TestDiscoverCheckpoint:
    def test_finds_single_pth_file(self, tmp_path: Path):
        (tmp_path / "model.pth").touch()
        result = discover_checkpoint(tmp_path, backend="torch")
        assert result == tmp_path / "model.pth"

    def test_finds_single_safetensors_file(self, tmp_path: Path):
        (tmp_path / "model.safetensors").touch()
        result = discover_checkpoint(tmp_path, backend="mlx")
        assert result == tmp_path / "model.safetensors"

    def test_raises_file_not_found_when_empty(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No .pth checkpoint"):
            discover_checkpoint(tmp_path, backend="torch")

    def test_raises_file_not_found_for_mlx_when_empty(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="No .safetensors checkpoint"):
            discover_checkpoint(tmp_path, backend="mlx")

    def test_raises_value_error_when_multiple_pth(self, tmp_path: Path):
        (tmp_path / "a.pth").touch()
        (tmp_path / "b.pth").touch()
        with pytest.raises(ValueError, match="Multiple .pth checkpoints"):
            discover_checkpoint(tmp_path, backend="torch")

    def test_raises_value_error_when_multiple_safetensors(self, tmp_path: Path):
        (tmp_path / "a.safetensors").touch()
        (tmp_path / "b.safetensors").touch()
        with pytest.raises(ValueError, match="Multiple .safetensors checkpoints"):
            discover_checkpoint(tmp_path, backend="mlx")

    def test_hint_shown_when_wrong_backend_files_present(self, tmp_path: Path):
        """When torch is requested but only .safetensors exists, hint mentions mlx."""
        (tmp_path / "model.safetensors").touch()
        with pytest.raises(FileNotFoundError, match="mlx"):
            discover_checkpoint(tmp_path, backend="torch")

    def test_hint_shown_when_pth_present_but_mlx_requested(self, tmp_path: Path):
        """When mlx is requested but only .pth exists, hint mentions torch."""
        (tmp_path / "model.pth").touch()
        with pytest.raises(FileNotFoundError, match="torch"):
            discover_checkpoint(tmp_path, backend="mlx")

    def test_accepts_string_path(self, tmp_path: Path):
        (tmp_path / "model.pth").touch()
        result = discover_checkpoint(str(tmp_path), backend="torch")
        assert result.suffix == ".pth"

    def test_ignores_non_matching_extensions(self, tmp_path: Path):
        """Other file types in the directory are ignored."""
        (tmp_path / "model.pth").touch()
        (tmp_path / "readme.txt").touch()
        (tmp_path / "config.yaml").touch()
        result = discover_checkpoint(tmp_path, backend="torch")
        assert result.name == "model.pth"

    def test_default_backend_is_torch(self, tmp_path: Path):
        (tmp_path / "model.pth").touch()
        result = discover_checkpoint(tmp_path)
        assert result.suffix == ".pth"


class TestResolveBackend:
    def test_torch_returns_torch(self):
        assert _resolve_backend("torch") == "torch"

    def test_auto_on_non_apple_returns_torch(self):
        with patch("sys.platform", "win32"):
            assert _resolve_backend("auto") == "torch"

    def test_env_var_overrides_auto(self):
        with (
            patch.dict(os.environ, {"CORRIDORKEY_BACKEND": "torch"}),
            patch("sys.platform", "linux"),
        ):
            assert _resolve_backend("auto") == "torch"

    def test_unknown_backend_raises(self):
        with pytest.raises(RuntimeError, match="Unknown backend"):
            _resolve_backend("tensorrt")

    def test_mlx_on_non_apple_raises(self):
        with patch("sys.platform", "linux"), pytest.raises(RuntimeError, match="Apple Silicon"):
            _resolve_backend("mlx")


class TestAutoDetect:
    def test_non_darwin_returns_torch(self):
        with patch("sys.platform", "win32"):
            assert _auto_detect() == "torch"

    def test_darwin_non_arm64_returns_torch(self):
        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="x86_64"):
            assert _auto_detect() == "torch"

    def test_darwin_arm64_no_mlx_returns_torch(self):
        with (
            patch("sys.platform", "darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("corridorkey.stages.inference.factory._mlx_importable", return_value=False),
        ):
            assert _auto_detect() == "torch"

    def test_darwin_arm64_with_mlx_returns_mlx(self):
        with (
            patch("sys.platform", "darwin"),
            patch("platform.machine", return_value="arm64"),
            patch("corridorkey.stages.inference.factory._mlx_importable", return_value=True),
        ):
            assert _auto_detect() == "mlx"


class TestMlxImportable:
    def test_returns_false_when_not_installed(self):
        with patch("importlib.util.find_spec", return_value=None):
            assert _mlx_importable() is False

    def test_returns_true_when_installed(self):
        with patch("importlib.util.find_spec", return_value=object()):
            assert _mlx_importable() is True
