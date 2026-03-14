"""Tests for corridorkey_core.engine_factory.

Backend resolution and checkpoint discovery are the two entry points that
every consumer calls before any inference happens. Getting them wrong means
either loading the wrong backend silently or crashing with an unhelpful
error. Tests here cover all resolution paths (explicit, auto-detect, env var)
and all checkpoint discovery edge cases (missing, ambiguous, wrong extension)
without requiring a GPU or model files.

All tests run in the fast suite.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from corridorkey_core.engine_factory import (
    BACKEND_ENV_VAR,
    MLX_EXT,
    TORCH_EXT,
    discover_checkpoint,
    resolve_backend,
)


class TestResolveBackend:
    """Backend selection logic - explicit, auto-detect, env var, and error cases.

    resolve_backend is called once at startup. A wrong result here means the
    wrong inference engine is loaded, which either crashes or silently produces
    incorrect output on the wrong hardware.
    """

    def test_none_triggers_auto_detect(self):
        """None must trigger auto-detection, which returns torch on non-Apple CI."""
        with patch("corridorkey_core.engine_factory.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = resolve_backend(None)
        assert result == "torch"

    def test_explicit_torch(self):
        """Explicit "torch" must always return "torch" regardless of platform."""
        result = resolve_backend("torch")
        assert result == "torch"

    def test_explicit_auto_triggers_auto_detect(self):
        """Explicit "auto" must behave identically to None."""
        with patch("corridorkey_core.engine_factory.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = resolve_backend("auto")
        assert result == "torch"

    def test_case_insensitive(self):
        """Backend names must be accepted in any case."""
        result = resolve_backend("TORCH")
        assert result == "torch"

    def test_unknown_backend_raises(self):
        """An unrecognised backend name must raise RuntimeError immediately."""
        with pytest.raises(RuntimeError, match="Unknown backend"):
            resolve_backend("tpu")

    def test_env_var_torch(self, monkeypatch):
        """The env var must be respected when no explicit backend is given."""
        monkeypatch.setenv(BACKEND_ENV_VAR, "torch")
        result = resolve_backend(None)
        assert result == "torch"

    def test_explicit_arg_overrides_env_var(self, monkeypatch):
        """An explicit argument must take priority over the env var."""
        monkeypatch.setenv(BACKEND_ENV_VAR, "mlx")
        result = resolve_backend("torch")
        assert result == "torch"

    def test_mlx_on_non_apple_raises(self):
        """Requesting MLX on non-Apple hardware must raise with a clear message."""
        with (
            patch("corridorkey_core.engine_factory.sys") as mock_sys,
            patch("corridorkey_core.engine_factory.platform") as mock_platform,
        ):
            mock_sys.platform = "linux"
            mock_platform.machine.return_value = "x86_64"
            with pytest.raises(RuntimeError, match="Apple Silicon"):
                resolve_backend("mlx")

    def test_mlx_on_apple_without_package_raises(self):
        """MLX on Apple Silicon without corridorkey_mlx installed must raise."""
        with (
            patch("corridorkey_core.engine_factory.sys") as mock_sys,
            patch("corridorkey_core.engine_factory.platform") as mock_platform,
            patch("corridorkey_core.engine_factory._mlx_available", return_value=False),
        ):
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "arm64"
            with pytest.raises(RuntimeError, match="corridorkey_mlx is not installed"):
                resolve_backend("mlx")

    def test_mlx_on_apple_with_package_returns_mlx(self):
        """MLX on Apple Silicon with the package installed must return "mlx"."""
        with (
            patch("corridorkey_core.engine_factory.sys") as mock_sys,
            patch("corridorkey_core.engine_factory.platform") as mock_platform,
            patch("corridorkey_core.engine_factory._mlx_available", return_value=True),
        ):
            mock_sys.platform = "darwin"
            mock_platform.machine.return_value = "arm64"
            result = resolve_backend("mlx")
        assert result == "mlx"


class TestDiscoverCheckpoint:
    """Checkpoint file discovery - finds, rejects ambiguous, and gives helpful errors.

    discover_checkpoint is called at engine load time. If it silently picks
    the wrong file or gives a cryptic error, debugging becomes very hard.
    The hint about the wrong extension (e.g. "did you mean mlx?") is
    specifically tested because it saves significant debugging time.
    """

    def test_finds_single_pth(self, tmp_path: Path):
        """A single .pth file must be found and returned."""
        ckpt = tmp_path / "model.pth"
        ckpt.touch()
        result = discover_checkpoint(tmp_path, TORCH_EXT)
        assert result == ckpt

    def test_finds_single_safetensors(self, tmp_path: Path):
        """A single .safetensors file must be found and returned."""
        ckpt = tmp_path / "model.safetensors"
        ckpt.touch()
        result = discover_checkpoint(tmp_path, MLX_EXT)
        assert result == ckpt

    def test_accepts_str_path(self, tmp_path: Path):
        """A string path must be accepted in addition to a Path object."""
        ckpt = tmp_path / "model.pth"
        ckpt.touch()
        result = discover_checkpoint(str(tmp_path), TORCH_EXT)
        assert result == ckpt

    def test_returns_path_object(self, tmp_path: Path):
        """The return type must always be a Path, not a string."""
        (tmp_path / "model.pth").touch()
        result = discover_checkpoint(tmp_path, TORCH_EXT)
        assert isinstance(result, Path)

    def test_no_match_raises_file_not_found(self, tmp_path: Path):
        """An empty directory must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            discover_checkpoint(tmp_path, TORCH_EXT)

    def test_multiple_matches_raises_value_error(self, tmp_path: Path):
        """Multiple matching files must raise ValueError to avoid silent wrong picks."""
        (tmp_path / "model_a.pth").touch()
        (tmp_path / "model_b.pth").touch()
        with pytest.raises(ValueError, match="Multiple"):
            discover_checkpoint(tmp_path, TORCH_EXT)

    def test_wrong_ext_hint_suggests_correct_backend(self, tmp_path: Path):
        """When the wrong extension is requested, the error must hint at the right backend."""
        (tmp_path / "model.safetensors").touch()
        with pytest.raises(FileNotFoundError, match="mlx"):
            discover_checkpoint(tmp_path, TORCH_EXT)

    def test_ignores_files_with_different_extension(self, tmp_path: Path):
        """Files with non-matching extensions must be ignored."""
        (tmp_path / "model.bin").touch()
        (tmp_path / "model.pth").touch()
        result = discover_checkpoint(tmp_path, TORCH_EXT)
        assert result.suffix == TORCH_EXT
