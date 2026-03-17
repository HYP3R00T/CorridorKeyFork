"""Unit tests for model_manager.py - is_model_present and download_model.

Tests mock urllib and filesystem so no network access or real files needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from corridorkey.config import CorridorKeyConfig
from corridorkey.model_manager import (
    _sha256,
    download_model,
    is_model_present,
)


def _config(checkpoint_dir: str) -> CorridorKeyConfig:
    return CorridorKeyConfig(checkpoint_dir=Path(checkpoint_dir))


class TestIsModelPresent:
    """is_model_present - checks for .pth or .safetensors in checkpoint_dir."""

    def test_false_when_dir_missing(self, tmp_path: Path):
        cfg = _config(str(tmp_path / "nonexistent"))
        assert is_model_present(cfg) is False

    def test_false_when_dir_empty(self, tmp_path: Path):
        cfg = _config(str(tmp_path))
        assert is_model_present(cfg) is False

    def test_true_when_pth_present(self, tmp_path: Path):
        (tmp_path / "model.pth").touch()
        cfg = _config(str(tmp_path))
        assert is_model_present(cfg) is True

    def test_true_when_safetensors_present(self, tmp_path: Path):
        (tmp_path / "model.safetensors").touch()
        cfg = _config(str(tmp_path))
        assert is_model_present(cfg) is True

    def test_false_when_only_unrelated_files(self, tmp_path: Path):
        (tmp_path / "readme.txt").touch()
        (tmp_path / "config.yaml").touch()
        cfg = _config(str(tmp_path))
        assert is_model_present(cfg) is False


class TestDownloadModelError:
    """download_model - network failure must raise RuntimeError and clean up tmp."""

    def test_raises_on_network_error(self, tmp_path: Path):
        cfg = _config(str(tmp_path))
        with (
            patch("urllib.request.urlopen", side_effect=OSError("connection refused")),
            pytest.raises(RuntimeError, match="Model download failed"),
        ):
            download_model(cfg, url="https://example.com/model.pth", filename="model.pth", checksum="")

    def test_tmp_file_cleaned_up_on_error(self, tmp_path: Path):
        cfg = _config(str(tmp_path))
        with patch("urllib.request.urlopen", side_effect=OSError("timeout")), pytest.raises(RuntimeError):
            download_model(cfg, url="https://example.com/model.pth", filename="model.pth", checksum="")
        # .tmp file must not remain
        assert not list(tmp_path.glob("*.tmp"))

    def test_checksum_mismatch_raises_and_removes_tmp(self, tmp_path: Path):
        cfg = _config(str(tmp_path))

        # Simulate a successful download that writes some bytes
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.headers = {"Content-Length": "5"}
        mock_response.read.side_effect = [b"hello", b""]

        with (
            patch("urllib.request.urlopen", return_value=mock_response),
            pytest.raises(RuntimeError, match="Checksum mismatch"),
        ):
            download_model(
                cfg,
                url="https://example.com/model.pth",
                filename="model.pth",
                checksum="0000000000000000000000000000000000000000000000000000000000000000",
            )
        assert not list(tmp_path.glob("*.tmp"))

    def test_checksum_skipped_when_empty(self, tmp_path: Path):
        cfg = _config(str(tmp_path))

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.headers = {"Content-Length": "5"}
        mock_response.read.side_effect = [b"hello", b""]

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = download_model(
                cfg,
                url="https://example.com/model.pth",
                filename="model.pth",
                checksum="",  # skip verification
            )
        assert result.name == "model.pth"
        assert result.exists()

    def test_on_progress_called(self, tmp_path: Path):
        cfg = _config(str(tmp_path))
        calls: list[tuple[int, int]] = []

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.headers = {"Content-Length": "10"}
        mock_response.read.side_effect = [b"hello", b"world", b""]

        with patch("urllib.request.urlopen", return_value=mock_response):
            download_model(
                cfg,
                on_progress=lambda d, t: calls.append((d, t)),
                url="https://example.com/model.pth",
                filename="model.pth",
                checksum="",
            )
        assert len(calls) == 2
        assert calls[0] == (5, 10)
        assert calls[1] == (10, 10)

    def test_creates_checkpoint_dir_if_missing(self, tmp_path: Path):
        new_dir = tmp_path / "deep" / "nested"
        cfg = _config(str(new_dir))

        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.headers = {}
        mock_response.read.side_effect = [b"data", b""]

        with patch("urllib.request.urlopen", return_value=mock_response):
            download_model(cfg, url="https://example.com/model.pth", filename="model.pth", checksum="")
        assert new_dir.is_dir()


class TestSha256:
    """_sha256 - computes correct hex digest."""

    def test_known_digest(self, tmp_path: Path):
        import hashlib

        data = b"corridorkey test data"
        expected = hashlib.sha256(data).hexdigest()
        f = tmp_path / "test.bin"
        f.write_bytes(data)
        assert _sha256(f) == expected

    def test_returns_lowercase_hex(self, tmp_path: Path):
        f = tmp_path / "test.bin"
        f.write_bytes(b"abc")
        result = _sha256(f)
        assert result == result.lower()
        assert all(c in "0123456789abcdef" for c in result)
