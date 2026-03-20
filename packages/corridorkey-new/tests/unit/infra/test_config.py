"""Unit tests for corridorkey_new.infra.config."""

from __future__ import annotations

import pytest
from corridorkey_new.infra.config import CorridorKeyConfig


class TestCorridorKeyConfigDefaults:
    def test_default_log_level(self):
        config = CorridorKeyConfig()
        assert config.log_level == "INFO"

    def test_default_device(self):
        config = CorridorKeyConfig()
        assert config.device == "auto"

    def test_default_log_dir_contains_corridorkey(self):
        config = CorridorKeyConfig()
        assert "corridorkey" in str(config.log_dir)


class TestCorridorKeyConfigValidation:
    def test_valid_log_levels(self):
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            config = CorridorKeyConfig(log_level=level)
            assert config.log_level == level

    def test_invalid_log_level_raises(self):
        with pytest.raises(Exception, match="log_level"):
            CorridorKeyConfig(log_level="VERBOSE")  # type: ignore[arg-type]

    def test_valid_devices(self):
        for device in ("auto", "cuda", "rocm", "mps", "cpu"):
            config = CorridorKeyConfig(device=device)
            assert config.device == device

    def test_invalid_device_raises(self):
        with pytest.raises(Exception, match="device"):
            CorridorKeyConfig(device="tpu")  # type: ignore[arg-type]


class TestCorridorKeyConfigOverrides:
    def test_override_log_level(self):
        config = CorridorKeyConfig(log_level="DEBUG")
        assert config.log_level == "DEBUG"

    def test_override_device(self):
        config = CorridorKeyConfig(device="cpu")
        assert config.device == "cpu"
