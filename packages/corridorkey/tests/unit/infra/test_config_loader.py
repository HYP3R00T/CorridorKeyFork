"""Unit tests for corridorkey.infra.config._loader."""

from __future__ import annotations

import os
from pathlib import Path

from corridorkey.infra.config._loader import APP_NAME, load_config, load_config_with_metadata
from corridorkey.infra.config.pipeline import CorridorKeyConfig


class TestAppName:
    def test_app_name_is_corridorkey(self):
        assert APP_NAME == "corridorkey"


class TestLoadConfigReturnsConfig:
    def test_returns_corridorkey_config(self):
        cfg = load_config()
        assert isinstance(cfg, CorridorKeyConfig)

    def test_default_device_is_auto(self):
        cfg = load_config()
        assert cfg.device == "auto"

    def test_overrides_applied(self):
        cfg = load_config(overrides={"device": "cpu"})
        assert cfg.device == "cpu"

    def test_nested_overrides_applied(self):
        cfg = load_config(overrides={"preprocess": {"img_size": 512}})
        assert cfg.preprocess.img_size == 512

    def test_env_vars_not_loaded(self):
        """env_vars=False — CK_* env vars must not affect config."""
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"CK_DEVICE": "cuda"}):
            cfg = load_config()
            assert cfg.device == "auto"

    def test_config_file_parameter(self, tmp_path: Path):
        """Explicit config_file path is loaded instead of auto-discovery."""
        toml = tmp_path / "custom.toml"
        toml.write_text('device = "cpu"\n', encoding="utf-8")
        cfg = load_config(config_file=toml)
        assert cfg.device == "cpu"


class TestLoadConfigWithMetadata:
    def test_returns_tuple(self):
        result = load_config_with_metadata()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_config(self):
        cfg, _ = load_config_with_metadata()
        assert isinstance(cfg, CorridorKeyConfig)

    def test_second_element_is_settings_metadata(self):
        from utilityhub_config.metadata import SettingsMetadata

        _, metadata = load_config_with_metadata()
        assert isinstance(metadata, SettingsMetadata)

    def test_metadata_has_per_field(self):
        _, metadata = load_config_with_metadata()
        assert hasattr(metadata, "per_field")
        assert isinstance(metadata.per_field, dict)

    def test_overrides_applied(self):
        cfg, _ = load_config_with_metadata(overrides={"device": "cpu"})
        assert cfg.device == "cpu"


class TestLoadConfigFromToml:
    def test_toml_file_overrides_defaults(self, tmp_path: Path):
        toml_content = 'device = "cpu"\n'
        config_file = tmp_path / "corridorkey.toml"
        config_file.write_text(toml_content, encoding="utf-8")

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            cfg = load_config()
            assert cfg.device == "cpu"
        finally:
            os.chdir(original_cwd)

    def test_toml_nested_settings_loaded(self, tmp_path: Path):
        toml_content = "[preprocess]\nimg_size = 1024\n"
        config_file = tmp_path / "corridorkey.toml"
        config_file.write_text(toml_content, encoding="utf-8")

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            cfg = load_config()
            assert cfg.preprocess.img_size == 1024
        finally:
            os.chdir(original_cwd)


class TestLoadConfigLoggingDirCreated:
    def test_logging_dir_created_on_load(self, tmp_path: Path):
        log_dir = tmp_path / "logs" / "subdir"
        assert not log_dir.exists()
        load_config(overrides={"logging": {"dir": str(log_dir), "level": "INFO"}})
        assert log_dir.exists()
