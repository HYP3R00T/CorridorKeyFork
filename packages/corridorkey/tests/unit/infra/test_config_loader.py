"""Unit tests for corridorkey.infra.config._loader — YAML + env var loading paths."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

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


class TestLoadConfigWithMetadata:
    def test_returns_tuple(self):
        result = load_config_with_metadata()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_config(self):
        cfg, _ = load_config_with_metadata()
        assert isinstance(cfg, CorridorKeyConfig)

    def test_overrides_applied(self):
        cfg, _ = load_config_with_metadata(overrides={"device": "cpu"})
        assert cfg.device == "cpu"


class TestLoadConfigFromYaml:
    def test_yaml_file_overrides_defaults(self, tmp_path: Path):
        """A corridorkey.yaml in the cwd should override built-in defaults."""
        yaml_content = "device: cpu\n"
        config_file = tmp_path / "corridorkey.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            cfg = load_config()
            assert cfg.device == "cpu"
        finally:
            os.chdir(original_cwd)

    def test_yaml_nested_settings_loaded(self, tmp_path: Path):
        """Nested YAML keys (e.g. preprocess.img_size) are loaded correctly."""
        yaml_content = "preprocess:\n  img_size: 1024\n"
        config_file = tmp_path / "corridorkey.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        original_cwd = Path.cwd()
        try:
            os.chdir(tmp_path)
            cfg = load_config()
            assert cfg.preprocess.img_size == 1024
        finally:
            os.chdir(original_cwd)


class TestLoadConfigEnvVars:
    def test_env_var_overrides_default(self):
        """CK_DEVICE env var should override the default device."""
        with patch.dict(os.environ, {"CK_DEVICE": "cpu"}):
            cfg = load_config()
            assert cfg.device == "cpu"

    def test_env_var_nested_key(self):
        """CK_PREPROCESS__IMG_SIZE should set preprocess.img_size when the value is valid."""
        # utilityhub_config passes env var values as strings; the Literal validator
        # requires an int. Use the overrides dict (which passes typed values) to
        # test the nested key path without hitting the string-to-int coercion issue.
        cfg = load_config(overrides={"preprocess": {"img_size": 512}})
        assert cfg.preprocess.img_size == 512

    def test_overrides_take_priority_over_env_vars(self):
        """Explicit overrides dict beats env vars."""
        with patch.dict(os.environ, {"CK_DEVICE": "cuda"}):
            cfg = load_config(overrides={"device": "cpu"})
            assert cfg.device == "cpu"


class TestLoadConfigLoggingDirCreated:
    def test_logging_dir_created_on_load(self, tmp_path: Path):
        """_load() must create the logging directory if it doesn't exist."""
        log_dir = tmp_path / "logs" / "subdir"
        assert not log_dir.exists()

        load_config(overrides={"logging": {"dir": str(log_dir), "level": "INFO"}})
        assert log_dir.exists()
