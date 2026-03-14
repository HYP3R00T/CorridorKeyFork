"""Unit tests for config.py.

CorridorKeyConfig is the single source of truth for all runtime settings.
export_config writes it to TOML so users can inspect and edit their config.
Tests verify field defaults, path handling, and that the exported TOML is
valid and complete. load_config is not tested here because it depends on
utilityhub_config and the filesystem - those paths are covered by the
integration surface.
"""

from __future__ import annotations

from pathlib import Path

from corridorkey.config import CorridorKeyConfig, export_config


def _config(tmp_path: Path, **kwargs) -> CorridorKeyConfig:
    """Build a CorridorKeyConfig with tmp_path-based dirs that exist on disk."""
    app_dir = tmp_path / "app"
    checkpoint_dir = tmp_path / "models"
    app_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return CorridorKeyConfig(app_dir=app_dir, checkpoint_dir=checkpoint_dir, **kwargs)


class TestCorridorKeyConfig:
    """Field defaults and override behaviour - the config must be predictable out of the box."""

    def test_defaults(self):
        """All fields must have the documented default values so new installs work without a config file."""
        config = CorridorKeyConfig()
        assert config.device == "auto"
        assert config.despill_strength == 1.0
        assert config.auto_despeckle is True
        assert config.despeckle_size == 400
        assert config.refiner_scale == 1.0
        assert config.input_is_linear is False
        assert config.fg_format == "exr"
        assert config.matte_format == "exr"
        assert config.comp_format == "png"
        assert config.processed_format == "png"

    def test_path_fields_are_path_objects(self, tmp_path: Path):
        """Path fields must be Path objects, not strings, so callers can use / operator."""
        config = _config(tmp_path)
        assert isinstance(config.app_dir, Path)
        assert isinstance(config.checkpoint_dir, Path)

    def test_override_device(self, tmp_path: Path):
        """Passing device= must override the default without affecting other fields."""
        config = _config(tmp_path, device="cpu")
        assert config.device == "cpu"

    def test_override_despill_strength(self, tmp_path: Path):
        """despill_strength must accept any float in [0, 1]."""
        config = _config(tmp_path, despill_strength=0.5)
        assert config.despill_strength == 0.5

    def test_path_fields_resolve_to_provided_dirs(self, tmp_path: Path):
        """Provided path values must be stored as-is after tilde expansion."""
        config = _config(tmp_path)
        assert config.app_dir == tmp_path / "app"
        assert config.checkpoint_dir == tmp_path / "models"


class TestExportConfig:
    """export_config() - TOML serialisation correctness and file placement."""

    def test_writes_toml_file(self, tmp_path: Path):
        """The exported file must exist and contain key field names."""
        config = _config(tmp_path)
        dest = export_config(config, path=tmp_path / "out.toml")
        assert dest.exists()
        content = dest.read_text()
        assert "device" in content
        assert "despill_strength" in content

    def test_default_path_is_inside_app_dir(self, tmp_path: Path):
        """Without an explicit path, the file must land inside app_dir for auto-discovery."""
        config = _config(tmp_path)
        dest = export_config(config)
        assert dest.parent == config.app_dir

    def test_bool_written_as_lowercase(self, tmp_path: Path):
        """TOML booleans must be lowercase true/false, not Python True/False."""
        config = _config(tmp_path, auto_despeckle=True, input_is_linear=False)
        dest = export_config(config, path=tmp_path / "out.toml")
        content = dest.read_text()
        assert "true" in content
        assert "false" in content
        assert "True" not in content
        assert "False" not in content

    def test_string_values_quoted(self, tmp_path: Path):
        """String values must be quoted in TOML format."""
        config = _config(tmp_path, device="cpu")
        dest = export_config(config, path=tmp_path / "out.toml")
        content = dest.read_text()
        assert 'device = "cpu"' in content

    def test_creates_parent_dirs(self, tmp_path: Path):
        """export_config must create any missing parent directories."""
        config = _config(tmp_path)
        dest = export_config(config, path=tmp_path / "nested" / "deep" / "out.toml")
        assert dest.exists()

    def test_roundtrip_contains_all_fields(self, tmp_path: Path):
        """Every model field must appear in the exported file - nothing silently omitted."""
        config = _config(tmp_path)
        dest = export_config(config, path=tmp_path / "out.toml")
        content = dest.read_text()
        for field_name in config.__class__.model_fields:
            assert field_name in content
