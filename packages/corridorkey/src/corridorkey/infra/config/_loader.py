"""Config loading utilities — wraps utilityhub_config.load_settings."""

from __future__ import annotations

import logging
from pathlib import Path

from utilityhub_config import load_settings
from utilityhub_config.metadata import SettingsMetadata

from corridorkey.infra.config.pipeline import CorridorKeyConfig

logger = logging.getLogger(__name__)

APP_NAME = "corridorkey"


def _load(overrides: dict | None) -> tuple[CorridorKeyConfig, SettingsMetadata]:
    config, metadata = load_settings(
        CorridorKeyConfig,
        app_name=APP_NAME,
        env_prefix="CK",
        overrides=overrides,
    )
    Path(config.logging.dir).expanduser().mkdir(parents=True, exist_ok=True)
    logger.debug("Config loaded: %s", config.model_dump())
    return config, metadata


def load_config(overrides: dict | None = None) -> CorridorKeyConfig:
    """Load and validate CorridorKey configuration from all sources.

    Resolution order (lowest to highest priority):
        defaults < global config < project config < env vars < overrides
    """
    config, _ = _load(overrides)
    return config


def load_config_with_metadata(overrides: dict | None = None) -> tuple[CorridorKeyConfig, SettingsMetadata]:
    """Load configuration and return per-field source attribution alongside it.

    Identical to :func:`load_config` but also returns a
    :class:`~utilityhub_config.metadata.SettingsMetadata` object that records
    where each field value came from (defaults, global config file, project
    config file, or environment variable).

    Use this when you need to display configuration provenance to the user —
    for example, a "show config" command that annotates each value with its
    source, or a settings UI that highlights overridden fields.

    Args:
        overrides: Optional dict of field overrides applied at highest priority.

    Returns:
        A ``(config, metadata)`` tuple. ``metadata.per_field`` maps field names
        to :class:`~utilityhub_config.metadata.FieldSource` objects. Use
        ``metadata.get_source("field_name")`` to look up a specific field.

    Example::

        config, metadata = load_config_with_metadata()
        src = metadata.get_source("device")
        print(f"device={config.device!r} (from {src.source})")
    """
    return _load(overrides)
