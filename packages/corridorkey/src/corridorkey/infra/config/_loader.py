"""Config loading utilities — wraps utilityhub_config.load_settings."""

from __future__ import annotations

import logging
from pathlib import Path

from utilityhub_config import load_settings
from utilityhub_config.metadata import SettingsMetadata

from corridorkey.infra.config.pipeline import CorridorKeyConfig

logger = logging.getLogger(__name__)

APP_NAME = "corridorkey"


def load_config(
    overrides: dict | None = None,
    config_file: Path | None = None,
) -> CorridorKeyConfig:
    """Load and validate CorridorKey configuration.

    Resolution order (lowest to highest priority):
        defaults → global config → project config → overrides

    Args:
        overrides: Runtime overrides using dot-notation paths, e.g.
            ``{"device": "cuda:1", "preprocess.img_size": 1024}``.
            Applied after all file sources, before validation.
        config_file: Explicit path to a project config file. If provided,
            skips auto-discovery of ``corridorkey.toml`` in the working
            directory and loads this file instead.

    Returns:
        Validated CorridorKeyConfig.
    """
    config, _ = load_config_with_metadata(overrides=overrides, config_file=config_file)
    return config


def load_config_with_metadata(
    overrides: dict | None = None,
    config_file: Path | None = None,
) -> tuple[CorridorKeyConfig, SettingsMetadata]:
    """Load configuration and return per-field source attribution alongside it.

    Same as :func:`load_config` but also returns a
    :class:`~utilityhub_config.metadata.SettingsMetadata` object that records
    where each field value came from. Use this when building a settings UI
    that shows the user which values are defaults vs explicitly set.

    Args:
        overrides: Runtime overrides using dot-notation paths.
        config_file: Explicit path to a project config file.

    Returns:
        ``(config, metadata)`` tuple. Use ``metadata.get_source("field")``
        to look up where a specific value came from.
    """
    config, metadata = load_settings(
        CorridorKeyConfig,
        app_name=APP_NAME,
        env_vars=False,
        overrides=overrides,
        config_file=config_file,
    )
    Path(config.logging.dir).expanduser().mkdir(parents=True, exist_ok=True)
    logger.debug("Config loaded: %s", config.model_dump())
    return config, metadata
