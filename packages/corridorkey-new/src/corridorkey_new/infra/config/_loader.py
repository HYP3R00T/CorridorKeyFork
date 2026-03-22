"""Config loading utilities — wraps utilityhub_config.load_settings."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from utilityhub_config import load_settings

from corridorkey_new.infra.config.pipeline import CorridorKeyConfig

logger = logging.getLogger(__name__)

APP_NAME = "corridorkey"


def _load(overrides: dict | None) -> tuple[CorridorKeyConfig, Any]:
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


def load_config_with_metadata(overrides: dict | None = None) -> tuple[CorridorKeyConfig, Any]:
    """Like :func:`load_config` but also returns per-field source metadata."""
    return _load(overrides)
