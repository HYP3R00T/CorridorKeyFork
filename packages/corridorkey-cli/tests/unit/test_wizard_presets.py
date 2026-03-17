"""Unit tests for wizard engine preset mapping."""

from __future__ import annotations

import pytest
from corridorkey_cli.commands.wizard import _ENGINE_PRESET_ALIASES, _ENGINE_PRESET_CHOICES, _resolve_engine_preset


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("speed", ("auto", "speed", "fp16", 1024)),
        ("balanced", ("auto", "auto", "auto", 1536)),
        ("quality", ("auto", "speed", "bf16", 2048)),
        ("max_quality", ("auto", "speed", "fp32", 2560)),
        ("lowvram", ("auto", "lowvram", "fp16", 1024)),
    ],
)
def test_resolve_engine_preset_defaults(preset: str, expected: tuple[str, str, str, int | None]) -> None:
    assert _resolve_engine_preset(preset) == expected


def test_resolve_engine_preset_uses_device_override() -> None:
    device, opt_mode, precision, img_size = _resolve_engine_preset("quality", default_device="cuda")
    assert device == "cuda"
    assert opt_mode == "speed"
    assert precision == "bf16"
    assert img_size == 2048


def test_resolve_engine_preset_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown preset"):
        _resolve_engine_preset("manual")


def test_engine_preset_choices_order() -> None:
    assert _ENGINE_PRESET_CHOICES == ["speed", "balanced", "quality", "max_quality", "lowvram", "manual"]


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("d", "speed"),
        ("default", "speed"),
        ("s", "speed"),
        ("b", "balanced"),
        ("q", "quality"),
        ("mq", "max_quality"),
        ("l", "lowvram"),
        ("m", "manual"),
    ],
)
def test_engine_preset_aliases(alias: str, expected: str) -> None:
    assert _ENGINE_PRESET_ALIASES[alias] == expected
