"""Unit tests for the Engine orchestration layer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from corridorkey.engine import Engine
from corridorkey.infra.config.inference import InferenceSettings
from corridorkey.infra.config.pipeline import CorridorKeyConfig


class TestEngineInitialise:
    def test_all_device_uses_multi_device_resolution(self, tmp_path: Path):
        config = CorridorKeyConfig(
            device="all",
            inference=InferenceSettings(checkpoint_path=tmp_path / "model.pth"),
        )
        engine = Engine(config)

        with (
            patch("corridorkey.infra.logging.setup_logging"),
            patch("corridorkey.infra.model_hub.ensure_model"),
            patch("corridorkey.infra.device_utils.resolve_device", side_effect=AssertionError("unexpected call")),
            patch(
                "corridorkey.infra.device_utils.resolve_devices", return_value=["cuda:0", "cuda:1"]
            ) as resolve_devices,
            patch.object(CorridorKeyConfig, "to_pipeline_config", return_value=MagicMock()) as to_pipeline_config,
        ):
            engine._initialise()

        assert engine._resolved_device == "cuda:0"
        resolve_devices.assert_called_once_with("all")
        assert to_pipeline_config.call_count == 1
        assert to_pipeline_config.call_args.kwargs["device"] == "cuda:0"
        assert to_pipeline_config.call_args.kwargs["devices"] == ["cuda:0", "cuda:1"]

    def test_single_device_uses_single_device_resolution(self, tmp_path: Path):
        config = CorridorKeyConfig(
            device="cpu",
            inference=InferenceSettings(checkpoint_path=tmp_path / "model.pth"),
        )
        engine = Engine(config)

        with (
            patch("corridorkey.infra.logging.setup_logging"),
            patch("corridorkey.infra.model_hub.ensure_model"),
            patch("corridorkey.infra.device_utils.resolve_device", return_value="cpu") as resolve_device,
            patch("corridorkey.infra.device_utils.resolve_devices", side_effect=AssertionError("unexpected call")),
            patch.object(CorridorKeyConfig, "to_pipeline_config", return_value=MagicMock()) as to_pipeline_config,
        ):
            engine._initialise()

        assert engine._resolved_device == "cpu"
        resolve_device.assert_called_once_with("cpu")
        assert to_pipeline_config.call_count == 1
        assert to_pipeline_config.call_args.kwargs["device"] == "cpu"
        assert to_pipeline_config.call_args.kwargs["devices"] is None
