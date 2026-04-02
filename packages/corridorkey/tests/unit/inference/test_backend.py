"""Unit tests for corridorkey.stages.inference.backend — TorchBackend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from corridorkey.stages.inference.backend import ModelBackend, TorchBackend
from corridorkey.stages.inference.config import InferenceConfig
from corridorkey.stages.inference.contracts import InferenceResult
from corridorkey.stages.preprocessor.contracts import FrameMeta, PreprocessedFrame


def _make_config(tmp_path: Path, **kwargs) -> InferenceConfig:
    return InferenceConfig(**{"checkpoint_path": tmp_path / "model.pth", "device": "cpu"} | kwargs)


def _make_frame() -> PreprocessedFrame:
    meta = FrameMeta(frame_index=0, original_h=32, original_w=32)
    return PreprocessedFrame(tensor=torch.zeros(1, 4, 32, 32), meta=meta)


def _make_result(frame: PreprocessedFrame) -> InferenceResult:
    return InferenceResult(
        alpha=torch.zeros(1, 1, 32, 32),
        fg=torch.zeros(1, 3, 32, 32),
        meta=frame.meta,
    )


class TestTorchBackendProtocolConformance:
    def test_satisfies_model_backend_protocol(self, tmp_path: Path):
        backend = TorchBackend(config=_make_config(tmp_path), model=MagicMock())
        assert isinstance(backend, ModelBackend)

    def test_backend_name_is_torch(self, tmp_path: Path):
        backend = TorchBackend(config=_make_config(tmp_path), model=MagicMock())
        assert backend.backend_name == "torch"


class TestTorchBackendResolvedConfig:
    def test_resolved_config_is_dict(self, tmp_path: Path):
        backend = TorchBackend(config=_make_config(tmp_path), model=MagicMock())
        assert isinstance(backend.resolved_config, dict)

    def test_resolved_config_keys(self, tmp_path: Path):
        backend = TorchBackend(config=_make_config(tmp_path), model=MagicMock())
        rc = backend.resolved_config
        for key in ("backend", "device", "refiner_mode", "precision", "img_size", "use_refiner", "mixed_precision"):
            assert key in rc, f"missing key: {key}"

    def test_resolved_config_all_values_are_strings(self, tmp_path: Path):
        backend = TorchBackend(config=_make_config(tmp_path), model=MagicMock())
        for k, v in backend.resolved_config.items():
            assert isinstance(v, str), f"value for '{k}' is not a string: {v!r}"

    def test_resolved_config_backend_is_torch(self, tmp_path: Path):
        backend = TorchBackend(config=_make_config(tmp_path), model=MagicMock())
        assert backend.resolved_config["backend"] == "torch"

    def test_resolved_config_device_matches_config(self, tmp_path: Path):
        backend = TorchBackend(config=_make_config(tmp_path, device="cpu"), model=MagicMock())
        assert backend.resolved_config["device"] == "cpu"

    def test_resolved_config_img_size_matches_config(self, tmp_path: Path):
        backend = TorchBackend(config=_make_config(tmp_path, img_size=1024), model=MagicMock())
        assert backend.resolved_config["img_size"] == "1024"

    def test_resolved_config_refiner_mode_uses_resolved_not_auto(self, tmp_path: Path):
        """When resolved_refiner_mode is passed, it should appear in resolved_config."""
        backend = TorchBackend(
            config=_make_config(tmp_path, refiner_mode="auto"),
            model=MagicMock(),
            resolved_refiner_mode="tiled",
        )
        assert backend.resolved_config["refiner_mode"] == "tiled"

    def test_resolved_config_precision_no_torch_prefix(self, tmp_path: Path):
        """Precision string must not contain 'torch.' prefix."""
        backend = TorchBackend(config=_make_config(tmp_path), model=MagicMock())
        assert "torch." not in backend.resolved_config["precision"]


class TestTorchBackendResolvedRefinerMode:
    def test_defaults_to_config_refiner_mode_when_not_provided(self, tmp_path: Path):
        """When resolved_refiner_mode is None, falls back to config.refiner_mode."""
        backend = TorchBackend(
            config=_make_config(tmp_path, refiner_mode="full_frame"),
            model=MagicMock(),
            resolved_refiner_mode=None,
        )
        assert backend._resolved_refiner_mode == "full_frame"

    def test_explicit_resolved_mode_takes_priority(self, tmp_path: Path):
        backend = TorchBackend(
            config=_make_config(tmp_path, refiner_mode="full_frame"),
            model=MagicMock(),
            resolved_refiner_mode="tiled",
        )
        assert backend._resolved_refiner_mode == "tiled"

    def test_auto_config_mode_falls_back_to_auto_string(self, tmp_path: Path):
        """refiner_mode='auto' with no resolved_refiner_mode → stored as 'auto'."""
        backend = TorchBackend(
            config=_make_config(tmp_path, refiner_mode="auto"),
            model=MagicMock(),
            resolved_refiner_mode=None,
        )
        # 'auto' is falsy-ish but it's a non-empty string — `or` picks config value
        assert backend._resolved_refiner_mode == "auto"


class TestTorchBackendRun:
    def test_run_returns_inference_result(self, tmp_path: Path):
        frame = _make_frame()
        result = _make_result(frame)
        model = MagicMock()

        with patch("corridorkey.stages.inference.orchestrator.run_inference", return_value=result):
            backend = TorchBackend(config=_make_config(tmp_path), model=model)
            out = backend.run(frame)

        assert out is result

    def test_run_passes_resolved_refiner_mode(self, tmp_path: Path):
        frame = _make_frame()
        result = _make_result(frame)
        model = MagicMock()
        captured = {}

        def fake_run(f, m, c, resolved_refiner_mode=None):
            captured["mode"] = resolved_refiner_mode
            return result

        with patch("corridorkey.stages.inference.orchestrator.run_inference", side_effect=fake_run):
            backend = TorchBackend(
                config=_make_config(tmp_path),
                model=model,
                resolved_refiner_mode="tiled",
            )
            backend.run(frame)

        assert captured["mode"] == "tiled"


class TestModelBackendProtocol:
    def test_torch_backend_is_instance_of_protocol(self, tmp_path: Path):
        backend = TorchBackend(config=_make_config(tmp_path), model=MagicMock())
        assert isinstance(backend, ModelBackend)

    def test_arbitrary_object_not_protocol(self):
        class NotABackend:
            pass

        assert not isinstance(NotABackend(), ModelBackend)

    def test_object_with_all_members_satisfies_protocol(self, tmp_path: Path):
        """Any object with backend_name, resolved_config, and run satisfies the protocol."""

        class MinimalBackend:
            @property
            def backend_name(self) -> str:
                return "minimal"

            @property
            def resolved_config(self) -> dict[str, str]:
                return {}

            def run(self, frame):
                return None

        assert isinstance(MinimalBackend(), ModelBackend)
