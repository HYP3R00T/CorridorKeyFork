"""Unit tests for corridorkey_new.infra.config."""

from __future__ import annotations

import pytest
from corridorkey_new.infra.config import CorridorKeyConfig


class TestCorridorKeyConfigDefaults:
    def test_default_log_level(self):
        config = CorridorKeyConfig()
        assert config.logging.level == "INFO"

    def test_default_device(self):
        config = CorridorKeyConfig()
        assert config.device == "auto"

    def test_default_log_dir_contains_corridorkey(self):
        config = CorridorKeyConfig()
        assert "corridorkey" in str(config.logging.dir)


class TestCorridorKeyConfigValidation:
    def test_valid_log_levels(self):
        from corridorkey_new.infra.config import LoggingSettings

        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            config = CorridorKeyConfig(logging=LoggingSettings(level=level))
            assert config.logging.level == level

    def test_invalid_log_level_raises(self):
        from corridorkey_new.infra.config import LoggingSettings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            LoggingSettings(level="VERBOSE")  # type: ignore[arg-type]

    def test_valid_devices(self):
        for device in ("auto", "cuda", "rocm", "mps", "cpu", "all", "cuda:0", "cuda:1", "rocm:0"):
            config = CorridorKeyConfig(device=device)
            assert config.device == device

    def test_cuda_index_devices(self):
        for device in ("cuda:0", "cuda:1", "cuda:7"):
            config = CorridorKeyConfig(device=device)
            assert config.device == device

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError):
            CorridorKeyConfig(device="tpu")

    def test_invalid_device_index_raises(self):
        with pytest.raises(ValueError):
            CorridorKeyConfig(device="cuda:abc")


class TestCorridorKeyConfigOverrides:
    def test_override_log_level(self):
        from corridorkey_new.infra.config import LoggingSettings

        config = CorridorKeyConfig(logging=LoggingSettings(level="DEBUG"))
        assert config.logging.level == "DEBUG"

    def test_override_device(self):
        config = CorridorKeyConfig(device="cpu")
        assert config.device == "cpu"


class TestPreprocessSettings:
    def test_defaults(self):
        cfg = CorridorKeyConfig()
        assert cfg.preprocess.img_size == 0  # 0 = auto-select based on VRAM
        assert cfg.preprocess.image_upsample_mode == "bicubic"
        assert cfg.preprocess.half_precision is False
        assert cfg.preprocess.source_passthrough is True

    def test_override_img_size(self):
        from corridorkey_new.infra.config import PreprocessSettings

        cfg = CorridorKeyConfig(preprocess=PreprocessSettings(img_size=512))
        assert cfg.preprocess.img_size == 512

    def test_override_upsample_mode(self):
        from corridorkey_new.infra.config import PreprocessSettings

        cfg = CorridorKeyConfig(preprocess=PreprocessSettings(image_upsample_mode="bilinear"))
        assert cfg.preprocess.image_upsample_mode == "bilinear"

    def test_invalid_upsample_mode_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CorridorKeyConfig(preprocess={"image_upsample_mode": "nearest"})  # type: ignore[arg-type]

    def test_img_size_minimum(self):
        from corridorkey_new.infra.config import PreprocessSettings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PreprocessSettings(img_size=768)  # type: ignore[call-arg]  # intentionally invalid

    def test_valid_img_sizes_accepted(self):
        from corridorkey_new.infra.config import PreprocessSettings

        for size in (0, 512, 1024, 1536, 2048):
            cfg = CorridorKeyConfig(preprocess=PreprocessSettings(img_size=size))
            assert cfg.preprocess.img_size == size


class TestInferenceSettings:
    def test_defaults(self):
        from corridorkey_new.infra.model_hub import default_checkpoint_path

        cfg = CorridorKeyConfig()
        assert cfg.inference.checkpoint_path == default_checkpoint_path()
        assert cfg.inference.use_refiner is True
        assert cfg.inference.mixed_precision is True
        assert cfg.inference.model_precision == "auto"
        assert cfg.inference.refiner_mode == "auto"

    def test_override_checkpoint_path(self, tmp_path):
        from corridorkey_new.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(inference=InferenceSettings(checkpoint_path=p))
        assert cfg.inference.checkpoint_path == p

    def test_override_optimization_mode(self):
        from corridorkey_new.infra.config import InferenceSettings

        cfg = CorridorKeyConfig(inference=InferenceSettings(refiner_mode="tiled"))
        assert cfg.inference.refiner_mode == "tiled"

    def test_invalid_optimization_mode_raises(self):
        from corridorkey_new.infra.config import InferenceSettings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            InferenceSettings(refiner_mode="turbo")  # type: ignore[arg-type]

    def test_invalid_model_precision_raises(self):
        from corridorkey_new.infra.config import InferenceSettings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            InferenceSettings(model_precision="fp8")  # type: ignore[arg-type]


class TestBridgeMethods:
    def test_to_preprocess_config_defaults(self):
        cfg = CorridorKeyConfig(device="cpu")
        pc = cfg.to_preprocess_config()
        assert pc.img_size == 2048
        assert pc.device == "cpu"
        assert pc.image_upsample_mode == "bicubic"
        assert pc.half_precision is False
        assert pc.source_passthrough is True

    def test_to_preprocess_config_device_override(self):
        cfg = CorridorKeyConfig(device="cuda")
        pc = cfg.to_preprocess_config(device="cpu")
        assert pc.device == "cpu"

    def test_to_preprocess_config_respects_img_size(self):
        from corridorkey_new.infra.config import PreprocessSettings

        cfg = CorridorKeyConfig(device="cpu", preprocess=PreprocessSettings(img_size=512))
        pc = cfg.to_preprocess_config()
        assert pc.img_size == 512

    def test_to_inference_config_defaults_to_standard_path(self):
        from corridorkey_new.infra.model_hub import default_checkpoint_path

        cfg = CorridorKeyConfig(device="cpu")
        ic = cfg.to_inference_config()
        assert ic.checkpoint_path == default_checkpoint_path()

    def test_to_inference_config_basic(self, tmp_path):
        import torch
        from corridorkey_new.infra.config import InferenceSettings, PreprocessSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            preprocess=PreprocessSettings(img_size=2048),
            inference=InferenceSettings(checkpoint_path=p),
        )
        ic = cfg.to_inference_config()
        assert ic.checkpoint_path == p
        assert ic.device == "cpu"
        assert ic.img_size == 2048
        assert ic.model_precision == torch.float32

    def test_to_inference_config_float16_precision(self, tmp_path):
        import torch
        from corridorkey_new.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(device="cpu", inference=InferenceSettings(checkpoint_path=p, model_precision="float16"))
        ic = cfg.to_inference_config()
        assert ic.model_precision == torch.float16

    def test_to_inference_config_device_override(self, tmp_path):
        from corridorkey_new.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(device="cuda", inference=InferenceSettings(checkpoint_path=p))
        ic = cfg.to_inference_config(device="cpu")
        assert ic.device == "cpu"

    def test_to_inference_config_img_size_from_preprocess(self, tmp_path):
        from corridorkey_new.infra.config import InferenceSettings, PreprocessSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            preprocess=PreprocessSettings(img_size=1024),
            inference=InferenceSettings(checkpoint_path=p),
        )
        ic = cfg.to_inference_config()
        assert ic.img_size == 1024
