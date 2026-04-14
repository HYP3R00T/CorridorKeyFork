"""Unit tests for corridorkey.infra.config."""

from __future__ import annotations

import pytest
from corridorkey.infra.config import CorridorKeyConfig


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
        from corridorkey.infra.config import LoggingSettings

        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            config = CorridorKeyConfig(logging=LoggingSettings(level=level))
            assert config.logging.level == level

    def test_invalid_log_level_raises(self):
        from corridorkey.infra.config import LoggingSettings
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
        from corridorkey.infra.config import LoggingSettings

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
        from corridorkey.infra.config import PreprocessSettings

        cfg = CorridorKeyConfig(preprocess=PreprocessSettings(img_size=512))
        assert cfg.preprocess.img_size == 512

    def test_override_upsample_mode(self):
        from corridorkey.infra.config import PreprocessSettings

        cfg = CorridorKeyConfig(preprocess=PreprocessSettings(image_upsample_mode="bilinear"))
        assert cfg.preprocess.image_upsample_mode == "bilinear"

    def test_invalid_upsample_mode_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CorridorKeyConfig(preprocess={"image_upsample_mode": "nearest"})  # type: ignore[arg-type]

    def test_img_size_minimum(self):
        from corridorkey.infra.config import PreprocessSettings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PreprocessSettings(img_size=768)  # type: ignore[call-arg]  # intentionally invalid

    def test_valid_img_sizes_accepted(self):
        from corridorkey.infra.config import PreprocessSettings

        for size in (0, 512, 1024, 1536, 2048):
            cfg = CorridorKeyConfig(preprocess=PreprocessSettings(img_size=size))
            assert cfg.preprocess.img_size == size


class TestInferenceSettings:
    def test_defaults(self):
        from corridorkey.infra.model_hub import default_checkpoint_path

        cfg = CorridorKeyConfig()
        assert cfg.inference.checkpoint_path == default_checkpoint_path()
        assert cfg.inference.use_refiner is True
        assert cfg.inference.mixed_precision is True
        assert cfg.inference.model_precision == "auto"
        assert cfg.inference.refiner_mode == "auto"

    def test_override_checkpoint_path(self, tmp_path):
        from corridorkey.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(inference=InferenceSettings(checkpoint_path=p))
        assert cfg.inference.checkpoint_path == p

    def test_override_optimization_mode(self):
        from corridorkey.infra.config import InferenceSettings

        cfg = CorridorKeyConfig(inference=InferenceSettings(refiner_mode="tiled"))
        assert cfg.inference.refiner_mode == "tiled"

    def test_invalid_optimization_mode_raises(self):
        from corridorkey.infra.config import InferenceSettings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            InferenceSettings(refiner_mode="turbo")  # type: ignore[arg-type]

    def test_invalid_model_precision_raises(self):
        from corridorkey.infra.config import InferenceSettings
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            InferenceSettings(model_precision="fp8")  # type: ignore[arg-type]


class TestBridgeMethods:
    def test_to_preprocess_config_defaults(self):
        cfg = CorridorKeyConfig(device="cpu")
        pc = cfg.to_preprocess_config()
        assert pc.img_size == 2048
        assert pc.device == "cpu"
        assert pc.half_precision is False
        assert pc.source_passthrough is True

    def test_to_preprocess_config_device_override(self):
        cfg = CorridorKeyConfig(device="cuda")
        pc = cfg.to_preprocess_config(device="cpu")
        assert pc.device == "cpu"

    def test_to_preprocess_config_respects_img_size(self):
        from corridorkey.infra.config import PreprocessSettings

        cfg = CorridorKeyConfig(device="cpu", preprocess=PreprocessSettings(img_size=512))
        pc = cfg.to_preprocess_config()
        assert pc.img_size == 512

    def test_to_inference_config_defaults_to_standard_path(self):
        from corridorkey.infra.model_hub import default_checkpoint_path

        cfg = CorridorKeyConfig(device="cpu")
        ic = cfg.to_inference_config()
        assert ic.checkpoint_path == default_checkpoint_path()

    def test_to_inference_config_basic(self, tmp_path):
        import torch
        from corridorkey.infra.config import InferenceSettings, PreprocessSettings

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
        from corridorkey.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(device="cpu", inference=InferenceSettings(checkpoint_path=p, model_precision="float16"))
        ic = cfg.to_inference_config()
        assert ic.model_precision == torch.float16

    def test_to_inference_config_device_override(self, tmp_path):
        from corridorkey.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(device="cuda", inference=InferenceSettings(checkpoint_path=p))
        ic = cfg.to_inference_config(device="cpu")
        assert ic.device == "cpu"

    def test_to_inference_config_img_size_from_preprocess(self, tmp_path):
        from corridorkey.infra.config import InferenceSettings, PreprocessSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            preprocess=PreprocessSettings(img_size=1024),
            inference=InferenceSettings(checkpoint_path=p),
        )
        ic = cfg.to_inference_config()
        assert ic.img_size == 1024


class TestDeviceValidatorNonStringInput:
    def test_non_string_device_raises(self):
        from pydantic import ValidationError

        with pytest.raises((ValidationError, ValueError)):
            CorridorKeyConfig(device=42)  # type: ignore[arg-type]

    def test_none_device_raises(self):
        from pydantic import ValidationError

        with pytest.raises((ValidationError, ValueError)):
            CorridorKeyConfig(device=None)  # type: ignore[arg-type]


class TestToInferenceConfigPrecisionAuto:
    def test_auto_precision_cpu_resolves_to_float32(self, tmp_path):
        import torch
        from corridorkey.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(device="cpu", inference=InferenceSettings(checkpoint_path=p, model_precision="auto"))
        ic = cfg.to_inference_config(device="cpu")
        assert ic.model_precision == torch.float32

    def test_auto_precision_cuda_ampere_resolves_to_bfloat16(self, tmp_path):
        from unittest.mock import patch

        import torch
        from corridorkey.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(device="cuda", inference=InferenceSettings(checkpoint_path=p, model_precision="auto"))
        mock_props = type("Props", (), {"major": 8, "name": "A100"})()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            ic = cfg.to_inference_config(device="cuda")
        assert ic.model_precision == torch.bfloat16

    def test_auto_precision_cuda_pre_ampere_resolves_to_float16(self, tmp_path):
        from unittest.mock import patch

        import torch
        from corridorkey.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(device="cuda", inference=InferenceSettings(checkpoint_path=p, model_precision="auto"))
        mock_props = type("Props", (), {"major": 7, "name": "V100"})()
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            ic = cfg.to_inference_config(device="cuda")
        assert ic.model_precision == torch.float16

    def test_auto_precision_cuda_unavailable_resolves_to_float32(self, tmp_path):
        from unittest.mock import patch

        import torch
        from corridorkey.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(device="cuda", inference=InferenceSettings(checkpoint_path=p, model_precision="auto"))
        with patch("torch.cuda.is_available", return_value=False):
            ic = cfg.to_inference_config(device="cuda")
        assert ic.model_precision == torch.float32


class TestToInferenceConfigRefinerModeAuto:
    def test_auto_refiner_mode_low_vram_resolves_to_tiled(self, tmp_path):
        from unittest.mock import patch

        from corridorkey.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(device="cpu", inference=InferenceSettings(checkpoint_path=p, refiner_mode="auto"))
        with patch("corridorkey.stages.inference.orchestrator._probe_vram_gb", return_value=8.0):
            _, resolved = cfg._resolve_inference_params(device="cuda")
        assert resolved == "tiled"

    def test_auto_refiner_mode_high_vram_resolves_to_full_frame(self, tmp_path):
        from unittest.mock import patch

        from corridorkey.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(device="cpu", inference=InferenceSettings(checkpoint_path=p, refiner_mode="auto"))
        with patch("corridorkey.stages.inference.orchestrator._probe_vram_gb", return_value=24.0):
            _, resolved = cfg._resolve_inference_params(device="cuda")
        assert resolved == "full_frame"

    def test_auto_refiner_mode_zero_vram_resolves_to_full_frame(self, tmp_path):
        from unittest.mock import patch

        from corridorkey.infra.config import InferenceSettings

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(device="cpu", inference=InferenceSettings(checkpoint_path=p, refiner_mode="auto"))
        with patch("corridorkey.stages.inference.orchestrator._probe_vram_gb", return_value=0.0):
            _, resolved = cfg._resolve_inference_params(device="cuda")
        assert resolved == "full_frame"


class TestToPostprocessConfig:
    def test_returns_postprocess_config(self):
        """to_postprocess_config returns a PostprocessConfig instance."""
        from corridorkey.stages.postprocessor.config import PostprocessConfig

        cfg = CorridorKeyConfig()
        result = cfg.to_postprocess_config()
        assert isinstance(result, PostprocessConfig)

    def test_despill_strength_passed_through(self):
        """despill_strength from PostprocessSettings is forwarded to PostprocessConfig."""
        from corridorkey.infra.config.postprocess import PostprocessSettings

        cfg = CorridorKeyConfig(postprocess=PostprocessSettings(despill_strength=0.3))
        assert cfg.to_postprocess_config().despill_strength == pytest.approx(0.3)

    def test_auto_despeckle_passed_through(self):
        """auto_despeckle=False is forwarded correctly."""
        from corridorkey.infra.config.postprocess import PostprocessSettings

        cfg = CorridorKeyConfig(postprocess=PostprocessSettings(auto_despeckle=False))
        assert cfg.to_postprocess_config().auto_despeckle is False

    def test_hint_sharpen_passed_through(self):
        """hint_sharpen=False is forwarded correctly."""
        from corridorkey.infra.config.postprocess import PostprocessSettings

        cfg = CorridorKeyConfig(postprocess=PostprocessSettings(hint_sharpen=False))
        assert cfg.to_postprocess_config().hint_sharpen is False

    def test_debug_dump_passed_through(self):
        """debug_dump=True is forwarded correctly."""
        from corridorkey.infra.config.postprocess import PostprocessSettings

        cfg = CorridorKeyConfig(postprocess=PostprocessSettings(debug_dump=True))
        assert cfg.to_postprocess_config().debug_dump is True

    def test_edge_blur_px_passed_through(self):
        """edge_blur_px is forwarded from PostprocessSettings to PostprocessConfig."""
        from corridorkey.infra.config.postprocess import PostprocessSettings

        cfg = CorridorKeyConfig(postprocess=PostprocessSettings(edge_blur_px=5))
        assert cfg.to_postprocess_config().edge_blur_px == 5

    def test_edge_blur_px_default_matches_postprocess_config(self):
        """PostprocessSettings.edge_blur_px default must match PostprocessConfig default."""
        from corridorkey.infra.config.postprocess import PostprocessSettings
        from corridorkey.stages.postprocessor.config import PostprocessConfig

        assert PostprocessSettings().edge_blur_px == PostprocessConfig().edge_blur_px


class TestToWriterConfig:
    def test_returns_write_config(self, tmp_path):
        """to_writer_config returns a WriteConfig instance."""
        from corridorkey.stages.writer.contracts import WriteConfig

        assert isinstance(CorridorKeyConfig().to_writer_config(tmp_path), WriteConfig)

    def test_output_dir_set(self, tmp_path):
        """output_dir on the returned config matches the argument."""
        assert CorridorKeyConfig().to_writer_config(tmp_path).output_dir == tmp_path

    def test_alpha_format_passed_through(self, tmp_path):
        """alpha_format='exr' is forwarded to WriteConfig."""
        from corridorkey.infra.config.writer import WriterSettings

        cfg = CorridorKeyConfig(writer=WriterSettings(alpha_format="exr"))
        assert cfg.to_writer_config(tmp_path).alpha_format == "exr"

    def test_comp_enabled_passed_through(self, tmp_path):
        """comp_enabled=False is forwarded to WriteConfig."""
        from corridorkey.infra.config.writer import WriterSettings

        cfg = CorridorKeyConfig(writer=WriterSettings(comp_enabled=False))
        assert cfg.to_writer_config(tmp_path).comp_enabled is False

    def test_output_dir_as_string(self, tmp_path):
        """Passing output_dir as a string is accepted and converted to Path."""
        assert CorridorKeyConfig().to_writer_config(str(tmp_path)).output_dir == tmp_path


class TestToPipelineConfig:
    def _cfg(self, tmp_path):
        from corridorkey.infra.config.inference import InferenceSettings
        from corridorkey.infra.config.preprocess import PreprocessSettings

        return CorridorKeyConfig(
            device="cpu",
            preprocess=PreprocessSettings(img_size=512),
            inference=InferenceSettings(
                checkpoint_path=tmp_path / "model.pth",
                refiner_mode="full_frame",
                model_precision="float32",
            ),
        )

    def test_returns_pipeline_config(self, tmp_path):
        """to_pipeline_config returns a PipelineConfig instance."""
        from corridorkey.runtime.runner import PipelineConfig

        assert isinstance(self._cfg(tmp_path).to_pipeline_config(device="cpu"), PipelineConfig)

    def test_device_propagated_to_preprocess_and_inference(self, tmp_path):
        """device='cpu' is forwarded to both preprocess and inference sub-configs."""
        result = self._cfg(tmp_path).to_pipeline_config(device="cpu")
        assert result.preprocess.device == "cpu"
        assert result.inference.device == "cpu"

    def test_model_passed_through(self, tmp_path):
        """A pre-loaded model is stored on the returned PipelineConfig."""
        import torch.nn as nn

        dummy = nn.Linear(1, 1)
        result = self._cfg(tmp_path).to_pipeline_config(device="cpu", model=dummy)
        assert result.model is dummy

    def test_devices_empty_by_default(self, tmp_path):
        """devices defaults to an empty list when not provided."""
        assert self._cfg(tmp_path).to_pipeline_config(device="cpu").devices == []

    def test_devices_passed_through(self, tmp_path):
        """An explicit devices list is stored on the returned PipelineConfig."""
        result = self._cfg(tmp_path).to_pipeline_config(device="cpu", devices=["cuda:0", "cuda:1"])
        assert result.devices == ["cuda:0", "cuda:1"]
