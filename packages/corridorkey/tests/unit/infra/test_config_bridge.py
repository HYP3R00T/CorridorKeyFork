"""Tests for CorridorKeyConfig bridge methods: to_postprocess_config, to_writer_config, to_pipeline_config."""

from __future__ import annotations

from pathlib import Path

import pytest
from corridorkey.infra.config import CorridorKeyConfig
from corridorkey.infra.config.inference import InferenceSettings
from corridorkey.infra.config.postprocess import PostprocessSettings
from corridorkey.infra.config.preprocess import PreprocessSettings
from corridorkey.infra.config.writer import WriterSettings


class TestToPostprocessConfig:
    def test_returns_postprocess_config(self):
        from corridorkey.stages.postprocessor.config import PostprocessConfig

        cfg = CorridorKeyConfig()
        result = cfg.to_postprocess_config()
        assert isinstance(result, PostprocessConfig)

    def test_despill_strength_passed_through(self):
        cfg = CorridorKeyConfig(postprocess=PostprocessSettings(despill_strength=0.3))
        result = cfg.to_postprocess_config()
        assert result.despill_strength == pytest.approx(0.3)

    def test_auto_despeckle_passed_through(self):
        cfg = CorridorKeyConfig(postprocess=PostprocessSettings(auto_despeckle=False))
        result = cfg.to_postprocess_config()
        assert result.auto_despeckle is False

    def test_hint_sharpen_passed_through(self):
        cfg = CorridorKeyConfig(postprocess=PostprocessSettings(hint_sharpen=False))
        result = cfg.to_postprocess_config()
        assert result.hint_sharpen is False

    def test_debug_dump_passed_through(self):
        cfg = CorridorKeyConfig(postprocess=PostprocessSettings(debug_dump=True))
        result = cfg.to_postprocess_config()
        assert result.debug_dump is True

    def test_upsample_modes_passed_through(self):
        cfg = CorridorKeyConfig(
            postprocess=PostprocessSettings(fg_upsample_mode="bilinear", alpha_upsample_mode="bilinear")
        )
        result = cfg.to_postprocess_config()
        assert result.fg_upsample_mode == "bilinear"
        assert result.alpha_upsample_mode == "bilinear"


class TestToWriterConfig:
    def test_returns_write_config(self, tmp_path: Path):
        from corridorkey.stages.writer.contracts import WriteConfig

        cfg = CorridorKeyConfig()
        result = cfg.to_writer_config(tmp_path)
        assert isinstance(result, WriteConfig)

    def test_output_dir_set(self, tmp_path: Path):
        cfg = CorridorKeyConfig()
        result = cfg.to_writer_config(tmp_path)
        assert result.output_dir == tmp_path

    def test_alpha_format_passed_through(self, tmp_path: Path):
        cfg = CorridorKeyConfig(writer=WriterSettings(alpha_format="exr"))
        result = cfg.to_writer_config(tmp_path)
        assert result.alpha_format == "exr"

    def test_fg_format_passed_through(self, tmp_path: Path):
        cfg = CorridorKeyConfig(writer=WriterSettings(fg_format="exr"))
        result = cfg.to_writer_config(tmp_path)
        assert result.fg_format == "exr"

    def test_comp_enabled_passed_through(self, tmp_path: Path):
        cfg = CorridorKeyConfig(writer=WriterSettings(comp_enabled=False))
        result = cfg.to_writer_config(tmp_path)
        assert result.comp_enabled is False

    def test_exr_compression_passed_through(self, tmp_path: Path):
        cfg = CorridorKeyConfig(writer=WriterSettings(exr_compression="zip"))
        result = cfg.to_writer_config(tmp_path)
        assert result.exr_compression == "zip"

    def test_output_dir_as_string(self, tmp_path: Path):
        cfg = CorridorKeyConfig()
        result = cfg.to_writer_config(str(tmp_path))
        assert result.output_dir == tmp_path


class TestToPipelineConfig:
    def test_returns_pipeline_config(self, tmp_path: Path):
        from corridorkey.runtime.runner import PipelineConfig

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            preprocess=PreprocessSettings(img_size=512),
            inference=InferenceSettings(checkpoint_path=p, refiner_mode="full_frame", model_precision="float32"),
        )
        result = cfg.to_pipeline_config(device="cpu")
        assert isinstance(result, PipelineConfig)

    def test_pipeline_config_device_propagated(self, tmp_path: Path):
        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            preprocess=PreprocessSettings(img_size=512),
            inference=InferenceSettings(checkpoint_path=p, refiner_mode="full_frame", model_precision="float32"),
        )
        result = cfg.to_pipeline_config(device="cpu")
        assert result.preprocess.device == "cpu"
        assert result.inference is not None
        assert result.inference.device == "cpu"

    def test_pipeline_config_model_passed_through(self, tmp_path: Path):
        import torch.nn as nn

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            preprocess=PreprocessSettings(img_size=512),
            inference=InferenceSettings(checkpoint_path=p, refiner_mode="full_frame", model_precision="float32"),
        )
        dummy_model = nn.Linear(1, 1)
        result = cfg.to_pipeline_config(device="cpu", model=dummy_model)
        assert result.model is dummy_model

    def test_pipeline_config_devices_empty_by_default(self, tmp_path: Path):
        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            preprocess=PreprocessSettings(img_size=512),
            inference=InferenceSettings(checkpoint_path=p, refiner_mode="full_frame", model_precision="float32"),
        )
        result = cfg.to_pipeline_config(device="cpu")
        assert result.devices == []

    def test_pipeline_config_devices_passed_through(self, tmp_path: Path):
        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            preprocess=PreprocessSettings(img_size=512),
            inference=InferenceSettings(checkpoint_path=p, refiner_mode="full_frame", model_precision="float32"),
        )
        result = cfg.to_pipeline_config(device="cpu", devices=["cuda:0", "cuda:1"])
        assert result.devices == ["cuda:0", "cuda:1"]

    def test_pipeline_config_model_and_devices_together(self, tmp_path: Path):
        import torch.nn as nn

        p = tmp_path / "model.pth"
        cfg = CorridorKeyConfig(
            device="cpu",
            preprocess=PreprocessSettings(img_size=512),
            inference=InferenceSettings(checkpoint_path=p, refiner_mode="full_frame", model_precision="float32"),
        )
        dummy_model = nn.Linear(1, 1)
        result = cfg.to_pipeline_config(device="cpu", model=dummy_model, devices=["cpu", "cpu"])
        assert result.model is dummy_model
        assert result.devices == ["cpu", "cpu"]
