"""Smoke tests for the public API surface.

Verifies that every symbol in corridorkey.__all__ is importable and that
key symbols are present and functional.
"""

from __future__ import annotations

import corridorkey


class TestAllSymbolsResolve:
    def test_all_symbols_importable(self):
        missing = [name for name in corridorkey.__all__ if getattr(corridorkey, name, None) is None]
        assert missing == [], f"Symbols in __all__ that do not resolve: {missing}"


class TestVersionExposed:
    def test_version_in_all(self):
        assert "__version__" in corridorkey.__all__

    def test_version_is_string(self):
        assert isinstance(corridorkey.__version__, str)

    def test_version_not_empty(self):
        assert corridorkey.__version__ != ""


class TestCoreExported:
    def test_engine_in_all(self):
        assert "Engine" in corridorkey.__all__

    def test_job_stats_in_all(self):
        assert "JobStats" in corridorkey.__all__

    def test_engine_is_class(self):
        from corridorkey import Engine

        assert isinstance(Engine, type)

    def test_job_stats_is_class(self):
        from corridorkey import JobStats

        assert isinstance(JobStats, type)


class TestDataContractsExported:
    def test_clip_in_all(self):
        assert "Clip" in corridorkey.__all__

    def test_skipped_clip_in_all(self):
        assert "SkippedClip" in corridorkey.__all__

    def test_clip_manifest_in_all(self):
        assert "ClipManifest" in corridorkey.__all__

    def test_clip_constructible(self, tmp_path):
        from corridorkey import Clip

        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        assert clip.name == "test"
        assert clip.alpha_path is None

    def test_skipped_clip_constructible(self, tmp_path):
        from corridorkey import SkippedClip

        s = SkippedClip(path=tmp_path, reason="test reason")
        assert s.reason == "test reason"


class TestAlphaGeneratorExported:
    def test_alpha_generator_in_all(self):
        assert "AlphaGenerator" in corridorkey.__all__

    def test_alpha_generator_is_protocol(self):
        from corridorkey import AlphaGenerator

        # Protocol — not a concrete class, but importable
        assert AlphaGenerator is not None

    def test_engine_accepts_alpha_generator(self, tmp_path):
        """set_alpha_generator() accepts any object with a generate() method."""
        from corridorkey import Engine, load_config_with_metadata

        config, _ = load_config_with_metadata()

        class MinimalAlpha:
            def generate(self, manifest):
                return manifest

        engine = Engine(config)
        engine.set_alpha_generator(MinimalAlpha())  # should not raise

    def test_engine_rejects_invalid_alpha_generator(self, tmp_path):
        """set_alpha_generator() raises EngineError for objects without generate()."""
        import pytest
        from corridorkey import Engine, EngineError, load_config_with_metadata

        config, _ = load_config_with_metadata()
        engine = Engine(config)
        with pytest.raises(EngineError):
            engine.set_alpha_generator(object())


class TestErrorsExported:
    def test_all_errors_in_all(self):
        expected = [
            "CorridorKeyError",
            "EngineError",
            "AlphaGeneratorError",
            "DeviceError",
            "ModelError",
            "ClipScanError",
            "ExtractionError",
            "FrameMismatchError",
            "JobCancelledError",
            "FrameReadError",
            "WriteFailureError",
            "VRAMInsufficientError",
        ]
        for name in expected:
            assert name in corridorkey.__all__, f"{name} missing from __all__"

    def test_alpha_generator_error_message(self):
        from corridorkey import AlphaGeneratorError

        err = AlphaGeneratorError("MyClip")
        assert "MyClip" in str(err)
        assert "AlphaGenerator" in str(err)

    def test_all_errors_catchable_as_base(self):
        from corridorkey import (
            AlphaGeneratorError,
            ClipScanError,
            CorridorKeyError,
            DeviceError,
            EngineError,
            ExtractionError,
            FrameMismatchError,
            FrameReadError,
            JobCancelledError,
            ModelError,
            VRAMInsufficientError,
            WriteFailureError,
        )

        errors = [
            EngineError("x"),
            AlphaGeneratorError("clip"),
            ClipScanError("x"),
            ExtractionError("clip", "detail"),
            FrameMismatchError("clip", 3, 2),
            FrameReadError("x"),
            JobCancelledError("clip"),
            DeviceError("x"),
            ModelError("x"),
            WriteFailureError("path"),
            VRAMInsufficientError(10.0, 4.0),
        ]
        for err in errors:
            assert isinstance(err, CorridorKeyError)


class TestRemovedFromPublicAPI:
    """Verify that internal symbols are not in __all__."""

    def test_clip_record_not_in_all(self):
        assert "ClipRecord" not in corridorkey.__all__

    def test_clip_state_not_in_all(self):
        assert "ClipState" not in corridorkey.__all__

    def test_frame_range_not_in_all(self):
        assert "FrameRange" not in corridorkey.__all__

    def test_get_clip_state_not_in_all(self):
        assert "get_clip_state" not in corridorkey.__all__

    def test_scan_result_not_in_all(self):
        assert "ScanResult" not in corridorkey.__all__

    def test_list_frames_not_in_all(self):
        assert "list_frames" not in corridorkey.__all__

    def test_model_backend_not_in_all(self):
        assert "ModelBackend" not in corridorkey.__all__

    def test_pipeline_config_not_in_all(self):
        assert "PipelineConfig" not in corridorkey.__all__

    def test_pipeline_events_not_in_all(self):
        assert "PipelineEvents" not in corridorkey.__all__

    def test_preprocess_frame_not_in_all(self):
        assert "preprocess_frame" not in corridorkey.__all__

    def test_postprocess_frame_not_in_all(self):
        assert "postprocess_frame" not in corridorkey.__all__

    def test_write_frame_not_in_all(self):
        assert "write_frame" not in corridorkey.__all__

    def test_inference_result_not_in_all(self):
        assert "InferenceResult" not in corridorkey.__all__

    def test_processed_frame_not_in_all(self):
        assert "ProcessedFrame" not in corridorkey.__all__

    def test_internal_still_importable_from_submodule(self):
        """Internal symbols stay in the codebase — just not in __all__."""
        from corridorkey.runtime.clip_state import ClipRecord, ClipState
        from corridorkey.stages.scanner.contracts import ScanResult

        assert ClipRecord is not None
        assert ClipState is not None
        assert ScanResult is not None


class TestConfigExported:
    def test_load_config_in_all(self):
        assert "load_config" in corridorkey.__all__

    def test_corridorkey_config_in_all(self):
        assert "CorridorKeyConfig" in corridorkey.__all__

    def test_settings_metadata_in_all(self):
        assert "SettingsMetadata" in corridorkey.__all__

    def test_load_config_with_metadata_returns_settings_metadata(self):
        from corridorkey import SettingsMetadata, load_config_with_metadata

        _, metadata = load_config_with_metadata()
        assert isinstance(metadata, SettingsMetadata)


class TestModelConstantsNotInAll:
    def test_model_url_not_in_all(self):
        assert "MODEL_URL" not in corridorkey.__all__

    def test_model_filename_not_in_all(self):
        assert "MODEL_FILENAME" not in corridorkey.__all__

    def test_model_constants_still_importable_from_submodule(self):
        from corridorkey.infra.model_hub import MODEL_FILENAME, MODEL_URL

        assert isinstance(MODEL_URL, str)
        assert MODEL_URL.startswith("https://")
        assert isinstance(MODEL_FILENAME, str)
        assert MODEL_FILENAME.endswith(".pth")
