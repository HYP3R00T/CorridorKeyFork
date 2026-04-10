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


class TestListFramesExported:
    def test_list_frames_in_all(self):
        assert "list_frames" in corridorkey.__all__

    def test_list_frames_callable(self):
        from corridorkey import list_frames

        assert callable(list_frames)

    def test_list_frames_returns_list(self, tmp_path):
        import cv2
        import numpy as np
        from corridorkey import list_frames

        assert list_frames(tmp_path) == []

        for i in range(3):
            cv2.imwrite(str(tmp_path / f"frame_{i:06d}.png"), np.zeros((4, 4, 3), dtype=np.uint8))

        files = list_frames(tmp_path)
        assert len(files) == 3
        assert [f.name for f in files] == [
            "frame_000000.png",
            "frame_000001.png",
            "frame_000002.png",
        ]


class TestClipRecordExported:
    def test_clip_record_in_all(self):
        assert "ClipRecord" in corridorkey.__all__

    def test_clip_record_is_class(self):
        from corridorkey import ClipRecord

        assert isinstance(ClipRecord, type)

    def test_clip_record_constructible_from_clip(self, tmp_path):
        import cv2
        import numpy as np
        from corridorkey import Clip, ClipRecord, ClipState

        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        cv2.imwrite(str(input_dir / "frame_000001.png"), np.zeros((8, 8, 3), dtype=np.uint8))
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        entry = ClipRecord.from_clip(clip)

        assert entry.name == "test"
        assert entry.state == ClipState.RAW
        assert entry.manifest is None
        assert entry.in_out_range is None

    def test_clip_record_state_machine_accessible(self, tmp_path):
        from corridorkey import Clip, ClipRecord, ClipState

        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        entry = ClipRecord(clip=clip, state=ClipState.RAW)
        entry.transition_to(ClipState.READY)
        assert entry.state == ClipState.READY


class TestFrameRangeExported:
    def test_frame_range_in_all(self):
        assert "FrameRange" in corridorkey.__all__

    def test_frame_range_is_class(self):
        from corridorkey import FrameRange

        assert isinstance(FrameRange, type)

    def test_frame_range_constructible(self):
        from corridorkey import FrameRange

        r = FrameRange(in_point=5, out_point=14)
        assert r.frame_count == 10
        assert r.contains(5)
        assert r.contains(14)
        assert not r.contains(15)

    def test_frame_range_to_frame_range(self):
        from corridorkey import FrameRange

        r = FrameRange(in_point=0, out_point=9)
        assert r.to_frame_range() == (0, 10)

    def test_frame_range_usable_on_clip_record(self, tmp_path):
        from corridorkey import Clip, ClipRecord, ClipState, FrameRange

        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        entry = ClipRecord(clip=clip, state=ClipState.RAW)
        entry.in_out_range = FrameRange(in_point=10, out_point=49)
        assert entry.in_out_range.frame_count == 40


class TestGetClipStateExported:
    def test_get_clip_state_in_all(self):
        assert "get_clip_state" in corridorkey.__all__

    def test_get_clip_state_callable(self):
        from corridorkey import get_clip_state

        assert callable(get_clip_state)

    def test_get_clip_state_returns_clip_state(self, tmp_path):
        import cv2
        import numpy as np
        from corridorkey import Clip, ClipState, get_clip_state

        input_dir = tmp_path / "Input"
        input_dir.mkdir()
        cv2.imwrite(str(input_dir / "frame_000001.png"), np.zeros((8, 8, 3), dtype=np.uint8))
        clip = Clip(name="test", root=tmp_path, input_path=input_dir, alpha_path=None)
        state = get_clip_state(clip)

        assert isinstance(state, ClipState)
        assert state == ClipState.RAW


class TestRemovedFromPublicAPI:
    def test_runner_not_in_all(self):
        assert "Runner" not in corridorkey.__all__

    def test_pipeline_config_not_in_all(self):
        assert "PipelineConfig" not in corridorkey.__all__

    def test_pipeline_events_not_in_all(self):
        assert "PipelineEvents" not in corridorkey.__all__

    def test_setup_logging_not_in_all(self):
        assert "setup_logging" not in corridorkey.__all__

    def test_ensure_model_not_in_all(self):
        assert "ensure_model" not in corridorkey.__all__


class TestNewErrorsExported:
    def test_engine_error_in_all(self):
        assert "EngineError" in corridorkey.__all__

    def test_alpha_generator_error_in_all(self):
        assert "AlphaGeneratorError" in corridorkey.__all__

    def test_alpha_generator_error_message(self):
        from corridorkey import AlphaGeneratorError

        err = AlphaGeneratorError("MyClip")
        assert "MyClip" in str(err)
        assert "AlphaGenerator" in str(err)


class TestProtocolsExported:
    def test_alpha_generator_in_all(self):
        assert "AlphaGenerator" in corridorkey.__all__

    def test_model_backend_in_all(self):
        assert "ModelBackend" in corridorkey.__all__


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


class TestSettingsMetadataExported:
    def test_settings_metadata_in_all(self):
        assert "SettingsMetadata" in corridorkey.__all__

    def test_settings_metadata_importable(self):
        from corridorkey import SettingsMetadata

        assert SettingsMetadata is not None

    def test_load_config_with_metadata_returns_settings_metadata(self):
        from corridorkey import SettingsMetadata, load_config_with_metadata

        _, metadata = load_config_with_metadata()
        assert isinstance(metadata, SettingsMetadata)

    def test_settings_metadata_get_source_callable(self):
        from corridorkey import load_config_with_metadata

        _, metadata = load_config_with_metadata()
        result = metadata.get_source("device")
        assert result is None or hasattr(result, "source")
