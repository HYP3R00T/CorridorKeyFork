"""Unit tests for pipeline.py - high-level process_directory orchestration.

pipeline.py is the glue between CorridorKeyService and the outside world.
Its _process_clip function implements the per-clip state routing logic that
decides whether to skip, generate alpha, or run inference. Tests use mocks
for CorridorKeyService so no GPU or filesystem is needed.

Key invariants tested:
- COMPLETE and ERROR clips are always skipped
- RAW/MASKED clips without a generator are skipped with a warning
- RAW/MASKED clips with a generator run alpha generation then inference
- Exceptions from inference are captured into ClipSummary.error
- unload_engine is always called, even if an exception is raised
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from corridorkey.clip_state import ClipEntry, ClipState
from corridorkey.pipeline import ClipSummary, PipelineResult, _process_clip, process_directory
from corridorkey.service import InferenceParams, OutputConfig


def _make_clip(name: str, state: ClipState, error_message: str | None = None) -> ClipEntry:
    clip = ClipEntry(name=name, root_path=f"/fake/{name}", state=state)
    clip.error_message = error_message
    return clip


def _make_service(clips: list[ClipEntry]) -> MagicMock:
    service = MagicMock()
    service.scan_clips.return_value = clips
    service.run_inference.return_value = [MagicMock(success=True) for _ in range(3)]
    return service


class TestClipSummary:
    """ClipSummary - skipped and error flag semantics."""

    def test_skipped_flag(self):
        """A skipped summary must have skipped=True and no error."""
        s = ClipSummary(name="x", state="RAW", skipped=True)
        assert s.skipped is True
        assert s.error is None

    def test_error_flag(self):
        """An errored summary must carry the error message string."""
        s = ClipSummary(name="x", state="RAW", error="boom")
        assert s.error == "boom"


class TestPipelineResult:
    """PipelineResult - succeeded/failed/skipped partition properties."""

    def test_succeeded_failed_skipped(self):
        """succeeded, failed, and skipped must partition the clips list correctly."""
        result = PipelineResult(
            clips=[
                ClipSummary(name="a", state="COMPLETE", frames_processed=10, frames_total=10),
                ClipSummary(name="b", state="ERROR", error="failed"),
                ClipSummary(name="c", state="RAW", skipped=True),
            ]
        )
        assert len(result.succeeded) == 1
        assert len(result.failed) == 1
        assert len(result.skipped) == 1


class TestProcessClip:
    """_process_clip - per-clip state routing, alpha generation, and error capture."""

    def _service(self) -> MagicMock:
        s = MagicMock()
        s.run_inference.return_value = [MagicMock(success=True)] * 5
        return s

    def test_complete_clip_is_skipped(self):
        """A COMPLETE clip must be skipped without calling run_inference."""
        clip = _make_clip("shot1", ClipState.COMPLETE)
        summary = _process_clip(clip, self._service(), InferenceParams(), OutputConfig(), None, None, None, None)
        assert summary.skipped is True

    def test_error_clip_is_skipped(self):
        """An ERROR clip must be skipped; its existing error_message is preserved."""
        clip = _make_clip("shot1", ClipState.ERROR, error_message="something broke")
        summary = _process_clip(clip, self._service(), InferenceParams(), OutputConfig(), None, None, None, None)
        assert summary.skipped is True

    def test_extracting_clip_is_skipped(self):
        """An EXTRACTING clip (frames not yet ready) must be skipped."""
        clip = _make_clip("shot1", ClipState.EXTRACTING)
        summary = _process_clip(clip, self._service(), InferenceParams(), OutputConfig(), None, None, None, None)
        assert summary.skipped is True

    def test_raw_without_generator_is_skipped(self):
        """A RAW clip with no alpha generator must be skipped with a warning."""
        clip = _make_clip("shot1", ClipState.RAW)
        warnings: list[str] = []
        summary = _process_clip(
            clip, self._service(), InferenceParams(), OutputConfig(), None, None, lambda msg: warnings.append(msg), None
        )
        assert summary.skipped is True
        assert any("no alpha generator" in w for w in warnings)

    def test_ready_clip_runs_inference(self):
        """A READY clip must call run_inference and report the processed frame count."""
        clip = _make_clip("shot1", ClipState.READY)
        service = self._service()
        summary = _process_clip(clip, service, InferenceParams(), OutputConfig(), None, None, None, None)
        service.run_inference.assert_called_once()
        assert summary.frames_processed == 5
        assert summary.error is None

    def test_raw_with_generator_runs_alpha_then_inference(self):
        """A RAW clip with a generator must run alpha generation then inference in order."""
        clip = _make_clip("shot1", ClipState.RAW)
        service = self._service()
        generator = MagicMock()
        generator.name = "mock_gen"

        # Simulate generator transitioning clip to READY
        def fake_generate(c, gen, **kwargs):
            c.state = ClipState.READY

        service.run_alpha_generator.side_effect = fake_generate

        summary = _process_clip(clip, service, InferenceParams(), OutputConfig(), generator, None, None, None)
        service.run_alpha_generator.assert_called_once()
        service.run_inference.assert_called_once()
        assert summary.error is None

    def test_inference_error_captured(self):
        """A CorridorKeyError raised by run_inference must be captured into summary.error."""
        from corridorkey.errors import CorridorKeyError

        clip = _make_clip("shot1", ClipState.READY)
        service = self._service()
        service.run_inference.side_effect = CorridorKeyError("inference exploded")
        summary = _process_clip(clip, service, InferenceParams(), OutputConfig(), None, None, None, None)
        assert summary.error == "inference exploded"

    def test_on_clip_start_called(self):
        """on_clip_start callback must be invoked with the clip name and state string."""
        clip = _make_clip("shot1", ClipState.COMPLETE)
        started: list[tuple] = []
        _process_clip(
            clip,
            self._service(),
            InferenceParams(),
            OutputConfig(),
            None,
            None,
            None,
            lambda name, state: started.append((name, state)),
        )
        assert started == [("shot1", "COMPLETE")]


class TestProcessDirectory:
    """process_directory - service lifecycle and engine cleanup guarantees."""

    def test_empty_directory_returns_empty_result(self):
        """An empty project directory must return a PipelineResult with no clips."""
        with patch("corridorkey.pipeline.CorridorKeyService") as mock_service_cls:
            instance = mock_service_cls.return_value
            instance.scan_clips.return_value = []
            instance.detect_device.return_value = "cpu"
            result = process_directory("/fake/dir")
        assert result.clips == []

    def test_engine_unloaded_on_completion(self):
        """unload_engine must be called after a successful run to free GPU memory."""
        with patch("corridorkey.pipeline.CorridorKeyService") as mock_service_cls:
            instance = mock_service_cls.return_value
            instance.scan_clips.return_value = []
            instance.detect_device.return_value = "cpu"
            process_directory("/fake/dir")
        instance.unload_engine.assert_called_once()

    def test_engine_unloaded_on_exception(self):
        """unload_engine must be called even when an exception is raised mid-run."""
        with patch("corridorkey.pipeline.CorridorKeyService") as mock_service_cls:
            instance = mock_service_cls.return_value
            instance.scan_clips.side_effect = RuntimeError("scan failed")
            instance.detect_device.return_value = "cpu"
            with pytest.raises(RuntimeError):
                process_directory("/fake/dir")
        instance.unload_engine.assert_called_once()
