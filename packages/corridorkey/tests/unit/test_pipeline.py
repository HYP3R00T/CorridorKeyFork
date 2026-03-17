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
from corridorkey.contracts import InferenceParams, OutputConfig
from corridorkey.pipeline import ClipSummary, PipelineResult, _process_clip, process_directory


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
        """An EXTRACTING clip with no extract_clip mock must be handled by the service."""
        clip = _make_clip("shot1", ClipState.EXTRACTING)
        service = self._service()

        # extract_clip transitions the clip to READY so inference runs
        def fake_extract(c, **kwargs):
            c.state = ClipState.READY
            c.input_asset = MagicMock(frame_count=5)
            c.alpha_asset = MagicMock(frame_count=5)

        service.extract_clip.side_effect = fake_extract
        summary = _process_clip(clip, service, InferenceParams(), OutputConfig(), None, None, None, None)
        service.extract_clip.assert_called_once()
        assert summary.error is None

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


class TestProcessDirectoryOverrides:
    """process_directory - device/optimization/precision overrides and on_clip_done."""

    def _patched_service(self, clips):
        instance = MagicMock()
        instance.scan_clips.return_value = clips
        instance.detect_device.return_value = "cpu"
        instance.default_inference_params.return_value = InferenceParams()
        instance.default_output_config.return_value = OutputConfig()
        return instance

    def test_device_override_passed_to_load_config(self):
        """device override must be forwarded to load_config."""
        with (
            patch("corridorkey.pipeline.CorridorKeyService") as mock_cls,
            patch("corridorkey.pipeline.load_config") as mock_lc,
        ):
            mock_lc.return_value = MagicMock()
            instance = self._patched_service([])
            mock_cls.return_value = instance
            process_directory("/fake", device="cuda")
        mock_lc.assert_called_once()
        # load_config was called with overrides containing device
        assert mock_lc.called

    def test_optimization_mode_override(self):
        """optimization_mode override must be forwarded to load_config."""
        with (
            patch("corridorkey.pipeline.CorridorKeyService") as mock_cls,
            patch("corridorkey.pipeline.load_config") as mock_lc,
        ):
            mock_lc.return_value = MagicMock()
            instance = self._patched_service([])
            mock_cls.return_value = instance
            process_directory("/fake", optimization_mode="speed")
        assert mock_lc.called

    def test_precision_override(self):
        """precision override must be forwarded to load_config."""
        with (
            patch("corridorkey.pipeline.CorridorKeyService") as mock_cls,
            patch("corridorkey.pipeline.load_config") as mock_lc,
        ):
            mock_lc.return_value = MagicMock()
            instance = self._patched_service([])
            mock_cls.return_value = instance
            process_directory("/fake", precision="fp16")
        assert mock_lc.called

    def test_on_clip_done_called_per_clip(self):
        """on_clip_done must be called once per clip with its ClipSummary."""
        clip = _make_clip("shot1", ClipState.COMPLETE)
        done_calls: list = []
        with patch("corridorkey.pipeline.CorridorKeyService") as mock_cls:
            instance = self._patched_service([clip])
            mock_cls.return_value = instance
            process_directory("/fake", on_clip_done=lambda s: done_calls.append(s))
        assert len(done_calls) == 1
        assert done_calls[0].name == "shot1"

    def test_multiple_clips_all_processed(self):
        """All clips in the scan result must be processed."""
        clips = [
            _make_clip("a", ClipState.COMPLETE),
            _make_clip("b", ClipState.COMPLETE),
            _make_clip("c", ClipState.COMPLETE),
        ]
        with patch("corridorkey.pipeline.CorridorKeyService") as mock_cls:
            instance = self._patched_service(clips)
            mock_cls.return_value = instance
            result = process_directory("/fake")
        assert len(result.clips) == 3


class TestProcessClipExtractingPaths:
    """_process_clip - EXTRACTING state branches."""

    def _service(self) -> MagicMock:
        s = MagicMock()
        s.run_inference.return_value = [MagicMock(success=True)] * 3
        return s

    def test_extracting_error_captured(self):
        """A CorridorKeyError from extract_clip must be captured into summary.error."""
        from corridorkey.errors import CorridorKeyError

        clip = _make_clip("shot1", ClipState.EXTRACTING)
        service = self._service()
        service.extract_clip.side_effect = CorridorKeyError("extraction failed")
        summary = _process_clip(clip, service, InferenceParams(), OutputConfig(), None, None, None, None)
        assert summary.error == "extraction failed"

    def test_extracting_to_raw_without_generator_skipped(self):
        """EXTRACTING -> RAW with no generator must be skipped with a warning."""
        clip = _make_clip("shot1", ClipState.EXTRACTING)
        service = self._service()

        def fake_extract(c, **kwargs):
            c.state = ClipState.RAW

        service.extract_clip.side_effect = fake_extract
        warnings: list[str] = []
        summary = _process_clip(
            clip, service, InferenceParams(), OutputConfig(), None, None, lambda m: warnings.append(m), None
        )
        assert summary.skipped is True
        assert any("no alpha generator" in w for w in warnings)

    def test_extracting_to_raw_with_generator_runs_alpha(self):
        """EXTRACTING -> RAW with a generator must run alpha generation."""
        clip = _make_clip("shot1", ClipState.EXTRACTING)
        service = self._service()
        generator = MagicMock()
        generator.name = "mock_gen"

        def fake_extract(c, **kwargs):
            c.state = ClipState.RAW

        def fake_alpha(c, gen, **kwargs):
            c.state = ClipState.READY

        service.extract_clip.side_effect = fake_extract
        service.run_alpha_generator.side_effect = fake_alpha

        summary = _process_clip(clip, service, InferenceParams(), OutputConfig(), generator, None, None, None)
        service.run_alpha_generator.assert_called_once()
        service.run_inference.assert_called_once()
        assert summary.error is None

    def test_extracting_to_raw_alpha_generator_error(self):
        """CorridorKeyError from alpha generation after extraction must be captured."""
        from corridorkey.errors import CorridorKeyError

        clip = _make_clip("shot1", ClipState.EXTRACTING)
        service = self._service()
        generator = MagicMock()

        def fake_extract(c, **kwargs):
            c.state = ClipState.RAW

        service.extract_clip.side_effect = fake_extract
        service.run_alpha_generator.side_effect = CorridorKeyError("alpha gen failed")

        summary = _process_clip(clip, service, InferenceParams(), OutputConfig(), generator, None, None, None)
        assert summary.error == "alpha gen failed"

    def test_extracting_to_raw_alpha_generator_cancelled(self):
        """JobCancelledError from alpha generation after extraction must produce skipped summary."""
        from corridorkey.errors import JobCancelledError

        clip = _make_clip("shot1", ClipState.EXTRACTING)
        service = self._service()
        generator = MagicMock()

        def fake_extract(c, **kwargs):
            c.state = ClipState.RAW

        service.extract_clip.side_effect = fake_extract
        service.run_alpha_generator.side_effect = JobCancelledError("shot1", 0)

        summary = _process_clip(clip, service, InferenceParams(), OutputConfig(), generator, None, None, None)
        assert summary.skipped is True

    def test_inference_cancelled_returns_skipped(self):
        """JobCancelledError from run_inference must produce a skipped summary."""
        from corridorkey.errors import JobCancelledError

        clip = _make_clip("shot1", ClipState.READY)
        service = self._service()
        service.run_inference.side_effect = JobCancelledError("shot1", 0)

        summary = _process_clip(clip, service, InferenceParams(), OutputConfig(), None, None, None, None)
        assert summary.skipped is True

    def test_raw_alpha_generator_cancelled(self):
        """JobCancelledError from alpha generation on RAW clip must produce skipped summary."""
        from corridorkey.errors import JobCancelledError

        clip = _make_clip("shot1", ClipState.RAW)
        service = self._service()
        generator = MagicMock()
        service.run_alpha_generator.side_effect = JobCancelledError("shot1", 0)

        summary = _process_clip(clip, service, InferenceParams(), OutputConfig(), generator, None, None, None)
        assert summary.skipped is True

    def test_raw_alpha_generator_error(self):
        """CorridorKeyError from alpha generation on RAW clip must be captured."""
        from corridorkey.errors import CorridorKeyError

        clip = _make_clip("shot1", ClipState.RAW)
        service = self._service()
        generator = MagicMock()
        service.run_alpha_generator.side_effect = CorridorKeyError("alpha failed")

        summary = _process_clip(clip, service, InferenceParams(), OutputConfig(), generator, None, None, None)
        assert summary.error == "alpha failed"
