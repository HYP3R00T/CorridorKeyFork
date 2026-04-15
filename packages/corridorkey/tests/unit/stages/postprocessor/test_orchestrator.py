"""Unit tests for corridorkey.stages.postprocessor.orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from corridorkey.stages.inference.contracts import InferenceResult
from corridorkey.stages.postprocessor.config import PostprocessConfig
from corridorkey.stages.postprocessor.contracts import ProcessedFrame
from corridorkey.stages.postprocessor.orchestrator import postprocess_frame
from corridorkey.stages.preprocessor.contracts import FrameMeta


def _make_result(h: int = 32, w: int = 32, frame_index: int = 0, alpha_hint=None, source_image=None) -> InferenceResult:
    meta = FrameMeta(
        frame_index=frame_index, original_h=h, original_w=w, alpha_hint=alpha_hint, source_image=source_image
    )
    return InferenceResult(
        alpha=torch.zeros(1, 1, 16, 16),
        fg=torch.zeros(1, 3, 16, 16),
        meta=meta,
    )


class TestPostprocessFrame:
    def test_returns_postprocessed_frame(self):
        """postprocess_frame returns a ProcessedFrame instance."""
        result = postprocess_frame(_make_result(), PostprocessConfig())
        assert isinstance(result, ProcessedFrame)

    def test_alpha_shape_at_source_resolution(self):
        """alpha field is resized to source resolution [H, W, 1]."""
        result = postprocess_frame(_make_result(h=48, w=64), PostprocessConfig())
        assert result.alpha.shape == (48, 64, 1)

    def test_fg_shape_at_source_resolution(self):
        """fg field is resized to source resolution [H, W, 3]."""
        result = postprocess_frame(_make_result(h=48, w=64), PostprocessConfig())
        assert result.fg.shape == (48, 64, 3)

    def test_comp_shape_at_source_resolution(self):
        """comp field is resized to source resolution [H, W, 3]."""
        result = postprocess_frame(_make_result(h=48, w=64), PostprocessConfig())
        assert result.comp.shape == (48, 64, 3)

    def test_processed_shape_at_source_resolution(self):
        """processed field is [H, W, 4] — premultiplied RGBA at source resolution."""
        result = postprocess_frame(_make_result(h=48, w=64), PostprocessConfig())
        assert result.processed.shape == (48, 64, 4)

    def test_processed_dtype_float32(self):
        """processed field dtype is float32."""
        result = postprocess_frame(_make_result(), PostprocessConfig())
        assert result.processed.dtype == np.float32

    def test_frame_index_carried_through(self):
        """frame_index from FrameMeta is carried through to ProcessedFrame."""
        result = postprocess_frame(_make_result(frame_index=7), PostprocessConfig())
        assert result.frame_index == 7

    def test_source_dims_carried_through(self):
        """source_h and source_w from FrameMeta are carried through."""
        result = postprocess_frame(_make_result(h=100, w=200), PostprocessConfig())
        assert result.source_h == 100
        assert result.source_w == 200

    def test_default_stem_from_frame_index(self):
        """When stem is empty, it defaults to 'frame_{frame_index:06d}'."""
        result = postprocess_frame(_make_result(frame_index=3), PostprocessConfig())
        assert result.stem == "frame_000003"

    def test_custom_stem(self):
        """When stem is provided, it is used as-is."""
        result = postprocess_frame(_make_result(), PostprocessConfig(), stem="my_frame")
        assert result.stem == "my_frame"

    def test_despeckle_called_when_enabled(self):
        """despeckle_alpha is called when auto_despeckle=True."""
        cfg = PostprocessConfig(auto_despeckle=True, despeckle_size=10)
        with patch("corridorkey.stages.postprocessor.orchestrator.despeckle_alpha") as mock:
            mock.side_effect = lambda a, min_area, dilation, blur_size: a
            postprocess_frame(_make_result(), cfg)
            mock.assert_called_once()

    def test_despeckle_skipped_when_disabled(self):
        """despeckle_alpha is not called when auto_despeckle=False."""
        cfg = PostprocessConfig(auto_despeckle=False)
        with patch("corridorkey.stages.postprocessor.orchestrator.despeckle_alpha") as mock:
            postprocess_frame(_make_result(), cfg)
            mock.assert_not_called()

    def test_output_dtypes_float32(self):
        """alpha, fg, and comp fields are all float32."""
        result = postprocess_frame(_make_result(), PostprocessConfig())
        assert result.alpha.dtype == np.float32
        assert result.fg.dtype == np.float32
        assert result.comp.dtype == np.float32

    def test_debug_dump_true_no_output_dir_emits_warning(self, caplog):
        """debug_dump=True with output_dir=None logs a warning."""
        import logging

        cfg = PostprocessConfig(debug_dump=True)
        with caplog.at_level(logging.WARNING, logger="corridorkey.stages.postprocessor.orchestrator"):
            postprocess_frame(_make_result(), cfg, output_dir=None)
        assert any("debug_dump" in r.message for r in caplog.records)


class TestHintSharpenPath:
    def test_hint_sharpen_called_when_enabled_and_hint_present(self):
        """sharpen_with_hint is called when hint_sharpen=True and alpha_hint is set."""
        hint = np.ones((32, 32, 1), dtype=np.float32)
        result = _make_result(alpha_hint=hint)
        cfg = PostprocessConfig(hint_sharpen=True)
        with patch("corridorkey.stages.postprocessor.orchestrator.sharpen_with_hint") as mock:
            mock.side_effect = lambda a, f, h, dilation_px: (a, f)
            postprocess_frame(result, cfg)
            mock.assert_called_once()

    def test_hint_sharpen_skipped_when_disabled(self):
        """sharpen_with_hint is not called when hint_sharpen=False."""
        hint = np.ones((32, 32, 1), dtype=np.float32)
        result = _make_result(alpha_hint=hint)
        cfg = PostprocessConfig(hint_sharpen=False)
        with patch("corridorkey.stages.postprocessor.orchestrator.sharpen_with_hint") as mock:
            postprocess_frame(result, cfg)
            mock.assert_not_called()

    def test_hint_sharpen_skipped_when_no_hint(self):
        """sharpen_with_hint is not called when alpha_hint is None."""
        result = _make_result(alpha_hint=None)
        cfg = PostprocessConfig(hint_sharpen=True)
        with patch("corridorkey.stages.postprocessor.orchestrator.sharpen_with_hint") as mock:
            postprocess_frame(result, cfg)
            mock.assert_not_called()


class TestSourcePassthroughPath:
    def test_source_passthrough_called_when_enabled_and_image_present(self):
        """apply_source_passthrough is called when source_passthrough=True and source_image is set."""
        src = np.zeros((32, 32, 3), dtype=np.float32)
        result = _make_result(source_image=src)
        cfg = PostprocessConfig(source_passthrough=True)
        with patch("corridorkey.stages.postprocessor.orchestrator.apply_source_passthrough") as mock:
            mock.side_effect = lambda fg, alpha, src, erode, blur: fg
            postprocess_frame(result, cfg)
            mock.assert_called_once()

    def test_source_passthrough_skipped_when_disabled(self):
        """apply_source_passthrough is not called when source_passthrough=False."""
        src = np.zeros((32, 32, 3), dtype=np.float32)
        result = _make_result(source_image=src)
        cfg = PostprocessConfig(source_passthrough=False)
        with patch("corridorkey.stages.postprocessor.orchestrator.apply_source_passthrough") as mock:
            postprocess_frame(result, cfg)
            mock.assert_not_called()

    def test_source_passthrough_skipped_when_no_source_image(self):
        """apply_source_passthrough is not called when source_image is None."""
        result = _make_result(source_image=None)
        cfg = PostprocessConfig(source_passthrough=True)
        with patch("corridorkey.stages.postprocessor.orchestrator.apply_source_passthrough") as mock:
            postprocess_frame(result, cfg)
            mock.assert_not_called()


class TestDebugDumpPath:
    def test_debug_dump_writes_files(self, tmp_path: Path):
        """debug_dump=True with output_dir writes PNG files to the debug/ subdirectory."""
        result = _make_result()
        cfg = PostprocessConfig(debug_dump=True, auto_despeckle=True)
        postprocess_frame(result, cfg, output_dir=tmp_path)
        debug_dir = tmp_path / "debug"
        assert debug_dir.is_dir()
        files = list(debug_dir.iterdir())
        assert len(files) > 0

    def test_debug_dump_skipped_when_no_output_dir(self):
        """_debug_write is never called when output_dir is None."""
        result = _make_result()
        cfg = PostprocessConfig(debug_dump=True)
        with patch("corridorkey.stages.postprocessor.orchestrator._debug_write") as mock:
            postprocess_frame(result, cfg, output_dir=None)
            mock.assert_not_called()

    def test_debug_dump_hint_sharpen_writes_extra_files(self, tmp_path: Path):
        """When hint_sharpen is active, hint debug PNGs are written."""
        hint = np.ones((32, 32, 1), dtype=np.float32)
        result = _make_result(alpha_hint=hint)
        cfg = PostprocessConfig(debug_dump=True, hint_sharpen=True)
        postprocess_frame(result, cfg, output_dir=tmp_path)
        debug_dir = tmp_path / "debug"
        names = [f.name for f in debug_dir.iterdir()]
        assert any("hint" in n for n in names)

    def test_debug_dump_source_passthrough_writes_extra_files(self, tmp_path: Path):
        """When source_passthrough is active, passthrough debug PNG is written."""
        src = np.zeros((32, 32, 3), dtype=np.float32)
        result = _make_result(source_image=src)
        cfg = PostprocessConfig(debug_dump=True, source_passthrough=True)
        postprocess_frame(result, cfg, output_dir=tmp_path)
        debug_dir = tmp_path / "debug"
        names = [f.name for f in debug_dir.iterdir()]
        assert any("passthrough" in n for n in names)
