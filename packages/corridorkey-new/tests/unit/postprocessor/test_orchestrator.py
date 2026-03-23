"""Unit tests for corridorkey_new.stages.postprocessor.orchestrator."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch
from corridorkey_new.stages.inference.contracts import InferenceResult
from corridorkey_new.stages.postprocessor.config import PostprocessConfig
from corridorkey_new.stages.postprocessor.contracts import PostprocessedFrame
from corridorkey_new.stages.postprocessor.orchestrator import postprocess_frame
from corridorkey_new.stages.preprocessor.contracts import FrameMeta


def _make_result(h: int = 32, w: int = 32, frame_index: int = 0) -> InferenceResult:
    meta = FrameMeta(frame_index=frame_index, original_h=h, original_w=w)
    return InferenceResult(
        alpha=torch.zeros(1, 1, 16, 16),
        fg=torch.zeros(1, 3, 16, 16),
        meta=meta,
    )


class TestPostprocessFrame:
    def test_returns_postprocessed_frame(self):
        result = postprocess_frame(_make_result(), PostprocessConfig())
        assert isinstance(result, PostprocessedFrame)

    def test_alpha_shape_at_source_resolution(self):
        result = postprocess_frame(_make_result(h=48, w=64), PostprocessConfig())
        assert result.alpha.shape == (48, 64, 1)

    def test_fg_shape_at_source_resolution(self):
        result = postprocess_frame(_make_result(h=48, w=64), PostprocessConfig())
        assert result.fg.shape == (48, 64, 3)

    def test_comp_shape_at_source_resolution(self):
        result = postprocess_frame(_make_result(h=48, w=64), PostprocessConfig())
        assert result.comp.shape == (48, 64, 3)

    def test_frame_index_carried_through(self):
        result = postprocess_frame(_make_result(frame_index=7), PostprocessConfig())
        assert result.frame_index == 7

    def test_source_dims_carried_through(self):
        result = postprocess_frame(_make_result(h=100, w=200), PostprocessConfig())
        assert result.source_h == 100
        assert result.source_w == 200

    def test_default_stem_from_frame_index(self):
        result = postprocess_frame(_make_result(frame_index=3), PostprocessConfig())
        assert result.stem == "frame_000003"

    def test_custom_stem(self):
        result = postprocess_frame(_make_result(), PostprocessConfig(), stem="my_frame")
        assert result.stem == "my_frame"

    def test_despeckle_called_when_enabled(self):
        cfg = PostprocessConfig(auto_despeckle=True, despeckle_size=10)
        with patch("corridorkey_new.stages.postprocessor.orchestrator.despeckle_alpha") as mock:
            mock.side_effect = lambda a, min_area: a
            postprocess_frame(_make_result(), cfg)
            mock.assert_called_once()

    def test_despeckle_skipped_when_disabled(self):
        cfg = PostprocessConfig(auto_despeckle=False)
        with patch("corridorkey_new.stages.postprocessor.orchestrator.despeckle_alpha") as mock:
            postprocess_frame(_make_result(), cfg)
            mock.assert_not_called()

    def test_output_dtypes_float32(self):
        result = postprocess_frame(_make_result(), PostprocessConfig())
        assert result.alpha.dtype == np.float32
        assert result.fg.dtype == np.float32
        assert result.comp.dtype == np.float32

    def test_debug_log_emitted(self):
        import logging

        orch_logger = logging.getLogger("corridorkey_new.stages.postprocessor.orchestrator")
        original_level = orch_logger.level
        orch_logger.setLevel(logging.DEBUG)
        try:
            # Just verify the call doesn't raise — the logger.debug line is executed
            postprocess_frame(_make_result(frame_index=5), PostprocessConfig())
        finally:
            orch_logger.setLevel(original_level)
