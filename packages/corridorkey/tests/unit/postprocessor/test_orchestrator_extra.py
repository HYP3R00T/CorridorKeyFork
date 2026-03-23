"""Additional orchestrator tests covering hint_sharpen, source_passthrough, and debug_dump paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from corridorkey.stages.inference.contracts import InferenceResult
from corridorkey.stages.postprocessor.config import PostprocessConfig
from corridorkey.stages.postprocessor.orchestrator import postprocess_frame
from corridorkey.stages.preprocessor.contracts import FrameMeta


def _make_result(h: int = 32, w: int = 32, frame_index: int = 0, alpha_hint=None, source_image=None) -> InferenceResult:
    meta = FrameMeta(
        frame_index=frame_index,
        original_h=h,
        original_w=w,
        alpha_hint=alpha_hint,
        source_image=source_image,
    )
    return InferenceResult(
        alpha=torch.zeros(1, 1, 16, 16),
        fg=torch.zeros(1, 3, 16, 16),
        meta=meta,
    )


class TestHintSharpenPath:
    def test_hint_sharpen_called_when_enabled_and_hint_present(self):
        hint = np.ones((32, 32, 1), dtype=np.float32)
        result = _make_result(alpha_hint=hint)
        cfg = PostprocessConfig(hint_sharpen=True)
        with patch("corridorkey.stages.postprocessor.orchestrator.sharpen_with_hint") as mock:
            mock.side_effect = lambda a, f, h, dilation_px: (a, f)
            postprocess_frame(result, cfg)
            mock.assert_called_once()

    def test_hint_sharpen_skipped_when_disabled(self):
        hint = np.ones((32, 32, 1), dtype=np.float32)
        result = _make_result(alpha_hint=hint)
        cfg = PostprocessConfig(hint_sharpen=False)
        with patch("corridorkey.stages.postprocessor.orchestrator.sharpen_with_hint") as mock:
            postprocess_frame(result, cfg)
            mock.assert_not_called()

    def test_hint_sharpen_skipped_when_no_hint(self):
        result = _make_result(alpha_hint=None)
        cfg = PostprocessConfig(hint_sharpen=True)
        with patch("corridorkey.stages.postprocessor.orchestrator.sharpen_with_hint") as mock:
            postprocess_frame(result, cfg)
            mock.assert_not_called()


class TestSourcePassthroughPath:
    def test_source_passthrough_called_when_enabled_and_image_present(self):
        src = np.zeros((32, 32, 3), dtype=np.float32)
        result = _make_result(source_image=src)
        cfg = PostprocessConfig(source_passthrough=True)
        with patch("corridorkey.stages.postprocessor.orchestrator.apply_source_passthrough") as mock:
            mock.side_effect = lambda fg, alpha, src, erode, blur: fg
            postprocess_frame(result, cfg)
            mock.assert_called_once()

    def test_source_passthrough_skipped_when_disabled(self):
        src = np.zeros((32, 32, 3), dtype=np.float32)
        result = _make_result(source_image=src)
        cfg = PostprocessConfig(source_passthrough=False)
        with patch("corridorkey.stages.postprocessor.orchestrator.apply_source_passthrough") as mock:
            postprocess_frame(result, cfg)
            mock.assert_not_called()

    def test_source_passthrough_skipped_when_no_source_image(self):
        result = _make_result(source_image=None)
        cfg = PostprocessConfig(source_passthrough=True)
        with patch("corridorkey.stages.postprocessor.orchestrator.apply_source_passthrough") as mock:
            postprocess_frame(result, cfg)
            mock.assert_not_called()


class TestDebugDumpPath:
    def test_debug_dump_writes_files(self, tmp_path: Path):
        result = _make_result()
        cfg = PostprocessConfig(debug_dump=True, auto_despeckle=True)
        postprocess_frame(result, cfg, output_dir=tmp_path)
        debug_dir = tmp_path / "debug"
        assert debug_dir.is_dir()
        files = list(debug_dir.iterdir())
        assert len(files) > 0

    def test_debug_dump_skipped_when_no_output_dir(self):
        result = _make_result()
        cfg = PostprocessConfig(debug_dump=True)
        with patch("corridorkey.stages.postprocessor.orchestrator._debug_write") as mock:
            postprocess_frame(result, cfg, output_dir=None)
            mock.assert_not_called()

    def test_debug_dump_hint_sharpen_writes_extra_files(self, tmp_path: Path):
        hint = np.ones((32, 32, 1), dtype=np.float32)
        result = _make_result(alpha_hint=hint)
        cfg = PostprocessConfig(debug_dump=True, hint_sharpen=True)
        postprocess_frame(result, cfg, output_dir=tmp_path)
        debug_dir = tmp_path / "debug"
        names = [f.name for f in debug_dir.iterdir()]
        assert any("hint" in n for n in names)

    def test_debug_dump_source_passthrough_writes_extra_files(self, tmp_path: Path):
        src = np.zeros((32, 32, 3), dtype=np.float32)
        result = _make_result(source_image=src)
        cfg = PostprocessConfig(debug_dump=True, source_passthrough=True)
        postprocess_frame(result, cfg, output_dir=tmp_path)
        debug_dir = tmp_path / "debug"
        names = [f.name for f in debug_dir.iterdir()]
        assert any("passthrough" in n for n in names)
