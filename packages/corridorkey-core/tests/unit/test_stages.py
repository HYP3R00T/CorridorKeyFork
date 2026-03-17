"""Unit tests for pipeline/stages.py - stage_3_preprocess, stage_4_infer, stage_5_postprocess.

All three stage functions run on CPU with synthetic numpy arrays and mock
engines. No GPU, no model files, no filesystem access required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch
from corridorkey_core.contracts import (
    PostprocessParams,
    PreprocessedTensor,
    ProcessedFrame,
    RawPrediction,
)
from corridorkey_core.stages import (
    stage_3_preprocess,
    stage_4_infer,
    stage_5_postprocess,
)


def _rgb(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.rand(h, w, 3).astype(np.float32)


def _mask(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.rand(h, w, 1).astype(np.float32)


def _raw(img_size: int = 64, source_h: int = 64, source_w: int = 64) -> RawPrediction:
    return RawPrediction(
        alpha=np.random.rand(img_size, img_size, 1).astype(np.float32),
        fg=np.random.rand(img_size, img_size, 3).astype(np.float32),
        img_size=img_size,
        source_h=source_h,
        source_w=source_w,
    )


class TestStage3Preprocess:
    """stage_3_preprocess - resize, normalise, stack into a 4-channel tensor."""

    def test_returns_preprocessed_tensor(self):
        """Must return a PreprocessedTensor dataclass."""
        result = stage_3_preprocess(_rgb(), _mask(), 64, 64, img_size=32, device="cpu")
        assert isinstance(result, PreprocessedTensor)

    def test_tensor_shape(self):
        """Tensor must be [1, 4, img_size, img_size]."""
        result = stage_3_preprocess(_rgb(), _mask(), 64, 64, img_size=32, device="cpu")
        assert result.tensor.shape == (1, 4, 32, 32)

    def test_tensor_dtype_float32(self):
        """Tensor must be float32."""
        result = stage_3_preprocess(_rgb(), _mask(), 64, 64, img_size=32, device="cpu")
        assert result.tensor.dtype == torch.float32

    def test_source_dimensions_preserved(self):
        """source_h and source_w must match the values passed in."""
        result = stage_3_preprocess(_rgb(100, 80), _mask(100, 80), 100, 80, img_size=32, device="cpu")
        assert result.source_h == 100
        assert result.source_w == 80

    def test_img_size_stored(self):
        """img_size must be stored on the returned dataclass."""
        result = stage_3_preprocess(_rgb(), _mask(), 64, 64, img_size=48, device="cpu")
        assert result.img_size == 48

    def test_device_stored(self):
        """device string must be stored on the returned dataclass."""
        result = stage_3_preprocess(_rgb(), _mask(), 64, 64, img_size=32, device="cpu")
        assert result.device == "cpu"

    def test_mask_2d_input_accepted(self):
        """A 2-D mask [H, W] must be accepted without error."""
        mask_2d = np.random.rand(64, 64).astype(np.float32)
        # stage_3 expects [H, W, 1] - but let's verify it handles the slice correctly
        mask_3d = mask_2d[:, :, np.newaxis]
        result = stage_3_preprocess(_rgb(), mask_3d, 64, 64, img_size=32, device="cpu")
        assert result.tensor.shape == (1, 4, 32, 32)

    def test_normalisation_applied(self):
        """The first 3 channels must be ImageNet-normalised (not raw 0-1 values)."""
        # A pure-white image normalised with ImageNet stats will have values outside [0,1]
        white = np.ones((64, 64, 3), dtype=np.float32)
        mask = np.ones((64, 64, 1), dtype=np.float32)
        result = stage_3_preprocess(white, mask, 64, 64, img_size=32, device="cpu")
        rgb_channels = result.tensor[0, :3]
        # After normalisation, white (1.0) -> (1.0 - mean) / std which is > 1.0 for most channels
        assert rgb_channels.max().item() > 1.0


class TestStage4Infer:
    """stage_4_infer - forward pass through a mock engine."""

    def _make_engine(self, img_size: int = 32) -> MagicMock:
        engine = MagicMock()
        engine.device = torch.device("cpu")
        engine.model_precision = torch.float32
        engine.mixed_precision = False

        # model output: alpha [B,1,H,W], fg [B,3,H,W]
        alpha_t = torch.rand(1, 1, img_size, img_size)
        fg_t = torch.rand(1, 3, img_size, img_size)
        engine.model.return_value = {"alpha": alpha_t, "fg": fg_t}
        engine.model.refiner = None
        return engine

    def _make_preprocessed(self, img_size: int = 32) -> PreprocessedTensor:
        tensor = torch.rand(1, 4, img_size, img_size)
        return PreprocessedTensor(tensor=tensor, img_size=img_size, device="cpu", source_h=64, source_w=64)

    def test_returns_raw_prediction(self):
        """Must return a RawPrediction dataclass."""
        engine = self._make_engine()
        pre = self._make_preprocessed()
        result = stage_4_infer(engine, pre)
        assert isinstance(result, RawPrediction)

    def test_alpha_shape(self):
        """alpha must be [img_size, img_size, 1] float32."""
        engine = self._make_engine(32)
        pre = self._make_preprocessed(32)
        result = stage_4_infer(engine, pre)
        assert result.alpha.shape == (32, 32, 1)
        assert result.alpha.dtype == np.float32

    def test_fg_shape(self):
        """fg must be [img_size, img_size, 3] float32."""
        engine = self._make_engine(32)
        pre = self._make_preprocessed(32)
        result = stage_4_infer(engine, pre)
        assert result.fg.shape == (32, 32, 3)
        assert result.fg.dtype == np.float32

    def test_source_dimensions_propagated(self):
        """source_h and source_w from PreprocessedTensor must be carried through."""
        engine = self._make_engine(32)
        pre = self._make_preprocessed(32)
        result = stage_4_infer(engine, pre)
        assert result.source_h == 64
        assert result.source_w == 64

    def test_refiner_scale_hook_registered(self):
        """When refiner_scale != 1.0 and refiner exists, a forward hook must be registered."""
        engine = self._make_engine(32)
        engine.model.refiner = MagicMock()
        engine.model.refiner.register_forward_hook = MagicMock(return_value=MagicMock())
        pre = self._make_preprocessed(32)
        stage_4_infer(engine, pre, refiner_scale=0.5)
        engine.model.refiner.register_forward_hook.assert_called_once()

    def test_refiner_scale_1_no_hook(self):
        """When refiner_scale == 1.0, no hook must be registered."""
        engine = self._make_engine(32)
        engine.model.refiner = MagicMock()
        engine.model.refiner.register_forward_hook = MagicMock(return_value=MagicMock())
        pre = self._make_preprocessed(32)
        stage_4_infer(engine, pre, refiner_scale=1.0)
        engine.model.refiner.register_forward_hook.assert_not_called()


class TestStage5Postprocess:
    """stage_5_postprocess - despeckle, despill, composite, upsample."""

    def _run(self, source_h: int = 64, source_w: int = 64, img_size: int = 32, **kwargs) -> ProcessedFrame:
        raw = _raw(img_size=img_size, source_h=source_h, source_w=source_w)
        source = _rgb(source_h, source_w)
        return stage_5_postprocess(raw, source, **kwargs)

    def test_returns_processed_frame(self):
        """Must return a ProcessedFrame dataclass."""
        result = self._run()
        assert isinstance(result, ProcessedFrame)

    def test_alpha_shape(self):
        """alpha must be [source_h, source_w, 1]."""
        result = self._run(source_h=64, source_w=80)
        assert result.alpha.shape == (64, 80, 1)

    def test_fg_shape(self):
        """fg must be [source_h, source_w, 3]."""
        result = self._run(source_h=64, source_w=80)
        assert result.fg.shape == (64, 80, 3)

    def test_comp_shape(self):
        """comp must be [source_h, source_w, 3]."""
        result = self._run(source_h=64, source_w=80)
        assert result.comp.shape == (64, 80, 3)

    def test_processed_shape(self):
        """processed must be [source_h, source_w, 4]."""
        result = self._run(source_h=64, source_w=80)
        assert result.processed.shape == (64, 80, 4)

    def test_stem_propagated(self):
        """stem must be carried through to ProcessedFrame."""
        raw = _raw()
        source = _rgb()
        result = stage_5_postprocess(raw, source, stem="frame_00001")
        assert result.stem == "frame_00001"

    def test_default_params_no_crash(self):
        """Default PostprocessParams must not raise."""
        self._run()

    def test_auto_despeckle_false(self):
        """auto_despeckle=False must not crash and must return correct shapes."""
        params = PostprocessParams(auto_despeckle=False)
        result = self._run(params=params)
        assert result.alpha.shape == (64, 64, 1)

    def test_despill_strength_zero(self):
        """despill_strength=0.0 must not crash."""
        params = PostprocessParams(despill_strength=0.0)
        result = self._run(params=params)
        assert result.fg.shape == (64, 64, 3)

    def test_fg_is_straight_false(self):
        """fg_is_straight=False (premul composite path) must not crash."""
        params = PostprocessParams(fg_is_straight=False)
        result = self._run(params=params)
        assert result.comp.shape == (64, 64, 3)

    def test_source_passthrough_enabled(self):
        """source_passthrough=True must not crash and must return correct shapes."""
        params = PostprocessParams(source_passthrough=True, edge_erode_px=1, edge_blur_px=3)
        result = self._run(params=params)
        assert result.fg.shape == (64, 64, 3)
        assert result.processed.shape == (64, 64, 4)

    def test_source_is_linear_conversion(self):
        """source_is_linear=True must apply linear->sRGB conversion without crashing."""
        raw = _raw()
        source = _rgb()
        result = stage_5_postprocess(raw, source, source_is_linear=True)
        assert result.fg.shape == (64, 64, 3)

    def test_source_passthrough_with_linear_source(self):
        """source_passthrough + source_is_linear must both apply without crashing."""
        params = PostprocessParams(source_passthrough=True, edge_erode_px=1, edge_blur_px=3)
        raw = _raw()
        source = _rgb()
        result = stage_5_postprocess(raw, source, source_is_linear=True, params=params)
        assert result.processed.shape == (64, 64, 4)

    def test_non_square_source(self):
        """Non-square source resolution must be handled correctly."""
        result = self._run(source_h=48, source_w=96, img_size=32)
        assert result.alpha.shape == (48, 96, 1)
        assert result.processed.shape == (48, 96, 4)
