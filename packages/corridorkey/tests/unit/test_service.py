"""Unit tests for service.py - InferenceParams, OutputConfig, and exr_flags.

These tests cover the dataclass contracts and the EXR flag builder without
requiring a GPU, model files, or real frame data. The inference loop itself
is exercised via the pipeline tests; here we focus on the data structures
and helpers that every consumer of CorridorKeyService depends on.
"""

from __future__ import annotations

import cv2
import numpy as np
from corridorkey.contracts import (
    InferenceParams,
    OutputConfig,
)
from corridorkey.writer import EXR_COMPRESSION_IDS, exr_flags


class TestInferenceParams:
    """InferenceParams - defaults, new fields, serialisation roundtrip."""

    def test_defaults(self):
        """All fields must have the documented default values."""
        p = InferenceParams()
        assert p.input_is_linear is False
        assert p.despill_strength == 1.0
        assert p.auto_despeckle is True
        assert p.despeckle_size == 400
        assert p.refiner_scale == 1.0
        assert p.source_passthrough is False
        assert p.edge_erode_px == 3
        assert p.edge_blur_px == 7

    def test_source_passthrough_override(self):
        """source_passthrough must be settable to True."""
        p = InferenceParams(source_passthrough=True)
        assert p.source_passthrough is True

    def test_edge_params_override(self):
        """edge_erode_px and edge_blur_px must accept custom values."""
        p = InferenceParams(edge_erode_px=10, edge_blur_px=15)
        assert p.edge_erode_px == 10
        assert p.edge_blur_px == 15

    def test_to_dict_includes_new_fields(self):
        """to_dict() must include source_passthrough, edge_erode_px, edge_blur_px."""
        p = InferenceParams(source_passthrough=True, edge_erode_px=5, edge_blur_px=9)
        d = p.to_dict()
        assert d["source_passthrough"] is True
        assert d["edge_erode_px"] == 5
        assert d["edge_blur_px"] == 9

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict()) must reconstruct an identical InferenceParams."""
        original = InferenceParams(source_passthrough=True, edge_erode_px=6, edge_blur_px=11, despill_strength=0.5)
        restored = InferenceParams.from_dict(original.to_dict())
        assert restored.source_passthrough == original.source_passthrough
        assert restored.edge_erode_px == original.edge_erode_px
        assert restored.edge_blur_px == original.edge_blur_px
        assert restored.despill_strength == original.despill_strength

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict() must silently ignore keys from future manifest versions."""
        p = InferenceParams.from_dict({"refiner_scale": 0.8, "future_param": "ignored"})
        assert p.refiner_scale == 0.8

    def test_from_dict_missing_keys_use_defaults(self):
        """from_dict() with a partial dict must fill missing fields with defaults."""
        p = InferenceParams.from_dict({})
        assert p.source_passthrough is False
        assert p.edge_erode_px == 3


class TestOutputConfig:
    """OutputConfig - defaults, exr_compression field, serialisation roundtrip."""

    def test_defaults(self):
        """All fields must have the documented default values."""
        cfg = OutputConfig()
        assert cfg.fg_enabled is True
        assert cfg.fg_format == "exr"
        assert cfg.matte_enabled is True
        assert cfg.matte_format == "exr"
        assert cfg.comp_enabled is True
        assert cfg.comp_format == "png"
        assert cfg.processed_enabled is True
        assert cfg.processed_format == "exr"
        assert cfg.exr_compression == "dwaa"

    def test_exr_compression_override(self):
        """exr_compression must accept any valid codec string."""
        for codec in ("dwaa", "pxr24", "zip", "none"):
            cfg = OutputConfig(exr_compression=codec)
            assert cfg.exr_compression == codec

    def test_to_dict_includes_exr_compression(self):
        """to_dict() must include the exr_compression field."""
        cfg = OutputConfig(exr_compression="pxr24")
        d = cfg.to_dict()
        assert d["exr_compression"] == "pxr24"

    def test_from_dict_roundtrip(self):
        """from_dict(to_dict()) must reconstruct an identical OutputConfig."""
        original = OutputConfig(exr_compression="zip", comp_format="exr")
        restored = OutputConfig.from_dict(original.to_dict())
        assert restored.exr_compression == original.exr_compression
        assert restored.comp_format == original.comp_format

    def test_from_dict_ignores_unknown_keys(self):
        """from_dict() must silently ignore keys from future manifest versions."""
        cfg = OutputConfig.from_dict({"fg_format": "png", "future_key": "ignored"})
        assert cfg.fg_format == "png"

    def test_from_dict_missing_exr_compression_uses_default(self):
        """from_dict() without exr_compression must fall back to 'dwaa'."""
        cfg = OutputConfig.from_dict({"fg_format": "exr"})
        assert cfg.exr_compression == "dwaa"

    def test_enabled_outputs_all_on(self):
        """enabled_outputs must list all four names when all outputs are enabled."""
        cfg = OutputConfig()
        assert set(cfg.enabled_outputs) == {"fg", "matte", "comp", "processed"}

    def test_enabled_outputs_partial(self):
        """enabled_outputs must only list outputs whose *_enabled flag is True."""
        cfg = OutputConfig(fg_enabled=True, matte_enabled=False, comp_enabled=True, processed_enabled=False)
        assert set(cfg.enabled_outputs) == {"fg", "comp"}


class TestBuildExrFlags:
    """exr_flags - codec ID mapping and cv2 flag list structure."""

    def test_returns_list(self):
        """exr_flags must return a list (cv2.imwrite flags format)."""
        assert isinstance(exr_flags("dwaa"), list)

    def test_default_is_dwaa(self):
        """Default compression must be DWAA (codec ID 6)."""
        flags = exr_flags("dwaa")
        compression_idx = flags.index(cv2.IMWRITE_EXR_COMPRESSION)
        assert flags[compression_idx + 1] == EXR_COMPRESSION_IDS["dwaa"]

    def test_pxr24_codec_id(self):
        """'pxr24' must map to codec ID 5."""
        flags = exr_flags("pxr24")
        compression_idx = flags.index(cv2.IMWRITE_EXR_COMPRESSION)
        assert flags[compression_idx + 1] == EXR_COMPRESSION_IDS["pxr24"]

    def test_zip_codec_id(self):
        """'zip' must map to codec ID 3."""
        flags = exr_flags("zip")
        compression_idx = flags.index(cv2.IMWRITE_EXR_COMPRESSION)
        assert flags[compression_idx + 1] == EXR_COMPRESSION_IDS["zip"]

    def test_none_codec_id(self):
        """'none' must map to codec ID 0 (uncompressed)."""
        flags = exr_flags("none")
        compression_idx = flags.index(cv2.IMWRITE_EXR_COMPRESSION)
        assert flags[compression_idx + 1] == EXR_COMPRESSION_IDS["none"]

    def test_unknown_codec_falls_back_to_dwaa(self):
        """An unrecognised codec name must silently fall back to DWAA."""
        flags = exr_flags("bogus_codec")
        compression_idx = flags.index(cv2.IMWRITE_EXR_COMPRESSION)
        assert flags[compression_idx + 1] == EXR_COMPRESSION_IDS["dwaa"]

    def test_half_float_type_always_set(self):
        """EXR_TYPE_HALF must always be present regardless of compression choice."""
        for codec in ("dwaa", "pxr24", "zip", "none"):
            flags = exr_flags(codec)
            assert cv2.IMWRITE_EXR_TYPE in flags
            type_idx = flags.index(cv2.IMWRITE_EXR_TYPE)
            assert flags[type_idx + 1] == cv2.IMWRITE_EXR_TYPE_HALF

    def test_case_insensitive(self):
        """Codec names must be accepted in any case."""
        flags_lower = exr_flags("dwaa")
        flags_upper = exr_flags("DWAA")
        assert flags_lower == flags_upper


class TestSourcePassthrough:
    """apply_source_passthrough - blend logic, shape contract, and edge cases.

    Tests call the extracted helper directly with synthetic numpy arrays.
    No engine construction, no torch, no GPU - runs in milliseconds.
    """

    def _make_inputs(self, h: int = 32, w: int = 32):
        """Return (source_srgb, fg_pred, alpha_pred) synthetic arrays."""
        source = np.ones((h, w, 3), dtype=np.float32) * np.array([1.0, 0.0, 0.0])
        fg = np.ones((h, w, 3), dtype=np.float32) * 0.5
        alpha = np.ones((h, w, 1), dtype=np.float32) * 0.9
        return source, fg, alpha

    def test_output_shapes(self):
        """blended_fg must be [H,W,3] and processed_rgba must be [H,W,4]."""
        from corridorkey_core.compositing import apply_source_passthrough

        source, fg, alpha = self._make_inputs()
        blended_fg, processed_rgba = apply_source_passthrough(source, fg, alpha, 1, 3)
        assert blended_fg.shape == (32, 32, 3)
        assert processed_rgba.shape == (32, 32, 4)

    def test_interior_pulled_toward_source(self):
        """With a fully opaque alpha, blended_fg must be closer to source than model fg."""
        from corridorkey_core.compositing import apply_source_passthrough

        source, fg, alpha = self._make_inputs()
        blended_fg, _ = apply_source_passthrough(source, fg, alpha, 1, 3)
        # Source red channel is 1.0, model fg is 0.5 - blend should be > 0.6
        assert blended_fg[:, :, 0].mean() > 0.6

    def test_zero_alpha_uses_model_fg(self):
        """With alpha=0 everywhere, erosion produces no interior - output equals model fg."""
        from corridorkey_core.compositing import apply_source_passthrough

        h, w = 32, 32
        source = np.ones((h, w, 3), dtype=np.float32) * np.array([1.0, 0.0, 0.0])
        fg = np.ones((h, w, 3), dtype=np.float32) * 0.5
        alpha = np.zeros((h, w, 1), dtype=np.float32)
        blended_fg, _ = apply_source_passthrough(source, fg, alpha, 1, 3)
        np.testing.assert_allclose(blended_fg, fg, atol=1e-5)

    def test_processed_rgba_alpha_channel_unchanged(self):
        """The alpha channel of processed_rgba must equal the input alpha_pred."""
        from corridorkey_core.compositing import apply_source_passthrough

        source, fg, alpha = self._make_inputs()
        _, processed_rgba = apply_source_passthrough(source, fg, alpha, 1, 3)
        np.testing.assert_allclose(processed_rgba[:, :, 3:4], alpha, atol=1e-5)

    def test_output_dtype_is_float32(self):
        """Both outputs must be float32 regardless of input values."""
        from corridorkey_core.compositing import apply_source_passthrough

        source, fg, alpha = self._make_inputs()
        blended_fg, processed_rgba = apply_source_passthrough(source, fg, alpha, 2, 5)
        assert blended_fg.dtype == np.float32
        assert processed_rgba.dtype == np.float32

    def test_large_erode_with_small_frame_does_not_crash(self):
        """edge_erode_px larger than the frame must not raise - kernel clamps to 1."""
        from corridorkey_core.compositing import apply_source_passthrough

        source, fg, alpha = self._make_inputs(8, 8)
        blended_fg, _ = apply_source_passthrough(source, fg, alpha, 100, 3)
        assert blended_fg.shape == (8, 8, 3)
