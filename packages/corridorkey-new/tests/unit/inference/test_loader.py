"""Unit tests for corridorkey_new.stages.inference.loader helpers.

We test the two pure helper functions that are unit-testable without a
real checkpoint or model: _strip_compiled_prefix and _resize_pos_embeds.
load_model itself requires corridorkey-core and a checkpoint file, so it
is not tested here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from corridorkey_new.stages.inference.loader import _resize_pos_embeds, _strip_compiled_prefix


class TestStripCompiledPrefix:
    def test_strips_prefix(self):
        sd = {"_orig_mod.layer.weight": torch.zeros(4, 4)}
        result = _strip_compiled_prefix(sd)
        assert "layer.weight" in result
        assert "_orig_mod.layer.weight" not in result

    def test_leaves_normal_keys_unchanged(self):
        sd = {"encoder.weight": torch.zeros(4, 4), "decoder.bias": torch.zeros(4)}
        result = _strip_compiled_prefix(sd)
        assert set(result.keys()) == {"encoder.weight", "decoder.bias"}

    def test_mixed_keys(self):
        sd = {
            "_orig_mod.encoder.weight": torch.zeros(2, 2),
            "decoder.bias": torch.zeros(2),
        }
        result = _strip_compiled_prefix(sd)
        assert "encoder.weight" in result
        assert "decoder.bias" in result
        assert len(result) == 2

    def test_empty_dict(self):
        assert _strip_compiled_prefix({}) == {}

    def test_values_preserved(self):
        t = torch.tensor([1.0, 2.0])
        result = _strip_compiled_prefix({"_orig_mod.x": t})
        assert torch.equal(result["x"], t)


class TestResizePosEmbeds:
    def _make_model_with_pos_embed(self, grid: int, embed_dim: int = 8) -> MagicMock:
        """Return a mock model whose state_dict has a pos_embed of shape [1, grid*grid, embed_dim]."""
        seq = grid * grid
        model = MagicMock()
        model.state_dict.return_value = {"pos_embed": torch.zeros(1, seq, embed_dim)}
        return model

    def test_no_resize_when_shapes_match(self):
        grid, dim = 4, 8
        model = self._make_model_with_pos_embed(grid, dim)
        sd = {"pos_embed": torch.zeros(1, grid * grid, dim)}
        result = _resize_pos_embeds(sd, model)
        assert result["pos_embed"].shape == (1, grid * grid, dim)

    def test_resizes_when_grid_differs(self):
        src_grid, dst_grid, dim = 4, 8, 8
        model = self._make_model_with_pos_embed(dst_grid, dim)
        sd = {"pos_embed": torch.randn(1, src_grid * src_grid, dim)}
        result = _resize_pos_embeds(sd, model)
        assert result["pos_embed"].shape == (1, dst_grid * dst_grid, dim)

    def test_non_pos_embed_keys_unchanged(self):
        model = self._make_model_with_pos_embed(4, 8)
        t = torch.zeros(3, 3)
        sd = {"encoder.weight": t, "pos_embed": torch.zeros(1, 16, 8)}
        result = _resize_pos_embeds(sd, model)
        assert torch.equal(result["encoder.weight"], t)

    def test_key_not_in_model_state_left_alone(self):
        """A pos_embed key absent from model state_dict should pass through unchanged."""
        model = MagicMock()
        model.state_dict.return_value = {}  # no matching key
        t = torch.zeros(1, 16, 8)
        sd = {"pos_embed": t}
        result = _resize_pos_embeds(sd, model)
        assert torch.equal(result["pos_embed"], t)
