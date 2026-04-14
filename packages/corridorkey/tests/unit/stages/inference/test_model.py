"""Unit tests for corridorkey.stages.inference.model."""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from corridorkey.stages.inference.model import patch_hiera_global_attention


def _make_attn(use_mask_unit_attn: bool, heads: int = 2, head_dim: int = 4) -> nn.Module:
    attn = MagicMock(spec=nn.Module)
    attn.use_mask_unit_attn = use_mask_unit_attn
    attn.heads = heads
    attn.head_dim = head_dim
    attn.q_stride = 1
    attn.dim_out = heads * head_dim
    attn.qkv = nn.Linear(heads * head_dim, 3 * heads * head_dim, bias=False)
    nn.init.eye_(attn.qkv.weight[: heads * head_dim])
    attn.proj = nn.Linear(heads * head_dim, heads * head_dim, bias=False)
    nn.init.eye_(attn.proj.weight)
    return attn


def _make_block(use_mask_unit_attn: bool) -> MagicMock:
    block = MagicMock()
    block.attn = _make_attn(use_mask_unit_attn)
    return block


def _make_hiera(blocks: list) -> MagicMock:
    hiera = MagicMock()
    hiera.blocks = blocks
    return hiera


class TestPatchHieraGlobalAttention:
    def test_patches_global_blocks_only(self):
        """Only blocks with use_mask_unit_attn=False are patched; count matches."""
        global_blk = _make_block(use_mask_unit_attn=False)
        windowed_blk = _make_block(use_mask_unit_attn=True)
        hiera = _make_hiera([global_blk, windowed_blk, global_blk])
        assert patch_hiera_global_attention(hiera) == 2

    def test_windowed_blocks_not_patched(self):
        """Windowed blocks (use_mask_unit_attn=True) are left untouched."""
        windowed_blk = _make_block(use_mask_unit_attn=True)
        original_forward = windowed_blk.attn.forward
        patch_hiera_global_attention(_make_hiera([windowed_blk]))
        assert windowed_blk.attn.forward is original_forward

    def test_returns_zero_when_no_blocks(self):
        """Empty blocks list returns 0 patches."""
        hiera = MagicMock()
        hiera.blocks = []
        assert patch_hiera_global_attention(hiera) == 0

    def test_returns_zero_when_no_blocks_attr(self):
        """Model with no 'blocks' attribute returns 0 patches without raising."""
        assert patch_hiera_global_attention(MagicMock(spec=[])) == 0

    def test_all_windowed_returns_zero(self):
        """All windowed blocks produces zero patches."""
        hiera = _make_hiera([_make_block(True), _make_block(True)])
        assert patch_hiera_global_attention(hiera) == 0

    def test_patched_forward_is_bound_method(self):
        """The replacement forward is a bound method on the attention object."""
        blk = _make_block(use_mask_unit_attn=False)
        patch_hiera_global_attention(_make_hiera([blk]))
        assert isinstance(blk.attn.forward, types.MethodType)

    def test_patched_forward_produces_correct_output_shape(self):
        """Patched forward returns [B, N, dim_out] matching the original contract."""
        heads, head_dim, batch, seq = 2, 4, 1, 16
        dim = heads * head_dim
        attn = _make_attn(use_mask_unit_attn=False, heads=heads, head_dim=head_dim)
        blk = MagicMock()
        blk.attn = attn
        patch_hiera_global_attention(_make_hiera([blk]))
        x = torch.randn(batch, seq, dim)
        with torch.no_grad():
            out = attn.forward(x)
        assert out.shape == (batch, seq, dim)

    def test_patched_forward_output_is_deterministic(self):
        """Same input produces the same output — no stochastic ops in the patched forward."""
        heads, head_dim, batch, seq = 2, 4, 1, 8
        dim = heads * head_dim
        attn = _make_attn(use_mask_unit_attn=False, heads=heads, head_dim=head_dim)
        blk = MagicMock()
        blk.attn = attn
        patch_hiera_global_attention(_make_hiera([blk]))
        x = torch.randn(batch, seq, dim)
        with torch.no_grad():
            assert torch.allclose(attn.forward(x), attn.forward(x))

    def test_patch_is_idempotent(self):
        """Patching the same model twice does not raise and returns a valid count both times."""
        blk = _make_block(use_mask_unit_attn=False)
        hiera = _make_hiera([blk])
        assert patch_hiera_global_attention(hiera) == 1
        assert patch_hiera_global_attention(hiera) == 1
