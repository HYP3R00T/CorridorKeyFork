"""Unit tests for patch_hiera_global_attention.

Tests the patching logic in isolation using lightweight mock objects that
replicate the relevant Hiera block/attention structure — no timm or GPU needed.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import torch
import torch.nn as nn
from corridorkey.stages.inference.model import patch_hiera_global_attention

# ---------------------------------------------------------------------------
# Helpers — minimal Hiera-like stubs
# ---------------------------------------------------------------------------


def _make_attn(use_mask_unit_attn: bool, heads: int = 2, head_dim: int = 4, q_stride: int = 1) -> nn.Module:
    """Return a minimal attention stub with the attributes the patch reads."""
    attn = MagicMock(spec=nn.Module)
    attn.use_mask_unit_attn = use_mask_unit_attn
    attn.heads = heads
    attn.head_dim = head_dim
    attn.q_stride = q_stride
    attn.dim_out = heads * head_dim
    # qkv projects [B, N, C] -> [B, N, 3 * heads * head_dim]
    attn.qkv = nn.Linear(heads * head_dim, 3 * heads * head_dim, bias=False)
    nn.init.eye_(attn.qkv.weight[: heads * head_dim])  # stable init
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPatchHieraGlobalAttention:
    def test_patches_global_blocks_only(self):
        """Only blocks with use_mask_unit_attn=False should be patched."""
        global_blk = _make_block(use_mask_unit_attn=False)
        windowed_blk = _make_block(use_mask_unit_attn=True)
        hiera = _make_hiera([global_blk, windowed_blk, global_blk])

        n = patch_hiera_global_attention(hiera)

        assert n == 2

    def test_windowed_blocks_not_patched(self):
        """Windowed blocks must not be touched."""
        windowed_blk = _make_block(use_mask_unit_attn=True)
        original_forward = windowed_blk.attn.forward
        hiera = _make_hiera([windowed_blk])

        patch_hiera_global_attention(hiera)

        assert windowed_blk.attn.forward is original_forward

    def test_returns_zero_when_no_blocks(self):
        hiera = MagicMock()
        hiera.blocks = []
        assert patch_hiera_global_attention(hiera) == 0

    def test_returns_zero_when_no_blocks_attr(self):
        hiera = MagicMock(spec=[])  # no 'blocks' attribute
        assert patch_hiera_global_attention(hiera) == 0

    def test_patched_forward_is_bound_method(self):
        """The replacement forward must be a bound method on the attn object."""
        blk = _make_block(use_mask_unit_attn=False)
        hiera = _make_hiera([blk])
        patch_hiera_global_attention(hiera)
        assert isinstance(blk.attn.forward, types.MethodType)

    def test_patched_forward_produces_correct_output_shape(self):
        """Patched forward must return [B, N, dim_out] matching the original contract."""
        heads, head_dim, batch, seq = 2, 4, 1, 16
        dim = heads * head_dim

        attn = _make_attn(use_mask_unit_attn=False, heads=heads, head_dim=head_dim)
        blk = MagicMock()
        blk.attn = attn
        hiera = _make_hiera([blk])

        patch_hiera_global_attention(hiera)

        x = torch.randn(batch, seq, dim)
        with torch.no_grad():
            out = attn.forward(x)

        assert out.shape == (batch, seq, dim)

    def test_patched_forward_output_is_deterministic(self):
        """Same input must produce the same output (no stochastic ops)."""
        heads, head_dim, batch, seq = 2, 4, 1, 8
        dim = heads * head_dim

        attn = _make_attn(use_mask_unit_attn=False, heads=heads, head_dim=head_dim)
        blk = MagicMock()
        blk.attn = attn
        hiera = _make_hiera([blk])
        patch_hiera_global_attention(hiera)

        x = torch.randn(batch, seq, dim)
        with torch.no_grad():
            out1 = attn.forward(x)
            out2 = attn.forward(x)

        assert torch.allclose(out1, out2)

    def test_all_windowed_returns_zero(self):
        """All windowed blocks → 0 patches."""
        hiera = _make_hiera([_make_block(True), _make_block(True)])
        assert patch_hiera_global_attention(hiera) == 0

    def test_patch_is_idempotent(self):
        """Patching twice must not raise and must still return a valid count."""
        blk = _make_block(use_mask_unit_attn=False)
        hiera = _make_hiera([blk])
        n1 = patch_hiera_global_attention(hiera)
        n2 = patch_hiera_global_attention(hiera)
        assert n1 == 1
        assert n2 == 1  # second pass re-patches the already-patched forward
