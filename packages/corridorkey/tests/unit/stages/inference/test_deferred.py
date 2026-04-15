"""Unit tests for corridorkey.stages.inference.deferred — DeferredTransfer."""

from __future__ import annotations

import torch
from corridorkey.stages.inference.deferred import DeferredTransfer
from corridorkey.stages.preprocessor.contracts import FrameMeta


def _meta() -> FrameMeta:
    return FrameMeta(frame_index=0, original_h=32, original_w=32)


def _alpha(h: int = 8, w: int = 8) -> torch.Tensor:
    return torch.rand(1, 1, h, w)


def _fg(h: int = 8, w: int = 8) -> torch.Tensor:
    return torch.rand(1, 3, h, w)


class TestDeferredTransferMeta:
    def test_meta_available_before_resolve(self):
        """meta property returns the FrameMeta without calling resolve()."""
        meta = _meta()
        transfer = DeferredTransfer(alpha=_alpha(), fg=_fg(), meta=meta)
        assert transfer.meta is meta

    def test_meta_frame_index_preserved(self):
        """meta.frame_index is accessible directly from the handle."""
        meta = FrameMeta(frame_index=7, original_h=64, original_w=64)
        transfer = DeferredTransfer(alpha=_alpha(), fg=_fg(), meta=meta)
        assert transfer.meta.frame_index == 7


class TestDeferredTransferStartSynchronous:
    def test_start_with_no_copy_stream_returns_handle(self):
        """start() with copy_stream=None returns a DeferredTransfer without raising."""
        meta = _meta()
        handle = DeferredTransfer.start(_alpha(), _fg(), meta, copy_stream=None)
        assert isinstance(handle, DeferredTransfer)

    def test_start_with_cpu_tensor_returns_handle(self):
        """CPU tensors always take the synchronous path regardless of copy_stream."""
        meta = _meta()
        alpha = _alpha()  # CPU tensor
        assert alpha.device.type == "cpu"
        handle = DeferredTransfer.start(alpha, _fg(), meta, copy_stream=None)
        assert isinstance(handle, DeferredTransfer)

    def test_start_meta_preserved(self):
        """meta is accessible on the returned handle before resolve()."""
        meta = _meta()
        handle = DeferredTransfer.start(_alpha(), _fg(), meta, copy_stream=None)
        assert handle.meta is meta


class TestDeferredTransferResolveSynchronous:
    def test_resolve_returns_three_tuple(self):
        """resolve() on the synchronous path returns (alpha, fg, meta)."""
        meta = _meta()
        handle = DeferredTransfer.start(_alpha(), _fg(), meta, copy_stream=None)
        result = handle.resolve()
        assert len(result) == 3

    def test_resolve_alpha_is_cpu(self):
        """Resolved alpha tensor is on CPU."""
        handle = DeferredTransfer.start(_alpha(), _fg(), _meta(), copy_stream=None)
        alpha_cpu, _, _ = handle.resolve()
        assert alpha_cpu.device.type == "cpu"

    def test_resolve_fg_is_cpu(self):
        """Resolved fg tensor is on CPU."""
        handle = DeferredTransfer.start(_alpha(), _fg(), _meta(), copy_stream=None)
        _, fg_cpu, _ = handle.resolve()
        assert fg_cpu.device.type == "cpu"

    def test_resolve_alpha_is_float32(self):
        """Resolved alpha is always float32 regardless of input dtype."""
        alpha = _alpha().half()
        handle = DeferredTransfer.start(alpha, _fg(), _meta(), copy_stream=None)
        alpha_cpu, _, _ = handle.resolve()
        assert alpha_cpu.dtype == torch.float32

    def test_resolve_fg_is_float32(self):
        """Resolved fg is always float32 regardless of input dtype."""
        fg = _fg().half()
        handle = DeferredTransfer.start(_alpha(), fg, _meta(), copy_stream=None)
        _, fg_cpu, _ = handle.resolve()
        assert fg_cpu.dtype == torch.float32

    def test_resolve_meta_is_same_object(self):
        """Resolved meta is the exact same FrameMeta passed to start()."""
        meta = _meta()
        handle = DeferredTransfer.start(_alpha(), _fg(), meta, copy_stream=None)
        _, _, resolved_meta = handle.resolve()
        assert resolved_meta is meta

    def test_resolve_alpha_values_preserved(self):
        """Alpha tensor values survive the synchronous CPU transfer."""
        alpha = torch.full((1, 1, 4, 4), 0.75)
        handle = DeferredTransfer.start(alpha, _fg(4, 4), _meta(), copy_stream=None)
        alpha_cpu, _, _ = handle.resolve()
        assert torch.allclose(alpha_cpu, torch.full_like(alpha_cpu, 0.75))

    def test_resolve_fg_values_preserved(self):
        """FG tensor values survive the synchronous CPU transfer."""
        fg = torch.full((1, 3, 4, 4), 0.5)
        handle = DeferredTransfer.start(_alpha(4, 4), fg, _meta(), copy_stream=None)
        _, fg_cpu, _ = handle.resolve()
        assert torch.allclose(fg_cpu, torch.full_like(fg_cpu, 0.5))

    def test_resolve_alpha_shape_preserved(self):
        """Alpha shape is unchanged after synchronous transfer."""
        alpha = _alpha(16, 24)
        handle = DeferredTransfer.start(alpha, _fg(16, 24), _meta(), copy_stream=None)
        alpha_cpu, _, _ = handle.resolve()
        assert alpha_cpu.shape == (1, 1, 16, 24)

    def test_resolve_fg_shape_preserved(self):
        """FG shape is unchanged after synchronous transfer."""
        fg = _fg(16, 24)
        handle = DeferredTransfer.start(_alpha(16, 24), fg, _meta(), copy_stream=None)
        _, fg_cpu, _ = handle.resolve()
        assert fg_cpu.shape == (1, 3, 16, 24)
