"""Deferred GPU→CPU DMA transfer for inference results.

After the model forward pass, alpha and fg tensors sit on the GPU. The
postprocessor needs them on CPU. Normally this transfer is synchronous —
the inference thread blocks until the copy completes before starting the
next frame's forward pass.

``DeferredTransfer`` starts the copy on a dedicated CUDA copy stream
immediately after inference and returns a handle. The inference thread
can then start the next frame's forward pass on the compute stream while
the DMA runs in parallel on the copy stream. The postwrite worker calls
``resolve()`` when it needs the data, which blocks only on the specific
CUDA event for that transfer.

On CPU or MPS devices the deferred path is a no-op — ``resolve()``
returns the tensors directly.

Usage::

    # In the inference worker (after model forward):
    transfer = DeferredTransfer.start(alpha, fg, meta, copy_stream)

    # In the postwrite worker (when ready to postprocess):
    alpha_t, fg_t, meta = transfer.resolve()
"""

from __future__ import annotations

import threading

import torch

from corridorkey.stages.preprocessor.contracts import FrameMeta


class DeferredTransfer:
    """Handle for a deferred GPU→CPU DMA transfer.

    Created by :meth:`start` immediately after the model forward pass.
    Call :meth:`resolve` to block until the DMA completes and get the
    tensors ready for postprocessing.

    On non-CUDA devices the transfer is synchronous and ``resolve()``
    returns immediately.
    """

    def __init__(
        self,
        alpha: torch.Tensor,
        fg: torch.Tensor,
        meta: FrameMeta,
        event: torch.cuda.Event | None = None,
        pinned_alpha: torch.Tensor | None = None,
        pinned_fg: torch.Tensor | None = None,
        released: threading.Event | None = None,
    ) -> None:
        self._alpha = alpha
        self._fg = fg
        self._meta = meta
        self._event = event
        self._pinned_alpha = pinned_alpha
        self._pinned_fg = pinned_fg
        self._released = released

    @property
    def meta(self) -> FrameMeta:
        """FrameMeta for this transfer — available before resolve()."""
        return self._meta

    @classmethod
    def start(
        cls,
        alpha: torch.Tensor,
        fg: torch.Tensor,
        meta: FrameMeta,
        copy_stream: torch.cuda.Stream | None,
    ) -> DeferredTransfer:
        """Start a non-blocking DMA from device to pinned CPU memory.

        Args:
            alpha: [1, 1, H, W] tensor on CUDA.
            fg: [1, 3, H, W] tensor on CUDA.
            meta: FrameMeta carried through from preprocessing.
            copy_stream: Dedicated CUDA copy stream. If None or device is
                not CUDA, falls back to synchronous transfer.

        Returns:
            DeferredTransfer handle. Call resolve() when ready.
        """
        if copy_stream is None or alpha.device.type != "cuda":
            # CPU/MPS path — no async DMA, just hold the tensors.
            return cls(alpha=alpha, fg=fg, meta=meta)

        # Allocate pinned CPU buffers for the DMA destination.
        pinned_alpha = torch.empty(alpha.shape, dtype=alpha.dtype, pin_memory=True)
        pinned_fg = torch.empty(fg.shape, dtype=fg.dtype, pin_memory=True)
        released = threading.Event()

        # Wait for the compute stream to finish before starting the copy.
        compute_event = torch.cuda.current_stream(alpha.device).record_event()
        copy_stream.wait_event(compute_event)

        # Start non-blocking DMA on the copy stream.
        with torch.cuda.stream(copy_stream):
            pinned_alpha.copy_(alpha, non_blocking=True)
            pinned_fg.copy_(fg, non_blocking=True)

        # Record an event on the copy stream so resolve() can wait on
        # exactly this transfer, not the entire stream.
        dma_event = copy_stream.record_event()

        return cls(
            alpha=alpha,
            fg=fg,
            meta=meta,
            event=dma_event,
            pinned_alpha=pinned_alpha,
            pinned_fg=pinned_fg,
            released=released,
        )

    def resolve(self) -> tuple[torch.Tensor, torch.Tensor, FrameMeta]:
        """Block until the DMA completes and return CPU tensors.

        Returns:
            (alpha, fg, meta) — alpha and fg are float32 CPU tensors.
        """
        if self._event is None:
            # Synchronous path (CPU/MPS or no copy stream).
            return self._alpha.cpu().float(), self._fg.cpu().float(), self._meta

        # Block only on this transfer's event, not the whole stream.
        self._event.synchronize()

        alpha_cpu = self._pinned_alpha.float()  # type: ignore[union-attr]
        fg_cpu = self._pinned_fg.float()  # type: ignore[union-attr]

        if self._released is not None:
            self._released.set()

        return alpha_cpu, fg_cpu, self._meta
