"""Inference stage — backend protocol and implementations.

Defines the ``ModelBackend`` protocol that all inference backends must satisfy,
plus the two concrete implementations:

  - ``TorchBackend``  — wraps the existing GreenFormer loader + orchestrator.
  - ``MLXBackend``    — wraps ``corridorkey-mlx`` (Apple Silicon only, optional).

Callers should use ``factory.load_backend()`` rather than instantiating these
directly. The protocol ensures both backends are interchangeable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from corridorkey_new.stages.inference.config import InferenceConfig
    from corridorkey_new.stages.inference.contracts import InferenceResult
    from corridorkey_new.stages.preprocessor.contracts import PreprocessedFrame

logger = logging.getLogger(__name__)


@runtime_checkable
class ModelBackend(Protocol):
    """Contract for all inference backends.

    Any object that satisfies this protocol can be used as a drop-in
    replacement for the default PyTorch backend. Implementations must be
    stateless with respect to individual frames — all per-frame state lives
    in ``PreprocessedFrame`` and ``InferenceResult``.
    """

    @property
    def backend_name(self) -> str:
        """Human-readable backend identifier, e.g. ``"torch"`` or ``"mlx"``."""
        ...

    @property
    def resolved_config(self) -> dict[str, str]:
        """What was actually resolved at runtime (device, precision, mode, etc.).

        Returned as a flat ``str -> str`` dict so any interface can display it
        without importing torch or mlx types.

        Example::

            {
                "backend": "torch",
                "device": "cuda",
                "optimization_mode": "lowvram",
                "precision": "bfloat16",
                "img_size": "2048",
            }
        """
        ...

    def run(self, frame: PreprocessedFrame) -> InferenceResult:
        """Run inference on a single preprocessed frame.

        Args:
            frame: Output of the preprocessing stage.

        Returns:
            InferenceResult with alpha and fg tensors.
        """
        ...


class TorchBackend:
    """PyTorch inference backend.

    Wraps the existing ``load_model`` + ``run_inference`` functions so they
    satisfy the ``ModelBackend`` protocol. This is the default backend on all
    platforms.

    Args:
        config: Fully resolved ``InferenceConfig`` (no "auto" values remaining).
        model: Loaded GreenFormer in eval mode, already on the correct device.
    """

    def __init__(self, config: InferenceConfig, model: nn.Module) -> None:
        self._config = config
        self._model = model

    @property
    def backend_name(self) -> str:
        return "torch"

    @property
    def resolved_config(self) -> dict[str, str]:
        cfg = self._config
        return {
            "backend": "torch",
            "device": str(cfg.device),
            "optimization_mode": str(cfg.optimization_mode),
            "precision": str(cfg.model_precision).replace("torch.", ""),
            "img_size": str(cfg.img_size),
            "use_refiner": str(cfg.use_refiner),
            "mixed_precision": str(cfg.mixed_precision),
        }

    def run(self, frame: PreprocessedFrame) -> InferenceResult:
        from corridorkey_new.stages.inference.orchestrator import run_inference

        return run_inference(frame, self._model, self._config)


class MLXBackend:  # pragma: no cover
    """Apple Silicon MLX inference backend.

    Wraps ``CorridorKeyMLXEngine`` from the optional ``corridorkey-mlx``
    package. Converts the MLX uint8 output to the same ``InferenceResult``
    contract as ``TorchBackend`` so the rest of the pipeline is unaffected.

    This class is only instantiated by ``factory.load_backend()`` when the
    resolved backend is ``"mlx"``. It should never be imported directly on
    non-Apple-Silicon platforms.

    Args:
        mlx_engine: A ``CorridorKeyMLXEngine`` instance.
        img_size: The square resolution the engine was loaded with.
        tile_size: Tile size used for tiled inference, or None for full-frame.
    """

    def __init__(self, mlx_engine, img_size: int, tile_size: int | None) -> None:
        self._engine = mlx_engine
        self._img_size = img_size
        self._tile_size = tile_size

    @property
    def backend_name(self) -> str:
        return "mlx"

    @property
    def resolved_config(self) -> dict[str, str]:
        mode = f"tiled-{self._tile_size}" if self._tile_size else "full-frame"
        return {
            "backend": "mlx",
            "device": "apple-silicon",
            "optimization_mode": mode,
            "precision": "mlx-default",
            "img_size": str(self._img_size),
            "use_refiner": "true",
            "mixed_precision": "n/a",
        }

    def run(self, frame: PreprocessedFrame) -> InferenceResult:
        """Run MLX inference and return an ``InferenceResult``.

        Converts the preprocessed tensor back to a uint8 numpy array (MLX
        expects uint8), runs the engine, then wraps the uint8 output back
        into float32 tensors matching the Torch contract.
        """
        from corridorkey_new.stages.inference.contracts import InferenceResult

        # MLX engine expects uint8 numpy [H, W, C] — convert from [1, C, H, W] float tensor.
        tensor = frame.tensor  # [1, 4, H, W]
        rgb_np = tensor[0, :3].permute(1, 2, 0).cpu().float().numpy()  # [H, W, 3]
        mask_np = tensor[0, 3].cpu().float().numpy()  # [H, W]

        image_u8 = (np.clip(rgb_np, 0.0, 1.0) * 255).astype(np.uint8)
        mask_u8 = (np.clip(mask_np, 0.0, 1.0) * 255).astype(np.uint8)

        mlx_output = self._engine.process_frame(
            image_u8,
            mask_u8,
            refiner_scale=1.0,
            input_is_linear=False,
            fg_is_straight=True,
            despill_strength=0.0,  # postprocessor stage handles despill
            auto_despeckle=False,  # postprocessor stage handles despeckle
        )

        # MLX returns uint8 [H, W] alpha and [H, W, 3] fg — normalise to float32.
        alpha_np = mlx_output["alpha"].astype(np.float32) / 255.0
        fg_np = mlx_output["fg"].astype(np.float32) / 255.0

        if alpha_np.ndim == 2:
            alpha_np = alpha_np[:, :, np.newaxis]  # [H, W, 1]

        # Convert to [1, C, H, W] tensors to match TorchBackend output shape.
        alpha_t = torch.from_numpy(alpha_np).permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]
        fg_t = torch.from_numpy(fg_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        return InferenceResult(alpha=alpha_t, fg=fg_t, meta=frame.meta)
