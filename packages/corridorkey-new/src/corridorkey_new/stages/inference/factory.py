"""Inference stage — backend factory.

Single entry point for constructing an inference backend. Callers receive a
``ModelBackend`` instance that always satisfies the same protocol regardless
of whether PyTorch or MLX is running underneath.

Backend resolution order:
    ``InferenceConfig.backend`` field > ``CORRIDORKEY_BACKEND`` env var > auto-detect

Auto-detect:
    Apple Silicon + ``corridorkey_mlx`` importable + ``.safetensors`` present → mlx
    Otherwise → torch

Usage::

    config = load_config().to_inference_config(device="cuda")
    backend = load_backend(config)
    result = backend.run(preprocessed_frame)
    print(backend.resolved_config)
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import sys
from pathlib import Path

from corridorkey_new.stages.inference.config import InferenceConfig

logger = logging.getLogger(__name__)

_BACKEND_ENV_VAR = "CORRIDORKEY_BACKEND"
_TORCH_EXT = ".pth"
_MLX_EXT = ".safetensors"
_DEFAULT_MLX_TILE_SIZE = 512
_DEFAULT_MLX_TILE_OVERLAP = 64


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_backend(config: InferenceConfig):  # -> ModelBackend
    """Construct and return the appropriate inference backend.

    Resolves the backend (torch or mlx), resolves "auto" refiner_mode to a
    concrete value, loads the model/engine, and returns a ``ModelBackend``
    instance ready to call ``.run(frame)``.

    Args:
        config: ``InferenceConfig`` with all fields populated. The
            ``backend`` field controls which backend is selected.

    Returns:
        A ``ModelBackend`` instance (``TorchBackend`` or ``MLXBackend``).

    Raises:
        RuntimeError: If the requested backend is unavailable.
        FileNotFoundError: If no suitable checkpoint is found.
    """
    from corridorkey_new.stages.inference.backend import TorchBackend

    backend_name = _resolve_backend(config.backend)

    if backend_name == "mlx":  # pragma: no cover
        return _load_mlx_backend(config)

    # Resolve "auto" refiner_mode to a concrete value before loading the model.
    # This ensures torch.compile decisions are made on the real mode, not "auto".
    from corridorkey_new.stages.inference.orchestrator import _should_tile_refiner

    resolved_refiner_mode = "tiled" if _should_tile_refiner(config) else "full_frame"
    logger.info(
        "Refiner mode resolved: %s → %s (device=%s)",
        config.refiner_mode,
        resolved_refiner_mode,
        config.device,
    )

    from corridorkey_new.stages.inference.loader import load_model

    model = load_model(config, resolved_refiner_mode=resolved_refiner_mode)
    logger.info("TorchBackend ready (device=%s, precision=%s)", config.device, config.model_precision)
    return TorchBackend(config=config, model=model, resolved_refiner_mode=resolved_refiner_mode)


def discover_checkpoint(checkpoint_dir: str | Path, backend: str = "torch") -> Path:
    """Find exactly one checkpoint file for the given backend in ``checkpoint_dir``.

    Args:
        checkpoint_dir: Directory to search.
        backend: ``"torch"`` (looks for ``.pth``) or ``"mlx"`` (looks for ``.safetensors``).

    Returns:
        Path to the single matching checkpoint file.

    Raises:
        FileNotFoundError: If no matching file is found.
        ValueError: If more than one matching file is found.
    """
    ext = _MLX_EXT if backend == "mlx" else _TORCH_EXT
    other_ext = _TORCH_EXT if backend == "mlx" else _MLX_EXT
    checkpoint_dir = Path(checkpoint_dir)
    matches = list(checkpoint_dir.glob(f"*{ext}"))

    if len(matches) == 0:
        other_files = list(checkpoint_dir.glob(f"*{other_ext}"))
        hint = ""
        if other_files:
            other_backend = "mlx" if other_ext == _MLX_EXT else "torch"
            hint = f" (Found {other_ext} files — did you mean backend='{other_backend}'?)"
        raise FileNotFoundError(f"No {ext} checkpoint found in {checkpoint_dir}.{hint}")

    if len(matches) > 1:
        names = [f.name for f in matches]
        raise ValueError(f"Multiple {ext} checkpoints in {checkpoint_dir}: {names}. Keep exactly one.")

    return matches[0]


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------


def _resolve_backend(requested: str) -> str:
    """Resolve the backend string to ``"torch"`` or ``"mlx"``.

    Priority: ``config.backend`` field > ``CORRIDORKEY_BACKEND`` env var > auto-detect.
    """
    if requested == "auto":
        env = os.environ.get(_BACKEND_ENV_VAR, "auto").lower()
        if env != "auto":
            requested = env

    if requested == "auto":
        return _auto_detect()

    if requested not in ("torch", "mlx", "auto"):
        raise RuntimeError(f"Unknown backend '{requested}'. Valid: torch, mlx, auto")

    if requested == "mlx":
        _assert_mlx_available()

    return requested


def _auto_detect() -> str:
    """Try MLX on Apple Silicon, fall back to torch."""
    if sys.platform != "darwin" or platform.machine() != "arm64":
        logger.info("Not Apple Silicon — using torch backend")
        return "torch"
    if not _mlx_importable():
        logger.info("corridorkey_mlx not installed — using torch backend")
        return "torch"
    logger.info("Apple Silicon + MLX available — using mlx backend")
    return "mlx"


def _mlx_importable() -> bool:
    return importlib.util.find_spec("corridorkey_mlx") is not None


def _assert_mlx_available() -> None:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("MLX backend requires Apple Silicon (M1+ Mac)")
    if not _mlx_importable():
        raise RuntimeError(
            "MLX backend requested but corridorkey_mlx is not installed. "
            "Install with: uv pip install 'corridorkey-new[mlx]'"
        )


# ---------------------------------------------------------------------------
# Backend loaders
# ---------------------------------------------------------------------------


def _load_mlx_backend(config: InferenceConfig):  # pragma: no cover
    """Load the MLX engine and wrap it in MLXBackend."""
    from corridorkey_new.stages.inference.backend import MLXBackend

    ckpt = discover_checkpoint(config.checkpoint_path.parent, backend="mlx")
    from corridorkey_mlx import CorridorKeyMLXEngine  # type: ignore[import-not-found]

    engine = CorridorKeyMLXEngine(
        str(ckpt),
        img_size=config.img_size,
        tile_size=_DEFAULT_MLX_TILE_SIZE,
        overlap=_DEFAULT_MLX_TILE_OVERLAP,
    )
    logger.info("MLXBackend ready: %s (img_size=%d)", ckpt.name, config.img_size)
    return MLXBackend(
        mlx_engine=engine,
        img_size=config.img_size,
        tile_size=_DEFAULT_MLX_TILE_SIZE,
        overlap=_DEFAULT_MLX_TILE_OVERLAP,
        refiner_scale=config.refiner_scale,
    )
