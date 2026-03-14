"""Centralized cross-platform device selection for CorridorKey."""

import logging

import torch

logger = logging.getLogger(__name__)

# Accepted device strings (including the sentinel "auto" for auto-detection).
VALID_DEVICES = ("auto", "cuda", "mps", "cpu")


def detect_best_device() -> str:
    """Auto-detect best available device: CUDA > MPS > CPU.

    Returns:
        Device string: "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Auto-selected device: %s", device)
    return device


def resolve_device(requested: str | None = None) -> str:
    """Resolve and validate a device string.

    Args:
        requested: Explicit device string ("cuda", "mps", "cpu", or "auto").
            None or "auto" triggers auto-detection. The config system
            (``CorridorKeyConfig.device``) is the preferred source for this
            value - callers should read from config rather than relying on
            environment variables directly.

    Returns:
        Validated device string ("cuda", "mps", or "cpu").

    Raises:
        RuntimeError: If the requested backend is unavailable.
    """
    device = requested or "auto"

    if device == "auto":
        return detect_best_device()

    device = device.lower()
    if device not in VALID_DEVICES:
        raise RuntimeError(f"Unknown device '{device}'. Valid options: {', '.join(VALID_DEVICES)}")

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False. Install a CUDA-enabled PyTorch build."
            )
    elif device == "mps":
        if not hasattr(torch.backends, "mps"):
            raise RuntimeError(
                "MPS requested but this PyTorch build has no MPS support. Install PyTorch >= 1.12 with MPS backend."
            )
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS requested but not available on this machine. Requires Apple Silicon (M1+) with macOS 12.3+."
            )

    return device


def clear_device_cache(device: torch.device | str) -> None:
    """Clear GPU memory cache if applicable (no-op for CPU).

    Args:
        device: Target device. Accepts a ``torch.device`` or a plain string
            such as ``"cuda"`` or ``"mps"``.
    """
    device_type = device.type if isinstance(device, torch.device) else device
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()
