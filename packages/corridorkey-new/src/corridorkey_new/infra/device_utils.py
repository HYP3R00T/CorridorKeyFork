"""Device detection and selection utilities for CorridorKey.

Call ``detect_gpu()`` to probe the system and ``resolve_device()`` to get
a validated device string for use with PyTorch.
"""

from __future__ import annotations

import logging
import platform

import torch
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GPUInfo(BaseModel):
    """Information about the available GPU backend.

    Attributes:
        vendor: GPU vendor name (e.g. "NVIDIA", "AMD", "Apple", "CPU").
        backend: PyTorch backend in use ("CUDA", "ROCm", "MPS", "CPU").
        version: Backend version string (CUDA or ROCm version). None for MPS/CPU.
        devices: List of device names detected.
        vram_gb: VRAM in GB per device. Empty for MPS and CPU.
    """

    vendor: str
    backend: str
    version: str | None = None
    devices: list[str] = Field(default_factory=list)
    vram_gb: list[float] = Field(default_factory=list)

    @property
    def device_str(self) -> str:
        """PyTorch device string to pass to torch.device()."""
        return self.backend.lower() if self.backend != "ROCm" else "cuda"


def detect_gpu() -> GPUInfo:
    """Probe the system for available GPU hardware.

    Detection order: ROCm (AMD) > CUDA (NVIDIA) > MPS (Apple) > CPU.

    Returns:
        GPUInfo describing the best available backend.
    """
    # AMD ROCm — torch.version.hip is set only in ROCm builds
    if torch.version.hip is not None:
        device_count = torch.cuda.device_count()
        return GPUInfo(
            vendor="AMD",
            backend="ROCm",
            version=torch.version.hip,
            devices=[torch.cuda.get_device_name(i) for i in range(device_count)],
            vram_gb=[torch.cuda.get_device_properties(i).total_memory / 1024**3 for i in range(device_count)],
        )

    # NVIDIA CUDA
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        return GPUInfo(
            vendor="NVIDIA",
            backend="CUDA",
            version=torch.version.cuda,
            devices=[torch.cuda.get_device_name(i) for i in range(device_count)],
            vram_gb=[torch.cuda.get_device_properties(i).total_memory / 1024**3 for i in range(device_count)],
        )

    # Apple Silicon MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        chip = platform.processor() or platform.machine() or "Apple Silicon"
        return GPUInfo(
            vendor="Apple",
            backend="MPS",
            version=None,
            devices=[chip],
            vram_gb=[],  # unified memory - not separately reported
        )

    # CPU fallback
    return GPUInfo(
        vendor="CPU",
        backend="CPU",
        version=None,
        devices=[platform.processor() or platform.machine() or "Unknown CPU"],
        vram_gb=[],
    )


def resolve_device(requested: str | None = None) -> str:
    """Resolve and validate a device string.

    Args:
        requested: One of "auto", "cuda", "rocm", "mps", "cpu", or None.
            None and "auto" trigger auto-detection via detect_gpu().

    Returns:
        Validated PyTorch device string ("cuda", "mps", or "cpu").

    Raises:
        RuntimeError: If the requested backend is unavailable.
    """
    if not requested or requested == "auto":
        gpu = detect_gpu()
        logger.info("Device auto-detected: %s", gpu)
        return gpu.device_str

    requested = requested.lower()

    if requested in ("cuda", "rocm"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"'{requested}' requested but torch.cuda.is_available() is False. "
                "Install a CUDA or ROCm-enabled PyTorch build."
            )
        return "cuda"

    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS requested but not available. Requires Apple Silicon (M1+) with macOS 12.3+ and PyTorch >= 1.12."
            )
        return "mps"

    if requested == "cpu":
        return "cpu"

    raise RuntimeError(f"Unknown device '{requested}'. Valid options: auto, cuda, rocm, mps, cpu.")


def clear_device_cache(device: str) -> None:
    """Clear GPU memory cache for the given device. No-op for CPU.

    Args:
        device: PyTorch device string ("cuda", "mps", or "cpu").
    """
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
