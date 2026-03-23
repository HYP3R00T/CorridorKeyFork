"""Device detection and selection utilities for CorridorKey.

Call ``detect_gpu()`` to probe the system and ``resolve_device()`` to get
a validated device string for use with PyTorch.
"""

from __future__ import annotations

import logging
import platform

import torch
from pydantic import BaseModel, Field

from corridorkey.errors import DeviceError

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

    Accepts:
        - "auto" / None  — auto-detect via detect_gpu()
        - "cuda"         — CUDA device 0
        - "cuda:N"       — specific CUDA device index N (e.g. "cuda:1")
        - "rocm"         — AMD ROCm (maps to "cuda" in PyTorch)
        - "rocm:N"       — specific ROCm device index N
        - "mps"          — Apple Silicon
        - "cpu"          — CPU

    Returns:
        Validated PyTorch device string (e.g. "cuda", "cuda:1", "mps", "cpu").

    Raises:
        DeviceError: If the requested backend is unavailable or the device
            index is out of range.
    """
    if not requested or requested == "auto":
        gpu = detect_gpu()
        logger.info("Device auto-detected: %s", gpu)
        return gpu.device_str

    requested = requested.lower()

    # Handle cuda:N and rocm:N
    if requested.startswith(("cuda", "rocm")):
        if not torch.cuda.is_available():
            raise DeviceError(
                f"'{requested}' requested but torch.cuda.is_available() is False. "
                "Install a CUDA or ROCm-enabled PyTorch build."
            )
        # Parse optional index suffix
        if ":" in requested:
            try:
                idx = int(requested.split(":", 1)[1])
            except ValueError:
                raise DeviceError(
                    f"Invalid device index in '{requested}'. Expected 'cuda:N' where N is an integer."
                ) from None
            device_count = torch.cuda.device_count()
            if idx >= device_count:
                raise DeviceError(
                    f"Device index {idx} out of range — {device_count} CUDA device(s) available "
                    f"(valid indices: 0–{device_count - 1})."
                )
            return f"cuda:{idx}"
        return "cuda"

    if requested == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise DeviceError(
                "MPS requested but not available. Requires Apple Silicon (M1+) with macOS 12.3+ and PyTorch >= 1.12."
            )
        return "mps"

    if requested == "cpu":
        return "cpu"

    raise DeviceError(f"Unknown device '{requested}'. Valid options: auto, cuda, cuda:N, rocm, rocm:N, mps, cpu.")


def resolve_devices(requested: str | list[str] | None = None) -> list[str]:
    """Resolve one or more device strings for multi-GPU dispatch.

    Args:
        requested: A single device string, a list of device strings, or
            "all" to use every available CUDA device.

    Returns:
        List of validated PyTorch device strings. Always has at least one entry.

    Examples:
        resolve_devices("cuda")          → ["cuda"]
        resolve_devices("cuda:1")        → ["cuda:1"]
        resolve_devices(["cuda:0", "cuda:1"]) → ["cuda:0", "cuda:1"]
        resolve_devices("all")           → ["cuda:0", "cuda:1", ...] (all GPUs)
    """
    if requested == "all":
        if not torch.cuda.is_available():
            raise DeviceError("'all' requested but no CUDA devices are available.")
        count = torch.cuda.device_count()
        if count == 0:
            raise DeviceError("'all' requested but torch.cuda.device_count() returned 0.")
        devices = [f"cuda:{i}" for i in range(count)]
        logger.info("Device 'all' resolved to %d CUDA device(s): %s", count, devices)
        return devices

    if isinstance(requested, list):
        return [resolve_device(d) for d in requested]

    return [resolve_device(requested)]


def clear_device_cache(device: str) -> None:
    """Clear GPU memory cache for the given device. No-op for CPU.

    Args:
        device: PyTorch device string ("cuda", "mps", or "cpu").
    """
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
