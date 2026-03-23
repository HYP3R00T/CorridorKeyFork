"""Inference stage — orchestrator.

Public entry point: ``run_inference(frame, model, config) -> InferenceResult``.

Owns:
  - fp16 autocast
  - tiled refiner (tiled refiner_mode)
  - VRAM probing for "auto" refiner_mode
  - converting raw model output to InferenceResult

Does NOT own:
  - model loading (loader.py)
  - despeckle / despill / compositing (postprocessor stage)
  - writing to disk (writer stage)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.nn import functional

from corridorkey.stages.inference.config import (
    _VRAM_FREE_CACHE_THRESHOLD_GB,
    _VRAM_TILED_THRESHOLD_GB,
    REFINER_TILE_OVERLAP,
    REFINER_TILE_SIZE,
    InferenceConfig,
)
from corridorkey.stages.inference.contracts import InferenceResult
from corridorkey.stages.preprocessor.contracts import PreprocessedFrame

logger = logging.getLogger(__name__)


def run_inference(
    frame: PreprocessedFrame,
    model: nn.Module,
    config: InferenceConfig,
    resolved_refiner_mode: str | None = None,
) -> InferenceResult:
    """Run model inference on a single preprocessed frame.

    Args:
        frame: Output of the preprocessing stage. tensor is [1, 4, H, W]
            on config.device, already ImageNet-normalised.
        model: Loaded GreenFormer in eval mode.
        config: Inference configuration.
        resolved_refiner_mode: Pre-resolved refiner mode ("full_frame" or
            "tiled"). When provided, skips the per-frame VRAM probe that
            ``_should_tile_refiner`` would otherwise perform for "auto" mode.
            ``TorchBackend.run()`` always passes this to avoid pynvml overhead
            on every frame.

    Returns:
        InferenceResult with alpha and fg tensors on device, plus FrameMeta.
    """
    tile_refiner = _should_tile_refiner(config, resolved_refiner_mode=resolved_refiner_mode)
    device_type = torch.device(config.device).type

    # Use the model's actual precision for autocast.
    # autocast is required whenever the model is not float32 — it handles both
    # the backbone (float32 input → half ops) and the float32 refiner/BatchNorm
    # boundaries (half activations → float32 ops). Without it, dtype mismatches
    # occur at those layer boundaries.
    autocast_dtype = config.model_precision
    autocast_enabled = device_type != "cpu" and config.model_precision != torch.float32

    hook_handle = None
    if tile_refiner:
        refiner = getattr(model, "refiner", None)
        if refiner is not None:
            hook_fn = _make_tiled_refiner_hook(refiner, config.refiner_scale)
            hook_handle = refiner.register_forward_hook(hook_fn)
    elif config.use_refiner and config.refiner_scale != 1.0:
        # Scale hook — only active when NOT tiling (tiled path handles scale internally).
        refiner = getattr(model, "refiner", None)
        if refiner is not None:
            scale = config.refiner_scale

            def _scale_hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
                return output * scale

            hook_handle = refiner.register_forward_hook(_scale_hook)

    try:
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type=device_type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ),
        ):
            output = model(frame.tensor)
    finally:
        # Always remove the hook — even if inference raises — so it can never
        # fire on a subsequent call or trigger recursion.
        if hook_handle is not None:
            hook_handle.remove()

    result = InferenceResult(
        alpha=output["alpha"],
        fg=output["fg"],
        meta=frame.meta,
    )

    _free_vram_if_needed(config.device)
    return result


def _free_vram_if_needed(device: str) -> None:
    """Release cached VRAM allocations on low-memory devices.

    On GPUs with <6 GB VRAM, PyTorch's caching allocator can hold onto
    freed blocks and cause OOM on the next frame. Calling empty_cache()
    after each frame keeps peak usage flat at the cost of a small sync.
    Only called when VRAM < 6 GB to avoid the sync overhead on larger GPUs.
    """
    dev = torch.device(device)
    if dev.type != "cuda":
        return
    try:
        vram_gb = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
        if vram_gb < _VRAM_FREE_CACHE_THRESHOLD_GB:
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Refiner mode resolution
# ---------------------------------------------------------------------------


def _should_tile_refiner(config: InferenceConfig, resolved_refiner_mode: str | None = None) -> bool:
    """Return True if the refiner should run in tiled mode.

    Args:
        config: Inference configuration.
        resolved_refiner_mode: Pre-resolved mode from a prior VRAM probe
            ("full_frame" or "tiled"). When provided, takes priority over
            everything else — avoids pynvml overhead on every frame when
            called from ``run_inference`` via ``TorchBackend.run()``.
    """
    if not config.use_refiner:
        return False

    # Pre-resolved mode takes full priority — caller already accounted for
    # device type (including MPS) when resolving.
    if resolved_refiner_mode is not None:
        return resolved_refiner_mode == "tiled"

    # MPS: Triton/inductor does not support Metal — always tile.
    if torch.device(config.device).type == "mps":
        return True

    mode = config.refiner_mode
    if mode == "tiled":
        return True
    if mode == "full_frame":
        return False

    # auto — probe VRAM (only reached when resolved_refiner_mode is None and
    # config.refiner_mode == "auto", i.e. called outside of TorchBackend.run())
    vram_gb = _probe_vram_gb(config.device)
    if vram_gb > 0 and vram_gb < _VRAM_TILED_THRESHOLD_GB:
        logger.info("Auto refiner_mode: %.1f GB VRAM → tiled refiner", vram_gb)
        return True
    logger.info("Auto refiner_mode: %.1f GB VRAM → full-frame refiner", vram_gb)
    return False


def _probe_vram_gb(device: str) -> float:
    """Return total VRAM in GB for the given device. Returns 0.0 on failure.

    Short-circuits immediately for non-CUDA devices (CPU, MPS) to avoid
    unnecessary pynvml import and init attempts. Correctly handles multi-GPU
    setups by parsing the device index from the device string (e.g. "cuda:1"
    → index 1). Falls back to index 0 for bare "cuda" strings.
    """
    dev = torch.device(device)
    if dev.type not in ("cuda", "rocm"):
        return 0.0
    # Parse the device index — torch.device("cuda:1").index == 1,
    # torch.device("cuda").index is None (defaults to 0).
    device_index = dev.index if dev.index is not None else 0
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem.total / (1024**3)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Tiled refiner hook
# ---------------------------------------------------------------------------


class _TiledRefinerState:
    """Mutable bypass flag shared between the hook and the tiled runner.

    Using an object (rather than a closure list) matches the core engine's
    ``self._bypass_tiled_refiner_hook`` pattern and avoids any closure
    scoping surprises.
    """

    __slots__ = ("bypass",)

    def __init__(self) -> None:
        self.bypass = False


def _make_tiled_refiner_hook(refiner: nn.Module, refiner_scale: float = 1.0):
    """Return a forward hook that replaces the refiner with a tiled pass.

    The ``state.bypass`` flag is set to True before each tile call inside
    ``_run_refiner_tiled`` and reset in a ``finally`` block, preventing the
    hook from firing recursively for tile-level refiner calls.

    Args:
        refiner: The CNNRefinerModule (used only for type clarity; the hook
            receives the module as its first argument at call time).
        refiner_scale: Multiplier applied to the tiled delta output before
            accumulation. 1.0 = full refinement, 0.0 = no refinement.
    """
    state = _TiledRefinerState()

    def hook(module: nn.Module, inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        if state.bypass:
            # Re-entrant call from inside _run_refiner_tiled — pass through.
            return output
        if len(inputs) != 2:
            raise RuntimeError(f"Tiled refiner hook expected 2 inputs (rgb, coarse_pred), got {len(inputs)}")
        rgb, coarse = inputs
        return _run_refiner_tiled(module, rgb, coarse, state, refiner_scale=refiner_scale)

    return hook


def _run_refiner_tiled(
    refiner: nn.Module,
    rgb: torch.Tensor,
    coarse: torch.Tensor,
    state: _TiledRefinerState,
    tile_size: int = REFINER_TILE_SIZE,
    overlap: int = REFINER_TILE_OVERLAP,
    refiner_scale: float = 1.0,
) -> torch.Tensor:
    """Run the CNN refiner in overlapping tiles to keep VRAM flat.

    Splits the full-resolution tensor into overlapping tiles, runs the
    refiner on each, and blends results back using linear ramp weights in
    the overlap regions. Identical output to full-frame inference.

    GroupNorm does not support bf16/fp16 on CUDA — the tiled pass always
    upcasts to float32 and casts the result back to the original dtype.

    Args:
        refiner: The CNNRefinerModule.
        rgb: [B, 3, H, W] float tensor.
        coarse: [B, 4, H, W] float tensor.
        state: Shared bypass flag — set to True around each tile call so the
            hook does not fire recursively for tile-level refiner invocations.
        tile_size: Spatial size of each tile (square).
        overlap: Overlap in pixels between adjacent tiles.
        refiner_scale: Multiplier applied to each tile's delta output before
            accumulation. Mirrors the full-frame ``refiner_scale`` behaviour.

    Returns:
        [B, 4, H, W] delta logits tensor, same dtype as coarse.
    """
    b, _, h, w = rgb.shape
    orig_dtype = rgb.dtype

    # GroupNorm does not support bf16/fp16 — upcast input tensors for the
    # tiled pass. Do NOT call refiner.float() — that would permanently mutate
    # the module's weights in-place. The refiner is already kept in float32 by
    # loader.py, so its parameters are always float32; only the activations
    # (inputs) need upcasting here.
    if orig_dtype != torch.float32:
        rgb = rgb.float()
        coarse = coarse.float()

    output = torch.zeros(b, 4, h, w, device=rgb.device, dtype=torch.float32)
    weight = torch.zeros(b, 1, h, w, device=rgb.device, dtype=torch.float32)

    stride = tile_size - overlap

    # Linear ramp blend window — starts at a small positive value so image
    # border pixels always accumulate non-zero weight (ramp starting at 0.0
    # would leave the outermost pixel row/column with weight=0 → output=0).
    safe_overlap = min(overlap, tile_size // 2 - 1)
    flat_len = tile_size - 2 * safe_overlap
    ramp = (
        torch.linspace(1.0 / (safe_overlap + 1), 1.0, safe_overlap, device=rgb.device)
        if safe_overlap > 0
        else torch.empty(0, device=rgb.device)
    )
    flat = torch.ones(flat_len, device=rgb.device)
    blend_1d = torch.cat([ramp, flat, ramp.flip(0)]) if safe_overlap > 0 else flat
    blend_2d = (blend_1d.unsqueeze(0) * blend_1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

    y = 0
    while y < h:
        y_end = min(y + tile_size, h)
        y_start = max(y_end - tile_size, 0)
        x = 0
        while x < w:
            x_end = min(x + tile_size, w)
            x_start = max(x_end - tile_size, 0)

            rgb_tile = rgb[:, :, y_start:y_end, x_start:x_end]
            coarse_tile = coarse[:, :, y_start:y_end, x_start:x_end]

            # Pad to tile_size if this is an edge tile smaller than tile_size.
            th, tw = rgb_tile.shape[2], rgb_tile.shape[3]
            pad_h, pad_w = tile_size - th, tile_size - tw
            if pad_h > 0 or pad_w > 0:
                rgb_tile = functional.pad(rgb_tile, (0, pad_w, 0, pad_h))
                coarse_tile = functional.pad(coarse_tile, (0, pad_w, 0, pad_h))

            with torch.inference_mode():
                state.bypass = True
                try:
                    delta_tile = refiner(rgb_tile, coarse_tile)
                finally:
                    state.bypass = False

            # Crop back to actual tile size before accumulating.
            delta_tile = delta_tile[:, :, :th, :tw]
            w_tile = blend_2d[:, :, :th, :tw]

            output[:, :, y_start:y_end, x_start:x_end] += delta_tile * w_tile * refiner_scale
            weight[:, :, y_start:y_end, x_start:x_end] += w_tile

            x += stride
            if x_end == w:
                break
        y += stride
        if y_end == h:
            break

    result = output / weight.clamp(min=1e-6)
    return result.to(orig_dtype)
