"""Inference stage — model loader.

Responsible for constructing a GreenFormer instance, loading a checkpoint,
and handling the two edge cases that arise in practice:

  1. Compiled checkpoints — keys prefixed with ``_orig_mod.`` from
     ``torch.compile`` must be stripped before loading.

  2. Position embedding shape mismatch — when a checkpoint trained at one
     img_size is loaded for a different img_size, pos_embed tensors are
     bicubically interpolated to the new grid size.

Post-load fixups (matching core engine behaviour):
  - Refiner kept in float32 (GroupNorm does not support bf16 on CUDA).
  - All BatchNorm2d layers kept in float32 for the same reason.
  - torch.compile applied in speed mode on CUDA (disabled in lowvram mode
    because hooks + compile are incompatible with Dynamo tracing).
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import torch
from torch.nn import functional

from corridorkey_new.stages.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


def load_model(config: InferenceConfig) -> torch.nn.Module:
    """Load a GreenFormer model from a checkpoint.

    Constructs the model architecture, moves it to the configured device,
    loads the checkpoint, applies dtype fixups, and optionally compiles
    the model with torch.compile.

    Args:
        config: InferenceConfig with checkpoint_path, device, img_size,
            use_refiner, model_precision, and optimization_mode.

    Returns:
        GreenFormer in eval mode on config.device, ready for inference.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint cannot be loaded.
    """
    from corridorkey_new.stages.inference.model import GreenFormer

    if not config.checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint_path}")

    logger.info("Building GreenFormer (img_size=%d, refiner=%s)", config.img_size, config.use_refiner)
    model = GreenFormer(
        encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k",
        img_size=config.img_size,
        use_refiner=config.use_refiner,
    )
    model = model.to(config.device)
    model = model.to(config.model_precision)
    model.eval()

    # Enable TF32 on Ampere+ for faster matmuls with minimal precision loss.
    if config.model_precision != torch.float32 or config.mixed_precision:
        torch.set_float32_matmul_precision("high")

    logger.info("Loading checkpoint: %s", config.checkpoint_path)
    checkpoint = torch.load(config.checkpoint_path, map_location=config.device, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = _strip_compiled_prefix(state_dict)
    state_dict = _resize_pos_embeds(state_dict, model)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Checkpoint missing keys: %s", missing)
    if unexpected:
        logger.warning("Checkpoint unexpected keys: %s", unexpected)

    # GroupNorm (refiner) and BatchNorm2d (decoder head) do not support
    # bf16/fp16 on CUDA — keep them in float32 regardless of backbone dtype.
    if config.model_precision != torch.float32:
        if model.refiner is not None:
            model.refiner.float()
            logger.info("Refiner kept in float32 (GroupNorm bf16/fp16 unsupported on CUDA)")
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.float()
        logger.info("BatchNorm2d layers kept in float32")

    # torch.compile: speed mode on CUDA only.
    # Disabled in lowvram mode — hooks + compile are incompatible (Dynamo
    # sees the refiner module twice and raises "already tracked for mutation").
    device_type = torch.device(config.device).type
    use_compile = config.optimization_mode == "speed" and device_type == "cuda" and sys.platform in ("linux", "win32")
    if use_compile:
        try:
            cache_dir = Path.home() / ".cache" / "corridorkey" / "torch_compile"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
            compiled = torch.compile(model)
            logger.info("Warming up compiled model...")
            dummy = torch.zeros(1, 4, config.img_size, config.img_size, dtype=torch.float32, device=config.device)
            with torch.inference_mode():
                compiled(dummy)
            del dummy
            torch.cuda.empty_cache()
            logger.info("Warm-up complete.")
            return compiled  # type: ignore[return-value]
        except Exception as e:
            logger.warning("torch.compile failed (%s) — falling back to eager mode.", e)
            torch.cuda.empty_cache()

    logger.info("Model loaded successfully")
    return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_compiled_prefix(state_dict: dict) -> dict:
    """Remove the ``_orig_mod.`` prefix added by ``torch.compile``."""
    return {(k[10:] if k.startswith("_orig_mod.") else k): v for k, v in state_dict.items()}


def _resize_pos_embeds(state_dict: dict, model: torch.nn.Module) -> dict:
    """Bicubically interpolate position embeddings when grid sizes differ."""
    model_state = model.state_dict()
    new_state: dict = {}

    for k, v in state_dict.items():
        if "pos_embed" in k and k in model_state and v.shape != model_state[k].shape:
            src_seq = v.shape[1]
            dst_seq = model_state[k].shape[1]
            embed_dim = v.shape[2]

            src_grid = int(math.sqrt(src_seq))
            dst_grid = int(math.sqrt(dst_seq))

            logger.debug("Resizing pos_embed %s: grid %d -> %d", k, src_grid, dst_grid)

            spatial = v.permute(0, 2, 1).view(1, embed_dim, src_grid, src_grid)
            resized = functional.interpolate(spatial, size=(dst_grid, dst_grid), mode="bicubic", align_corners=False)
            v = resized.flatten(2).transpose(1, 2)

        new_state[k] = v

    return new_state
