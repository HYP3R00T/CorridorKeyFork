"""Inference stage — model loader.

Responsible for constructing a GreenFormer instance, loading a checkpoint,
and handling the two edge cases that arise in practice:

  1. Compiled checkpoints — keys prefixed with ``_orig_mod.`` from
     ``torch.compile`` must be stripped before loading.

  2. Position embedding shape mismatch — when a checkpoint trained at one
     img_size is loaded for a different img_size, pos_embed tensors are
     bicubically interpolated to the new grid size.
"""

from __future__ import annotations

import logging
import math

import torch
from torch.nn import functional

from corridorkey_new.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


def load_model(config: InferenceConfig) -> torch.nn.Module:
    """Load a GreenFormer model from a checkpoint.

    Constructs the model architecture, moves it to the configured device,
    loads the checkpoint, and returns the model in eval mode.

    Args:
        config: InferenceConfig with checkpoint_path, device, img_size,
            use_refiner, and model_precision.

    Returns:
        GreenFormer in eval mode on config.device.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint cannot be loaded.
    """
    # GreenFormer lives in corridorkey-core, which is the model architecture
    # package. This is the one intentional cross-package dependency in
    # corridorkey-new: we need the model class to instantiate it. Once
    # corridorkey-new is ready to replace corridorkey-core, GreenFormer will
    # be moved here and this import will be updated.
    from corridorkey_core.model_transformer import GreenFormer

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
