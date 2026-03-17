"""Runtime inference engine for the CorridorKey chroma keying model.

Wraps GreenFormer with checkpoint loading, optional torch.compile, and a
full per-frame processing pipeline including preprocessing, inference,
despilling, matte cleanup, and compositing.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional

from corridorkey_core.compositing import (
    clean_matte,
    composite_premul,
    composite_straight,
    create_checkerboard,
    despill,
    linear_to_srgb,
    premultiply,
    srgb_to_linear,
)
from corridorkey_core.model_transformer import GreenFormer

logger = logging.getLogger(__name__)

# VRAM threshold for optimization profile auto-selection.
# Below this: tiled refiner. At or above: full-frame.
_VRAM_TILE_THRESHOLD_GB = 12.0

# Refiner tile size and overlap for lowvram mode.
_REFINER_TILE_SIZE = 512
_REFINER_TILE_OVERLAP = 128

VALID_OPT_MODES = ("auto", "speed", "lowvram")
OPT_MODE_ENV_VAR = "CORRIDORKEY_OPT_MODE"


def _probe_vram_gb() -> float:
    """Return total GPU VRAM in GB using pynvml (driver-level, no CUDA context).

    Falls back to torch.cuda if pynvml is unavailable. Returns 0.0 if both fail.
    Using pynvml avoids stalling when called immediately after another model
    (e.g. GVM) has released the CUDA context.
    """
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return mem.total / (1024**3)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        pass
    return 0.0


def _apply_source_passthrough(
    source_srgb: np.ndarray,
    fg_pred: np.ndarray,
    alpha_pred: np.ndarray,
    edge_erode_px: int,
    edge_blur_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend original source pixels into opaque interior regions of the fg prediction.

    Interior pixels (where alpha ≈ 1.0) are taken from the source frame untouched.
    The model's fg prediction is only used in the edge transition band where it
    handles green-screen separation, hair strands, and semi-transparency.

    Args:
        source_srgb: Original source frame [H, W, 3] sRGB float32.
        fg_pred: Model fg prediction [H, W, 3] sRGB float32.
        alpha_pred: Predicted alpha [H, W, 1] float32 in [0, 1].
        edge_erode_px: Pixels to erode the interior mask inward.
        edge_blur_px: Gaussian blur radius for the transition seam.

    Returns:
        Tuple of (blended_fg [H, W, 3], rebuilt_processed_rgba [H, W, 4]).
    """
    alpha_2d = alpha_pred[:, :, 0]
    kernel_size = max(1, edge_erode_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    interior = cv2.erode(alpha_2d, kernel, iterations=1)
    blur_k = max(1, edge_blur_px * 2 + 1)
    blend_mask = cv2.GaussianBlur(interior, (blur_k, blur_k), 0)[:, :, np.newaxis]
    blended_fg = np.asarray(blend_mask * source_srgb + (1.0 - blend_mask) * fg_pred, dtype=np.float32)
    fg_lin = np.asarray(srgb_to_linear(blended_fg), dtype=np.float32)
    fg_premul = np.asarray(premultiply(fg_lin, alpha_pred), dtype=np.float32)
    processed_rgba = np.concatenate([fg_premul, alpha_pred], axis=-1)
    return blended_fg, processed_rgba


class CorridorKeyEngine:  # pragma: no cover
    """Inference engine for the CorridorKey chroma keying model.

    Loads a GreenFormer checkpoint, optionally compiles it with torch.compile,
    and exposes process_frame for per-frame alpha matte prediction.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        img_size: int = 2048,
        use_refiner: bool = True,
        mixed_precision: bool = True,
        model_precision: torch.dtype = torch.float32,
        optimization_mode: str = "auto",
    ) -> None:
        """Initialize the engine and load the model from a checkpoint.

        Args:
            checkpoint_path: Path to the .pt or .pth checkpoint file.
            device: Torch device string, e.g. "cpu", "cuda", "cuda:0".
            img_size: Square resolution the model runs at internally.
            use_refiner: Whether to enable the CNN refiner module.
            mixed_precision: Whether to run inference in fp16 autocast.
            model_precision: Weight dtype for the model (float32 or float16).
            optimization_mode: Refiner execution strategy.
                "auto"    — probe VRAM via pynvml; < 12 GB → lowvram, else → speed.
                "speed"   — full 2048×2048 refiner pass + torch.compile.
                "lowvram" — tiled refiner (512×512, 128px overlap) + torch.compile.
                MPS always forces "lowvram" (Triton does not support Metal).
                Override via CORRIDORKEY_OPT_MODE environment variable.
        """
        self.device = torch.device(device)
        self.img_size = img_size
        self.checkpoint_path = Path(checkpoint_path)
        self.use_refiner = use_refiner

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

        self._checkerboard_cache: dict[tuple[int, int], np.ndarray] = {}

        if mixed_precision or model_precision != torch.float32:
            torch.set_float32_matmul_precision("high")

        self.mixed_precision = mixed_precision
        if mixed_precision and model_precision == torch.float16:
            self.mixed_precision = False

        self.model_precision = model_precision

        # Resolve optimization mode — env var overrides argument.
        env_mode = os.environ.get(OPT_MODE_ENV_VAR, "").lower()
        if env_mode in VALID_OPT_MODES:
            optimization_mode = env_mode
            logger.info("Optimization mode override from env: %s", env_mode)

        # MPS (Apple Silicon via PyTorch) always forces lowvram —
        # Triton/inductor does not support Metal.
        is_mps = self.device.type == "mps"
        if is_mps:
            optimization_mode = "lowvram"
            logger.info("MPS device detected — forcing lowvram mode (no torch.compile on Metal)")

        if optimization_mode == "speed":
            self._tile_refiner = False
            self._use_compile = True
            logger.info("Optimization: speed (full-frame refiner, torch.compile)")
        elif optimization_mode == "lowvram":
            self._tile_refiner = True
            self._use_compile = not is_mps
            logger.info(
                "Optimization: lowvram (tiled refiner %dx%d, %dpx overlap%s)",
                _REFINER_TILE_SIZE,
                _REFINER_TILE_SIZE,
                _REFINER_TILE_OVERLAP,
                ", no torch.compile" if is_mps else "",
            )
        else:  # auto
            vram_gb = _probe_vram_gb()
            if 0 < vram_gb < _VRAM_TILE_THRESHOLD_GB:
                self._tile_refiner = True
                self._use_compile = True
                logger.info(
                    "Optimization: auto → lowvram (%.1f GB < %.0f GB threshold)", vram_gb, _VRAM_TILE_THRESHOLD_GB
                )
            else:
                self._tile_refiner = False
                self._use_compile = True
                logger.info(
                    "Optimization: auto → speed (%.1f GB ≥ %.0f GB threshold)", vram_gb, _VRAM_TILE_THRESHOLD_GB
                )

        model = self._load_model().to(model_precision)

        # torch.compile gives a large speedup on CUDA. Skip on unsupported platforms
        # and when explicitly disabled (MPS lowvram path).
        if self._use_compile and sys.platform in ("linux", "win32"):
            try:
                cache_dir = Path.home() / ".cache" / "corridorkey" / "torch_compile"
                cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
                self.model = torch.compile(model)
                logger.info("Warming up compiled model...")
                dummy = torch.zeros(1, 4, img_size, img_size, dtype=model_precision, device=self.device)
                with torch.inference_mode():
                    self.model(dummy)
                del dummy
                logger.info("Warm-up complete.")
            except Exception as e:
                logger.warning("torch.compile failed (%s) — falling back to eager mode.", e)
                torch.cuda.empty_cache()
                self.model = model
        else:
            self.model = model

    def _load_model(self) -> GreenFormer:
        """Load and return a GreenFormer model from the configured checkpoint path.

        Handles the _orig_mod. prefix left by torch.compile and resizes positional
        embeddings when the checkpoint resolution differs from img_size.
        """
        logger.info("Loading CorridorKey from %s", self.checkpoint_path)
        model = GreenFormer(
            encoder_name="hiera_base_plus_224.mae_in1k_ft_in1k", img_size=self.img_size, use_refiner=self.use_refiner
        )
        model = model.to(self.device)
        model.eval()

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Strip the _orig_mod. prefix that torch.compile adds to state dict keys.
        # Also resize positional embeddings when the checkpoint was trained at a different resolution.
        new_state_dict = {}
        model_state = model.state_dict()

        for k, v in state_dict.items():
            if k.startswith("_orig_mod."):
                k = k[10:]

            if "pos_embed" in k and k in model_state and v.shape != model_state[k].shape:
                logger.debug("Resizing %s from %s to %s", k, v.shape, model_state[k].shape)
                # Treat the sequence dimension as a square spatial grid and bicubic-interpolate.
                seq_len_src = v.shape[1]
                seq_len_dst = model_state[k].shape[1]
                embed_dim = v.shape[2]

                grid_size_src = int(math.sqrt(seq_len_src))
                grid_size_dst = int(math.sqrt(seq_len_dst))

                pos_embed_spatial = v.permute(0, 2, 1).view(1, embed_dim, grid_size_src, grid_size_src)
                pos_embed_resized = functional.interpolate(
                    pos_embed_spatial, size=(grid_size_dst, grid_size_dst), mode="bicubic", align_corners=False
                )
                v = pos_embed_resized.flatten(2).transpose(1, 2)

            new_state_dict[k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        if len(missing) > 0:
            logger.warning("Missing keys in checkpoint: %s", missing)
        if len(unexpected) > 0:
            logger.warning("Unexpected keys in checkpoint: %s", unexpected)

        return model

    def _run_refiner_tiled(
        self,
        rgb_input: torch.Tensor,
        coarse_pred: torch.Tensor,
        tile_size: int = _REFINER_TILE_SIZE,
        overlap: int = _REFINER_TILE_OVERLAP,
    ) -> torch.Tensor:
        """Run the CNN refiner in tiles to keep VRAM flat on low-memory GPUs.

        Splits the full-resolution tensor into overlapping tiles, runs the
        refiner on each, and blends results back using cosine weights in the
        overlap regions. Identical output to full-frame inference — the tiling
        is invisible to the caller.

        Args:
            rgb_input: RGB tensor [B, 3, H, W].
            coarse_pred: Coarse alpha+fg tensor [B, 4, H, W].
            tile_size: Spatial size of each tile (square).
            overlap: Overlap in pixels between adjacent tiles.

        Returns:
            Delta logits tensor [B, 4, H, W].
        """
        b, _, h, w = rgb_input.shape
        refiner = self.model.refiner  # ty:ignore[unresolved-attribute]

        output = torch.zeros(b, 4, h, w, device=rgb_input.device, dtype=rgb_input.dtype)
        weight = torch.zeros(b, 1, h, w, device=rgb_input.device, dtype=rgb_input.dtype)

        stride = tile_size - overlap

        # Build 1D cosine blend weights for one tile.
        ramp = torch.linspace(0.0, 1.0, overlap, device=rgb_input.device)
        flat = torch.ones(tile_size - 2 * overlap, device=rgb_input.device)
        blend_1d = torch.cat([ramp, flat, ramp.flip(0)])  # length = tile_size
        blend_2d = (blend_1d.unsqueeze(0) * blend_1d.unsqueeze(1)).unsqueeze(0).unsqueeze(0)  # [1,1,T,T]

        y = 0
        while y < h:
            y_end = min(y + tile_size, h)
            y_start = max(y_end - tile_size, 0)
            x = 0
            while x < w:
                x_end = min(x + tile_size, w)
                x_start = max(x_end - tile_size, 0)

                rgb_tile = rgb_input[:, :, y_start:y_end, x_start:x_end]
                coarse_tile = coarse_pred[:, :, y_start:y_end, x_start:x_end]

                # Pad to tile_size if the tile is smaller (edge tiles).
                th, tw = rgb_tile.shape[2], rgb_tile.shape[3]
                pad_h, pad_w = tile_size - th, tile_size - tw
                if pad_h > 0 or pad_w > 0:
                    rgb_tile = functional.pad(rgb_tile, (0, pad_w, 0, pad_h))
                    coarse_tile = functional.pad(coarse_tile, (0, pad_w, 0, pad_h))

                delta_tile = refiner(rgb_tile, coarse_tile)

                # Crop back to actual tile size before accumulating.
                delta_tile = delta_tile[:, :, :th, :tw]
                w_tile = blend_2d[:, :, :th, :tw]

                output[:, :, y_start:y_end, x_start:x_end] += delta_tile * w_tile
                weight[:, :, y_start:y_end, x_start:x_end] += w_tile

                x += stride
                if x_end == w:
                    break
            y += stride
            if y_end == h:
                break

        # Normalise by accumulated blend weights.
        return output / weight.clamp(min=1e-6)

    @torch.inference_mode()
    def process_frame(
        self,
        image: np.ndarray,
        mask_linear: np.ndarray,
        refiner_scale: float = 1.0,
        input_is_linear: bool = False,
        fg_is_straight: bool = True,
        despill_strength: float = 1.0,
        auto_despeckle: bool = True,
        despeckle_size: int = 400,
        source_passthrough: bool = False,
        edge_erode_px: int = 3,
        edge_blur_px: int = 7,
    ) -> dict[str, np.ndarray]:
        """Run the full keying pipeline on a single frame.

        Args:
            image: RGB float array [H, W, 3] in range 0.0-1.0 or uint8 0-255.
                Assumed sRGB unless input_is_linear is True.
            mask_linear: Grayscale float array [H, W] or [H, W, 1] in range 0.0-1.0.
                Always assumed to be linear.
            refiner_scale: Multiplier applied to the CNN refiner output deltas.
                Values above 1.0 strengthen refinement, below 1.0 weaken it.
            input_is_linear: If True, the image is treated as linear light and
                converted to sRGB before being passed to the model.
            fg_is_straight: If True, the foreground output is treated as straight
                (unpremultiplied) during compositing.
            despill_strength: Blend factor for the despill effect (0.0 to 1.0).
            auto_despeckle: If True, removes small disconnected foreground islands
                from the predicted alpha matte.
            despeckle_size: Minimum pixel area for a foreground island to be kept
                when auto_despeckle is enabled.
            source_passthrough: If True, passes original source pixels through in
                opaque interior regions. Only the edge transition band uses the
                model's fg prediction. Preserves full source quality in solid areas.
            edge_erode_px: Pixels to erode the interior mask inward before blending.
                Safety buffer to avoid using original pixels where green spill
                might contaminate.
            edge_blur_px: Gaussian blur radius for the transition blend between
                source and model fg. Controls smoothness of the seam.

        Returns:
            A dict with four keys:
                "alpha": Raw predicted alpha [H, W, 1], linear float.
                "fg": Raw predicted foreground [H, W, 3], sRGB straight float.
                "comp": Preview composite over checkerboard [H, W, 3], sRGB float.
                "processed": Final RGBA [H, W, 4], linear premultiplied float.
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0

        if mask_linear.dtype == np.uint8:
            mask_linear = mask_linear.astype(np.float32) / 255.0

        h, w = image.shape[:2]

        if mask_linear.ndim == 2:
            mask_linear = mask_linear[:, :, np.newaxis]

        # Resize to the model's internal resolution.
        # When the input is linear, resize before converting to sRGB to preserve
        # energy in highlights during downsampling.
        if input_is_linear:
            image_resized_linear = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            image_resized_srgb = np.asarray(linear_to_srgb(image_resized_linear), dtype=np.float32)
        else:
            image_resized_srgb = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        mask_resized = cv2.resize(mask_linear, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        if mask_resized.ndim == 2:
            mask_resized = mask_resized[:, :, np.newaxis]

        # Normalize with ImageNet mean and std, which the Hiera encoder was pretrained with.
        image_normalized = (image_resized_srgb - self.mean) / self.std

        # Stack image and mask into a single [B, 4, H, W] tensor.
        model_input_np = np.concatenate([image_normalized, mask_resized], axis=-1)
        model_input = (
            torch.from_numpy(model_input_np.transpose((2, 0, 1))).unsqueeze(0).to(self.model_precision).to(self.device)
        )

        # Optionally scale the refiner's delta output via a forward hook.
        # In lowvram mode, replace the refiner's forward entirely with the tiled version.
        hook_handle = None
        if self._tile_refiner and self.model.refiner is not None:  # ty:ignore[unresolved-attribute]
            # Capture rgb_input in closure for the tiled call.
            _rgb = model_input[:, :3]
            _coarse_ref: list[torch.Tensor] = []

            def _tiled_refiner_hook(module, inputs, output):
                # inputs[0] is the concatenated [img, coarse_pred] tensor.
                # We ignore it and re-run tiled using the captured rgb + coarse.
                coarse = _coarse_ref[0] if _coarse_ref else inputs[0][:, 3:]
                return self._run_refiner_tiled(_rgb, coarse)

            # Capture coarse_pred before the refiner runs via a pre-hook.
            def _capture_coarse_hook(module, inputs):
                # inputs[0] is [img, coarse_pred] concatenated — coarse is channels 3:7.
                _coarse_ref.clear()
                _coarse_ref.append(inputs[0][:, 3:])

            pre_handle = self.model.refiner.register_forward_pre_hook(_capture_coarse_hook)  # ty:ignore[unresolved-attribute]
            hook_handle = self.model.refiner.register_forward_hook(_tiled_refiner_hook)  # ty:ignore[unresolved-attribute]
        elif refiner_scale != 1.0 and self.model.refiner is not None:  # ty:ignore[unresolved-attribute]

            def scale_hook(module, input, output):
                return output * refiner_scale

            hook_handle = self.model.refiner.register_forward_hook(scale_hook)  # ty:ignore[unresolved-attribute]
            pre_handle = None
        else:
            pre_handle = None

        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.mixed_precision):
            model_output = self.model(model_input)

        if hook_handle:
            hook_handle.remove()
        if pre_handle:
            pre_handle.remove()

        # Transfer at model resolution (2048x2048) - small tensors, fast DMA.
        # All post-processing runs at this resolution (~4x fewer pixels than 4K).
        # Only the final outputs are upsampled to original resolution.
        alpha_s = model_output["alpha"][0].permute(1, 2, 0).float().cpu().numpy()  # [S, S, 1]
        fg_s = model_output["fg"][0].permute(1, 2, 0).float().cpu().numpy()  # [S, S, 3]

        # Despeckle at model resolution (cv2 connected-components, no GPU equivalent).
        if auto_despeckle:
            alpha_s = clean_matte(alpha_s, area_threshold=despeckle_size, dilation=25, blur_size=5)

        # Despill and linearise at model resolution - color-only ops, resolution-independent.
        fg_despilled_s = np.asarray(
            despill(fg_s, green_limit_mode="average", strength=despill_strength), dtype=np.float32
        )
        fg_linear_s = np.asarray(srgb_to_linear(fg_despilled_s), dtype=np.float32)

        # Checkerboard composite for preview - cached after first frame.
        cb_key = (self.img_size, self.img_size)
        if cb_key not in self._checkerboard_cache:
            cb_srgb = create_checkerboard(self.img_size, self.img_size, checker_size=64, color1=0.15, color2=0.55)
            self._checkerboard_cache[cb_key] = np.asarray(srgb_to_linear(cb_srgb), dtype=np.float32)
        cb_linear_s = self._checkerboard_cache[cb_key]

        if fg_is_straight:
            comp_linear_s = np.asarray(composite_straight(fg_linear_s, cb_linear_s, alpha_s), dtype=np.float32)
        else:
            comp_linear_s = np.asarray(composite_premul(fg_linear_s, cb_linear_s, alpha_s), dtype=np.float32)

        comp_srgb_s = np.asarray(linear_to_srgb(comp_linear_s), dtype=np.float32)
        fg_premul_s = np.asarray(premultiply(fg_linear_s, alpha_s), dtype=np.float32)
        rgba_s = np.concatenate([fg_premul_s, alpha_s], axis=-1)

        # Upsample all outputs to original resolution.
        alpha_pred = cv2.resize(alpha_s, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
        fg_pred = cv2.resize(fg_s, (w, h), interpolation=cv2.INTER_LINEAR)
        composite_srgb = cv2.resize(comp_srgb_s, (w, h), interpolation=cv2.INTER_LINEAR)
        output_rgba = cv2.resize(rgba_s, (w, h), interpolation=cv2.INTER_LINEAR)

        # Source passthrough: blend original source pixels into opaque interior regions.
        # The model's fg prediction is only used in the edge transition band.
        # This preserves full source quality (no model artifacts) in solid areas.
        if source_passthrough:
            source_srgb = np.asarray(linear_to_srgb(image), dtype=np.float32) if input_is_linear else image
            fg_pred, output_rgba = _apply_source_passthrough(
                source_srgb, fg_pred, alpha_pred, edge_erode_px, edge_blur_px
            )

        return {
            "alpha": alpha_pred,  # linear float, raw prediction
            "fg": fg_pred,  # sRGB float, straight (unpremultiplied)
            "comp": composite_srgb,  # sRGB float, preview composite over checkerboard
            "processed": output_rgba,  # linear float, premultiplied RGBA
        }
