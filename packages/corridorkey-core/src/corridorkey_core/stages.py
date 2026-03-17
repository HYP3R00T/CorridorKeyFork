"""Standalone stage functions for pipeline stages 3, 4, and 5.

These functions exist for tooling that needs to call individual stages
independently - e.g. benchmarking, visualising intermediate outputs, or
building a custom postprocessing step.

In the hot path, all three stages run fused inside CorridorKeyEngine.process_frame.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from corridorkey_core.compositing import (
    apply_source_passthrough,
    clean_matte,
    composite_premul,
    composite_straight,
    create_checkerboard,
    despill,
    linear_to_srgb,
    premultiply,
    srgb_to_linear,
)
from corridorkey_core.contracts import (
    PostprocessParams,
    PreprocessedTensor,
    ProcessedFrame,
    RawPrediction,
)

# ImageNet normalisation constants - the Hiera encoder was pretrained with these.
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def stage_3_preprocess(
    image: np.ndarray,
    mask: np.ndarray,
    source_h: int,
    source_w: int,
    img_size: int = 2048,
    device: str = "cpu",
    stem: str = "",
) -> PreprocessedTensor:
    """Stage 3: Resize, normalise, and stack image + mask into a model-ready tensor.

    Args:
        image: RGB float32 [H, W, 3] sRGB, values 0-1.
        mask: Grayscale float32 [H, W, 1] linear, values 0-1.
        source_h: Original frame height in pixels.
        source_w: Original frame width in pixels.
        img_size: Square resolution to resize to (must match model training resolution).
        device: Torch device string ("cuda", "mps", "cpu").
        stem: Filename stem carried through for output naming.

    Returns:
        PreprocessedTensor ready for stage_4_infer.
    """
    import torch

    image_resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask[:, :, 0], (img_size, img_size), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]

    image_norm = (image_resized - _IMAGENET_MEAN) / _IMAGENET_STD
    stacked = np.concatenate([image_norm, mask_resized], axis=-1)
    tensor = torch.from_numpy(stacked.transpose((2, 0, 1))).unsqueeze(0).to(torch.float32).to(device)

    return PreprocessedTensor(tensor=tensor, img_size=img_size, device=device, source_h=source_h, source_w=source_w)


def stage_4_infer(
    engine: Any,
    preprocessed: PreprocessedTensor,
    refiner_scale: float = 1.0,
) -> RawPrediction:
    """Stage 4: Run the model forward pass and return raw alpha + fg predictions.

    Args:
        engine: A CorridorKeyEngine instance.
        preprocessed: PreprocessedTensor from stage_3_preprocess.
        refiner_scale: Multiplier on the CNN refiner's delta corrections.

    Returns:
        RawPrediction with alpha and fg at model resolution.
    """
    import torch

    model_input = preprocessed.tensor.to(engine.model_precision)

    hook_handle = None
    if refiner_scale != 1.0 and hasattr(engine, "model") and engine.model.refiner is not None:

        def _scale_hook(module, inp, output):
            return output * refiner_scale

        hook_handle = engine.model.refiner.register_forward_hook(_scale_hook)

    try:
        with (
            torch.autocast(
                device_type=engine.device.type,
                dtype=torch.float16,
                enabled=getattr(engine, "mixed_precision", False),
            ),
            torch.inference_mode(),
        ):
            model_output = engine.model(model_input)
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    alpha = model_output["alpha"][0].permute(1, 2, 0).float().cpu().numpy()
    fg = model_output["fg"][0].permute(1, 2, 0).float().cpu().numpy()

    return RawPrediction(
        alpha=alpha,
        fg=fg,
        img_size=preprocessed.img_size,
        source_h=preprocessed.source_h,
        source_w=preprocessed.source_w,
    )


def stage_5_postprocess(
    raw: RawPrediction,
    source_image: np.ndarray,
    source_is_linear: bool = False,
    params: PostprocessParams | None = None,
    stem: str = "",
) -> ProcessedFrame:
    """Stage 5: Despeckle, despill, composite, upsample, and apply source passthrough.

    All operations run on CPU in NumPy. No filesystem access. No engine dependency.

    Args:
        raw: RawPrediction from stage_4_infer.
        source_image: Original source frame [H, W, 3] float32 sRGB, values 0-1.
        source_is_linear: True if the source image was originally in linear light.
        params: PostprocessParams controlling each step. Defaults to PostprocessParams().
        stem: Filename stem to carry through to ProcessedFrame.

    Returns:
        ProcessedFrame with all four outputs at original source resolution.
    """
    p = params or PostprocessParams()
    h, w = raw.source_h, raw.source_w

    alpha_s = raw.alpha
    fg_s = raw.fg

    if p.auto_despeckle:
        alpha_s = clean_matte(alpha_s, area_threshold=p.despeckle_size, dilation=25, blur_size=5)

    fg_despilled = np.asarray(despill(fg_s, green_limit_mode="average", strength=p.despill_strength), dtype=np.float32)
    fg_linear = np.asarray(srgb_to_linear(fg_despilled), dtype=np.float32)

    s = raw.img_size
    cb_srgb = create_checkerboard(s, s, checker_size=64, color1=0.15, color2=0.55)
    cb_linear = np.asarray(srgb_to_linear(cb_srgb), dtype=np.float32)

    if p.fg_is_straight:
        comp_linear = np.asarray(composite_straight(fg_linear, cb_linear, alpha_s), dtype=np.float32)
    else:
        comp_linear = np.asarray(composite_premul(fg_linear, cb_linear, alpha_s), dtype=np.float32)

    comp_srgb = np.asarray(linear_to_srgb(comp_linear), dtype=np.float32)
    fg_premul = np.asarray(premultiply(fg_linear, alpha_s), dtype=np.float32)
    rgba = np.concatenate([fg_premul, alpha_s], axis=-1)

    alpha_out = cv2.resize(alpha_s, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, np.newaxis]
    fg_out = cv2.resize(fg_s, (w, h), interpolation=cv2.INTER_LINEAR)
    comp_out = cv2.resize(comp_srgb, (w, h), interpolation=cv2.INTER_LINEAR)
    rgba_out = cv2.resize(rgba, (w, h), interpolation=cv2.INTER_LINEAR)

    if p.source_passthrough:
        src = np.asarray(linear_to_srgb(source_image), dtype=np.float32) if source_is_linear else source_image
        fg_out, rgba_out = apply_source_passthrough(src, fg_out, alpha_out, p.edge_erode_px, p.edge_blur_px)

    return ProcessedFrame(
        alpha=alpha_out,
        fg=fg_out,
        comp=comp_out,
        processed=rgba_out,
        source_h=h,
        source_w=w,
        stem=stem,
    )
