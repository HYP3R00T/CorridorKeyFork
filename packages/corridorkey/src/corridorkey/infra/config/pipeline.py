"""Top-level pipeline configuration — CorridorKeyConfig."""

from __future__ import annotations

import logging
import re
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from corridorkey.infra.config.inference import InferenceSettings
from corridorkey.infra.config.logging import LoggingSettings
from corridorkey.infra.config.postprocess import PostprocessSettings
from corridorkey.infra.config.preprocess import PreprocessSettings
from corridorkey.infra.config.writer import WriterSettings

logger = logging.getLogger(__name__)


class CorridorKeyConfig(BaseModel):
    """Validated top-level configuration for the CorridorKey pipeline.

    Nests one settings block per stage plus cross-cutting concerns (logging,
    device). All Path fields support tilde and environment variable expansion.

    Load with :func:`~corridorkey.infra.config.load_config`.

    Example ``corridorkey.toml``::

        [logging]
        level = "INFO"
        dir = "~/.config/corridorkey/logs"

        [preprocess]
        img_size = 2048
        image_upsample_mode = "bicubic"

        [inference]
        checkpoint_path = "~/models/greenformer.pth"
        use_refiner = true

        [postprocess]
        fg_upsample_mode = "bicubic"
        alpha_upsample_mode = "lanczos4"

        [writer]
        alpha_format = "exr"
        processed_format = "exr"
    """

    device: Annotated[
        str,
        Field(
            default="auto",
            description=(
                "Compute device(s) for inference. "
                "'auto' detects the best available device at runtime (ROCm > CUDA > MPS > CPU). "
                "'cuda' / 'cuda:0' forces NVIDIA GPU 0. "
                "'cuda:N' targets a specific GPU by index (e.g. 'cuda:1'). "
                "'all' uses every available CUDA GPU in parallel (frame-level dispatch). "
                "'rocm' / 'rocm:N' forces AMD GPU. "
                "'mps' forces Apple Silicon. 'cpu' forces CPU."
            ),
        ),
    ] = "auto"

    @field_validator("device", mode="before")
    @classmethod
    def _validate_device(cls, v: object) -> str:
        if not isinstance(v, str):
            raise ValueError(f"device must be a string, got {type(v).__name__}")
        v = v.strip().lower()
        # Valid simple tokens
        if v in ("auto", "cuda", "rocm", "mps", "cpu", "all"):
            return v
        # cuda:N or rocm:N
        if re.fullmatch(r"(cuda|rocm):\d+", v):
            return v
        raise ValueError(f"Invalid device '{v}'. Valid options: auto, cuda, cuda:N, rocm, rocm:N, mps, cpu, all.")

    logging: Annotated[
        LoggingSettings,
        Field(default_factory=LoggingSettings, description="Logging settings (level, output directory)."),
    ] = Field(default_factory=LoggingSettings)

    preprocess: Annotated[
        PreprocessSettings,
        Field(default_factory=PreprocessSettings, description="Preprocessing stage settings."),
    ] = Field(default_factory=PreprocessSettings)

    inference: Annotated[
        InferenceSettings,
        Field(default_factory=InferenceSettings, description="Inference stage settings."),
    ] = Field(default_factory=InferenceSettings)

    postprocess: Annotated[
        PostprocessSettings,
        Field(default_factory=PostprocessSettings, description="Postprocessing stage settings."),
    ] = Field(default_factory=PostprocessSettings)

    writer: Annotated[
        WriterSettings,
        Field(default_factory=WriterSettings, description="Writer stage settings."),
    ] = Field(default_factory=WriterSettings)

    # ------------------------------------------------------------------
    # Bridge methods — build stage runtime configs from this config
    # ------------------------------------------------------------------

    def to_pipeline_config(
        self,
        device: str | None = None,
        model=None,
    ):  # -> PipelineConfig
        """Build a :class:`~corridorkey.runtime.runner.PipelineConfig` from this config.

        Resolves device and img_size once, then builds all stage configs
        consistently. Pass the result directly to ``PipelineRunner``.

        Args:
            device: Resolved device string (from ``resolve_device(config.device)``).
                If None, uses ``self.device`` as-is.
            model: Pre-loaded model (``nn.Module``). If None, ``PipelineRunner``
                will load it from the checkpoint path at run time.

        Returns:
            PipelineConfig ready to pass to ``PipelineRunner``.
        """
        from corridorkey.runtime.runner import PipelineConfig

        resolved_device = device or self.device
        inference_config, resolved_refiner_mode = self.to_inference_config(
            device=resolved_device, _return_resolved_refiner_mode=True
        )

        return PipelineConfig(
            preprocess=self.to_preprocess_config(
                device=resolved_device,
                resolved_img_size=inference_config.img_size,
            ),
            inference=inference_config,
            model=model,
            postprocess=self.to_postprocess_config(),
            resolved_refiner_mode=resolved_refiner_mode,
        )

    def to_preprocess_config(
        self,
        device: str | None = None,
        resolved_img_size: int | None = None,
    ):  # -> PreprocessConfig
        """Build a :class:`~corridorkey.stages.preprocessor.PreprocessConfig`.

        Args:
            device: Override the device string. If None, uses ``self.device``.
            resolved_img_size: Pre-resolved img_size (from to_inference_config).
                If None, uses preprocess.img_size (or 2048 if 0).
        """
        from corridorkey.stages.preprocessor import PreprocessConfig

        img_size = resolved_img_size or self.preprocess.img_size or 2048

        return PreprocessConfig(
            img_size=img_size,
            device=device or self.device,
            image_upsample_mode=self.preprocess.image_upsample_mode,
            half_precision=self.preprocess.half_precision,
            source_passthrough=self.preprocess.source_passthrough,
            sharpen_strength=self.preprocess.sharpen_strength,
        )

    def to_postprocess_config(self):  # -> PostprocessConfig
        """Build a :class:`~corridorkey.stages.postprocessor.PostprocessConfig`."""
        from corridorkey.stages.postprocessor.config import PostprocessConfig

        return PostprocessConfig(
            fg_upsample_mode=self.postprocess.fg_upsample_mode,
            alpha_upsample_mode=self.postprocess.alpha_upsample_mode,
            despill_strength=self.postprocess.despill_strength,
            auto_despeckle=self.postprocess.auto_despeckle,
            despeckle_size=self.postprocess.despeckle_size,
            despeckle_dilation=self.postprocess.despeckle_dilation,
            despeckle_blur=self.postprocess.despeckle_blur,
            source_passthrough=self.postprocess.source_passthrough,
            edge_erode_px=self.postprocess.edge_erode_px,
            edge_blur_px=self.postprocess.edge_blur_px,
            hint_sharpen=self.postprocess.hint_sharpen,
            hint_sharpen_dilation=self.postprocess.hint_sharpen_dilation,
        )

    def to_writer_config(self, output_dir):  # -> WriteConfig
        """Build a :class:`~corridorkey.stages.writer.WriteConfig`.

        Args:
            output_dir: Root directory for all outputs (clip-specific).
        """
        from pathlib import Path

        from corridorkey.stages.writer.contracts import WriteConfig

        return WriteConfig(
            output_dir=Path(output_dir),
            alpha_enabled=self.writer.alpha_enabled,
            alpha_format=self.writer.alpha_format,
            fg_enabled=self.writer.fg_enabled,
            fg_format=self.writer.fg_format,
            processed_enabled=self.writer.processed_enabled,
            processed_format=self.writer.processed_format,
            comp_enabled=self.writer.comp_enabled,
            exr_compression=self.writer.exr_compression,
        )

    def to_inference_config(
        self, device: str | None = None, _return_resolved_refiner_mode: bool = False
    ):  # -> InferenceConfig | tuple[InferenceConfig, str]
        """Build an :class:`~corridorkey.stages.inference.InferenceConfig`.

        Probes VRAM at most once — the same measurement resolves both
        ``img_size`` (when set to 0/auto) and ``refiner_mode`` (when set to
        "auto"), avoiding two separate pynvml calls at startup.

        Args:
            device: Override the device string. If None, uses ``self.device``.
            _return_resolved_refiner_mode: Internal flag used by
                ``to_pipeline_config`` to receive the resolved refiner mode
                alongside the config so it can be stored in ``PipelineConfig``
                and passed to ``PipelineRunner`` without a second VRAM probe.
        """
        import torch

        from corridorkey.stages.inference import InferenceConfig
        from corridorkey.stages.inference.config import (
            _VRAM_TILED_THRESHOLD_GB,
            adaptive_img_size,
        )
        from corridorkey.stages.inference.orchestrator import _probe_vram_gb

        checkpoint = self.inference.checkpoint_path
        resolved_device = device or self.device

        _precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        if self.inference.model_precision == "auto":
            dev = torch.device(resolved_device)
            if dev.type == "cpu":
                logger.info("Precision auto -> float32 (CPU)")
                model_dtype = torch.float32
            elif dev.type == "mps":
                logger.info("Precision auto -> bfloat16 (Apple Silicon MPS)")
                model_dtype = torch.bfloat16
            elif dev.type == "cuda" and torch.cuda.is_available():
                props = torch.cuda.get_device_properties(dev)
                if props.major >= 8:
                    logger.info("Precision auto -> bfloat16 (Ampere+ GPU: %s)", props.name)
                    model_dtype = torch.bfloat16
                else:
                    logger.info("Precision auto -> float16 (pre-Ampere GPU: %s)", props.name)
                    model_dtype = torch.float16
            else:
                logger.info("Precision auto -> float32 (fallback)")
                model_dtype = torch.float32
        else:
            model_dtype = _precision_map[self.inference.model_precision]

        # Probe VRAM at most once — used for both img_size and refiner_mode
        # resolution when either is set to "auto" / 0.
        needs_vram_probe = self.preprocess.img_size == 0 or self.inference.refiner_mode == "auto"
        vram_gb = _probe_vram_gb(resolved_device) if needs_vram_probe else 0.0

        if self.preprocess.img_size == 0:
            img_size = adaptive_img_size(vram_gb)
            logger.info("img_size auto: %.1f GB VRAM detected → img_size=%d", vram_gb, img_size)
        else:
            img_size = self.preprocess.img_size

        # Resolve refiner_mode from the same VRAM reading — no second probe.
        if self.inference.refiner_mode == "auto":
            dev_type = torch.device(resolved_device).type
            if dev_type == "mps":
                resolved_refiner_mode = "tiled"
            elif vram_gb > 0 and vram_gb < _VRAM_TILED_THRESHOLD_GB:
                resolved_refiner_mode = "tiled"
                logger.info("refiner_mode auto: %.1f GB VRAM → tiled", vram_gb)
            else:
                resolved_refiner_mode = "full_frame"
                logger.info("refiner_mode auto: %.1f GB VRAM → full_frame", vram_gb)
        else:
            resolved_refiner_mode = self.inference.refiner_mode

        config = InferenceConfig(
            checkpoint_path=checkpoint,
            device=resolved_device,
            img_size=img_size,
            use_refiner=self.inference.use_refiner,
            mixed_precision=self.inference.mixed_precision,
            model_precision=model_dtype,
            refiner_mode=self.inference.refiner_mode,
            refiner_scale=self.inference.refiner_scale,
        )

        if _return_resolved_refiner_mode:
            return config, resolved_refiner_mode
        return config
