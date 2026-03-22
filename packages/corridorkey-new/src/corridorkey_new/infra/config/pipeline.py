"""Top-level pipeline configuration — CorridorKeyConfig."""

from __future__ import annotations

import logging
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from corridorkey_new.infra.config.inference import InferenceSettings
from corridorkey_new.infra.config.logging import LoggingSettings
from corridorkey_new.infra.config.postprocess import PostprocessSettings
from corridorkey_new.infra.config.preprocess import PreprocessSettings
from corridorkey_new.infra.config.writer import WriterSettings

logger = logging.getLogger(__name__)


class CorridorKeyConfig(BaseModel):
    """Validated top-level configuration for the CorridorKey pipeline.

    Nests one settings block per stage plus cross-cutting concerns (logging,
    device). All Path fields support tilde and environment variable expansion.

    Load with :func:`~corridorkey_new.infra.config.load_config`.

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
        Literal["auto", "cuda", "rocm", "mps", "cpu"],
        Field(
            default="auto",
            description=(
                "Compute device for inference. "
                "'auto' detects the best available device at runtime (ROCm > CUDA > MPS > CPU). "
                "'cuda' forces NVIDIA GPU. 'rocm' forces AMD GPU. "
                "'mps' forces Apple Silicon. 'cpu' forces CPU."
            ),
        ),
    ] = "auto"

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
    # Convenience shims — keep old flat field access working during
    # transition. These delegate to the nested settings blocks.
    # ------------------------------------------------------------------

    @property
    def log_dir(self):
        return self.logging.dir

    @property
    def log_level(self):
        return self.logging.level

    # ------------------------------------------------------------------
    # Bridge methods — build stage runtime configs from this config
    # ------------------------------------------------------------------

    def to_pipeline_config(
        self,
        device: str | None = None,
        model=None,
    ):  # -> PipelineConfig
        """Build a :class:`~corridorkey_new.runtime.runner.PipelineConfig` from this config.

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
        from corridorkey_new.runtime.runner import PipelineConfig

        resolved_device = device or self.device
        inference_config = self.to_inference_config(device=resolved_device)

        return PipelineConfig(
            preprocess=self.to_preprocess_config(
                device=resolved_device,
                resolved_img_size=inference_config.img_size,
            ),
            inference=inference_config,
            model=model,
            postprocess=self.to_postprocess_config(),
        )

    def to_preprocess_config(
        self,
        device: str | None = None,
        resolved_img_size: int | None = None,
    ):  # -> PreprocessConfig
        """Build a :class:`~corridorkey_new.stages.preprocessor.PreprocessConfig`.

        Args:
            device: Override the device string. If None, uses ``self.device``.
            resolved_img_size: Pre-resolved img_size (from to_inference_config).
                If None, uses preprocess.img_size (or 2048 if 0).
        """
        from corridorkey_new.stages.preprocessor import PreprocessConfig

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
        """Build a :class:`~corridorkey_new.stages.postprocessor.PostprocessConfig`."""
        from corridorkey_new.stages.postprocessor.config import PostprocessConfig

        return PostprocessConfig(
            fg_upsample_mode=self.postprocess.fg_upsample_mode,
            alpha_upsample_mode=self.postprocess.alpha_upsample_mode,
            despill_strength=self.postprocess.despill_strength,
            auto_despeckle=self.postprocess.auto_despeckle,
            despeckle_size=self.postprocess.despeckle_size,
            source_passthrough=self.postprocess.source_passthrough,
            edge_erode_px=self.postprocess.edge_erode_px,
            edge_blur_px=self.postprocess.edge_blur_px,
        )

    def to_writer_config(self, output_dir):  # -> WriteConfig
        """Build a :class:`~corridorkey_new.stages.writer.WriteConfig`.

        Args:
            output_dir: Root directory for all outputs (clip-specific).
        """
        from pathlib import Path

        from corridorkey_new.stages.writer.contracts import WriteConfig

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

    def to_inference_config(self, device: str | None = None):  # -> InferenceConfig
        """Build an :class:`~corridorkey_new.stages.inference.InferenceConfig`.

        Args:
            device: Override the device string. If None, uses ``self.device``.
        """
        import torch

        from corridorkey_new.infra.model_hub import default_checkpoint_path
        from corridorkey_new.stages.inference import InferenceConfig
        from corridorkey_new.stages.inference.config import adaptive_img_size
        from corridorkey_new.stages.inference.orchestrator import _probe_vram_gb

        checkpoint = self.inference.checkpoint_path or default_checkpoint_path()
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

        if self.preprocess.img_size == 0:
            vram_gb = _probe_vram_gb(resolved_device)
            img_size = adaptive_img_size(vram_gb)
            logger.info("img_size auto: %.1f GB VRAM detected → img_size=%d", vram_gb, img_size)
        else:
            img_size = self.preprocess.img_size

        return InferenceConfig(
            checkpoint_path=checkpoint,
            device=resolved_device,
            img_size=img_size,
            use_refiner=self.inference.use_refiner,
            mixed_precision=self.inference.mixed_precision,
            model_precision=model_dtype,
            optimization_mode=self.inference.optimization_mode,
            refiner_scale=self.inference.refiner_scale,
        )
