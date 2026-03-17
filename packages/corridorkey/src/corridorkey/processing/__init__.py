"""Processing subpackage -- inference contracts, output writing, and the service layer."""

from corridorkey.processing.contracts import FrameResult, InferenceParams, OutputConfig, WriteConfig
from corridorkey.processing.service import (
    CorridorKeyService,
    inference_params_to_postprocess,
    output_config_to_write_config,
)
from corridorkey.processing.writer import generate_masks, write_outputs

__all__ = [
    "CorridorKeyService",
    "FrameResult",
    "InferenceParams",
    "OutputConfig",
    "WriteConfig",
    "generate_masks",
    "inference_params_to_postprocess",
    "output_config_to_write_config",
    "write_outputs",
]
