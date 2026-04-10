"""Inference stage — public surface.

Entry points:
    load_model_backend(config)            -> ModelBackend  (backend-agnostic, preferred)
    load_model(config)                    -> nn.Module     (torch-only, lower-level)
    run_inference(frame, model, config)   -> InferenceResult

Contracts:
    InferenceConfig   — checkpoint path, device, precision, backend, refiner_mode
    InferenceResult   — alpha [1,1,H,W], fg [1,3,H,W], FrameMeta
    ModelBackend      — protocol satisfied by TorchBackend and MLXBackend
"""

from corridorkey.stages.inference.backend import ModelBackend
from corridorkey.stages.inference.config import InferenceConfig
from corridorkey.stages.inference.contracts import InferenceResult
from corridorkey.stages.inference.factory import load_model_backend

__all__ = [
    "load_model_backend",
    "InferenceConfig",
    "InferenceResult",
    "ModelBackend",
]
