"""Inference stage — public surface.

Entry points:
    load_backend(config)                  -> ModelBackend  (backend-agnostic, preferred)
    load_model(config)                    -> nn.Module     (torch-only, lower-level)
    run_inference(frame, model, config)   -> InferenceResult

Contracts:
    InferenceConfig   — checkpoint path, device, precision, backend, refiner_mode
    InferenceResult   — alpha [1,1,H,W], fg [1,3,H,W], FrameMeta
    ModelBackend      — protocol satisfied by TorchBackend and MLXBackend
"""

from corridorkey.stages.inference.backend import ModelBackend
from corridorkey.stages.inference.config import BackendChoice, InferenceConfig, RefinerMode
from corridorkey.stages.inference.contracts import InferenceResult
from corridorkey.stages.inference.factory import load_backend

__all__ = [
    # Preferred entry point
    "load_backend",
    # Contracts
    "InferenceConfig",
    "InferenceResult",
    "RefinerMode",
    "BackendChoice",
    # Backend protocol
    "ModelBackend",
]
