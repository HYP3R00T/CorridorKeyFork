"""Inference stage — public surface.

Entry points:
    load_backend(config)                  -> ModelBackend  (backend-agnostic, preferred)
    load_model(config)                    -> nn.Module     (torch-only, lower-level)
    run_inference(frame, model, config)   -> InferenceResult

Contracts:
    InferenceConfig   — checkpoint path, device, precision, backend, optimization mode
    InferenceResult   — alpha [1,1,H,W], fg [1,3,H,W], FrameMeta
    ModelBackend      — protocol satisfied by TorchBackend and MLXBackend
"""

from corridorkey_new.stages.inference.backend import ModelBackend, TorchBackend
from corridorkey_new.stages.inference.config import BackendChoice, InferenceConfig, OptimizationMode
from corridorkey_new.stages.inference.contracts import InferenceResult
from corridorkey_new.stages.inference.factory import discover_checkpoint, load_backend
from corridorkey_new.stages.inference.loader import load_model
from corridorkey_new.stages.inference.orchestrator import run_inference

__all__ = [
    # Preferred entry point
    "load_backend",
    # Lower-level torch-only entry points
    "load_model",
    "run_inference",
    # Contracts
    "InferenceConfig",
    "InferenceResult",
    "OptimizationMode",
    "BackendChoice",
    # Backend protocol + implementations
    "ModelBackend",
    "TorchBackend",
    # Utilities
    "discover_checkpoint",
]
