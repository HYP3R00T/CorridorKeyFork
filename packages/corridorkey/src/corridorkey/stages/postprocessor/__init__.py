"""Postprocessor stage — public surface.

Entry point:
    postprocess_frame(result, config, stem="") -> ProcessedFrame

Contracts:
    PostprocessConfig   — despill, despeckle, checkerboard options
    ProcessedFrame      — alpha [H,W,1], fg [H,W,3], comp [H,W,3], numpy float32
"""

from corridorkey.stages.postprocessor.config import PostprocessConfig
from corridorkey.stages.postprocessor.contracts import ProcessedFrame
from corridorkey.stages.postprocessor.orchestrator import postprocess_frame

__all__ = [
    "postprocess_frame",
    "PostprocessConfig",
    "ProcessedFrame",
]
