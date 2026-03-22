"""Postprocessor stage — public surface.

Entry point:
    postprocess_frame(result, config, stem="") -> PostprocessedFrame

Contracts:
    PostprocessConfig   — despill, despeckle, checkerboard options
    PostprocessedFrame  — alpha [H,W,1], fg [H,W,3], comp [H,W,3], numpy float32
"""

from corridorkey_new.stages.postprocessor.config import PostprocessConfig
from corridorkey_new.stages.postprocessor.contracts import PostprocessedFrame
from corridorkey_new.stages.postprocessor.orchestrator import postprocess_frame

__all__ = [
    "postprocess_frame",
    "PostprocessConfig",
    "PostprocessedFrame",
]
