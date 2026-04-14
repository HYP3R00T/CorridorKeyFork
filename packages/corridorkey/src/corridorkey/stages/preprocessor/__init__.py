"""Preprocessing stage.

Public API::

    from corridorkey.stages.preprocessor import (
        preprocess_frame,
        PreprocessConfig,
        PreprocessedFrame,
        FrameMeta,
        FrameReadError,
    )
"""

from corridorkey.stages.preprocessor.contracts import FrameMeta, PreprocessedFrame
from corridorkey.stages.preprocessor.orchestrator import PreprocessConfig, preprocess_frame
from corridorkey.stages.preprocessor.reader import FrameReadError

__all__ = [
    "preprocess_frame",
    "PreprocessConfig",
    "PreprocessedFrame",
    "FrameMeta",
    "FrameReadError",
]
