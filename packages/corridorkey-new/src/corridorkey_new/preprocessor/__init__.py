"""Preprocessing stage.

Public API::

    from corridorkey_new.preprocessor import (
        preprocess_frame,
        PreprocessConfig,
        PreprocessedFrame,
        FrameMeta,
        FrameReadError,
    )
"""

from corridorkey_new.preprocessor.orchestrator import FrameMeta, PreprocessConfig, PreprocessedFrame, preprocess_frame
from corridorkey_new.preprocessor.reader import FrameReadError

__all__ = ["preprocess_frame", "PreprocessConfig", "PreprocessedFrame", "FrameMeta", "FrameReadError"]
