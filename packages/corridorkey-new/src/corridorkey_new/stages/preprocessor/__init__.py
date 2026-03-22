"""Preprocessing stage.

Public API::

    from corridorkey_new.stages.preprocessor import (
        preprocess_frame,
        PreprocessConfig,
        PreprocessedFrame,
        FrameMeta,
        FrameReadError,
        UpsampleMode,
    )
"""

from corridorkey_new.stages.preprocessor.orchestrator import (
    FrameMeta,
    PreprocessConfig,
    PreprocessedFrame,
    preprocess_frame,
)
from corridorkey_new.stages.preprocessor.reader import FrameReadError
from corridorkey_new.stages.preprocessor.resize import (
    DEFAULT_ALPHA_UPSAMPLE_MODE,
    DEFAULT_UPSAMPLE_MODE,
    UpsampleMode,
)

__all__ = [
    "preprocess_frame",
    "PreprocessConfig",
    "PreprocessedFrame",
    "FrameMeta",
    "FrameReadError",
    "UpsampleMode",
    "DEFAULT_UPSAMPLE_MODE",
    "DEFAULT_ALPHA_UPSAMPLE_MODE",
]
