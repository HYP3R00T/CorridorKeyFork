"""Preprocessing stage.

Public API::

    from corridorkey_new.stages.preprocessor import (
        preprocess_frame,
        PreprocessConfig,
        PreprocessedFrame,
        FrameMeta,
        FrameReadError,
        LetterboxPad,
        ImageUpsampleMode,
        DEFAULT_IMAGE_UPSAMPLE_MODE,
    )
"""

from corridorkey_new.stages.preprocessor.contracts import FrameMeta, PreprocessedFrame
from corridorkey_new.stages.preprocessor.orchestrator import (
    PreprocessConfig,
    preprocess_frame,
)
from corridorkey_new.stages.preprocessor.reader import FrameReadError
from corridorkey_new.stages.preprocessor.resize import (
    DEFAULT_IMAGE_UPSAMPLE_MODE,
    ImageUpsampleMode,
    LetterboxPad,
)

__all__ = [
    "preprocess_frame",
    "PreprocessConfig",
    "PreprocessedFrame",
    "FrameMeta",
    "FrameReadError",
    "LetterboxPad",
    "ImageUpsampleMode",
    "DEFAULT_IMAGE_UPSAMPLE_MODE",
]
