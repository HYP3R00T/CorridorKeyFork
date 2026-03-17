"""Pipeline subpackage -- internal compute stages for corridorkey-core.

Not part of the public API. Import create_engine from corridorkey_core instead.
"""

from corridorkey_core.pipeline.contracts import (
    PostprocessParams,
    PreprocessedTensor,
    ProcessedFrame,
    RawPrediction,
)
from corridorkey_core.pipeline.engine import CorridorKeyEngine
from corridorkey_core.pipeline.stages import stage_3_preprocess, stage_4_infer, stage_5_postprocess

__all__ = [
    "CorridorKeyEngine",
    "PostprocessParams",
    "PreprocessedTensor",
    "ProcessedFrame",
    "RawPrediction",
    "stage_3_preprocess",
    "stage_4_infer",
    "stage_5_postprocess",
]
