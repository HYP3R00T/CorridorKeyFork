"""Pipeline — wires preprocessing, inference, postprocessing, and writing
into a pipelined assembly line where all stages run concurrently.

Public API::

    from corridorkey_new.pipeline import PipelineRunner, PipelineConfig
"""

from corridorkey_new.pipeline.runner import PipelineConfig, PipelineRunner

__all__ = ["PipelineRunner", "PipelineConfig"]
