"""Runtime — stateful coordination layer.

Contains everything that tracks state across pipeline invocations:

    clip_state.py  — ClipEntry / ClipState / InOutRange state machine
    queue.py       — BoundedQueue with sentinel-based shutdown
    worker.py      — PreprocessWorker, InferenceWorker, PostWriteWorker
    runner.py      — PipelineRunner / PipelineConfig

Import from corridorkey_new directly, not from this package.
"""

from corridorkey_new.events import PipelineEvents
from corridorkey_new.runtime.runner import PipelineConfig, PipelineRunner

__all__ = ["PipelineRunner", "PipelineConfig", "PipelineEvents"]
