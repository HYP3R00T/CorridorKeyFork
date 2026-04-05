"""Runtime — stateful coordination layer.

Contains everything that tracks state across pipeline invocations:

    clip_state.py  — ClipEntry / ClipState / InOutRange state machine
    queue.py       — BoundedQueue with sentinel-based shutdown
    worker.py      — PreprocessWorker, PostWriteWorker
    runner.py      — Runner / PipelineConfig

Import from corridorkey directly, not from this package.
"""

from corridorkey.events import PipelineEvents
from corridorkey.runtime.runner import PipelineConfig, Runner

__all__ = [
    "Runner",
    "PipelineConfig",
    "PipelineEvents",
]
