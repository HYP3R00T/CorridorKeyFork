"""Runtime — internal coordination layer.

Contains the threading infrastructure used by the Engine internally:

    clip_state.py  — ClipRecord / ClipState / FrameRange state machine
    queue.py       — BoundedQueue with sentinel-based shutdown
    worker.py      — PreprocessWorker, PostWriteWorker
    runner.py      — run_clip / PipelineConfig (internal frame-loop engine)

Import from corridorkey directly, not from this package.
"""
