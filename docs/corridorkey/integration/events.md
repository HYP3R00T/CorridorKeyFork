# Events

`PipelineEvents` is the package's progress reporting mechanism. It is a dataclass of optional callbacks that the pipeline fires as work progresses. The interface assigns only the callbacks it needs; unassigned callbacks are silently ignored.

Source: [`corridorkey/events.py`](https://github.com/hyp3r00t/CorridorKeyFork/blob/main/packages/corridorkey/src/corridorkey/events.py)

## Purpose

The pipeline runs on worker threads. The interface needs a way to receive notifications from those threads without polling or shared state. `PipelineEvents` provides that channel. Each callback fires on the worker thread that triggers it, so the interface receives notifications in real time as frames move through the pipeline.

## How It Works

A `PipelineEvents` instance is created by the interface and passed into `PipelineConfig` before the internal frame loop starts. The same instance is shared across the preprocess, inference, and postwrite workers, so the interface gets a unified view of the whole pipeline from a single object.

The callbacks are plain Python callables. They can be functions, methods, or lambdas. The only constraint is that they must return quickly. The worker thread that fires a callback blocks until it returns. A slow callback stalls the pipeline.

If a callback needs to do something slow (write to a database, send a network request, update a complex UI), it should put a message onto a separate queue and return immediately. A dedicated thread on the interface side can drain that queue without blocking the pipeline.

## Callback Reference

### Stage lifecycle

`on_stage_start(stage, total)` fires when a named stage begins. The `stage` argument is one of `"preprocess"`, `"inference"`, `"postwrite"`, or `"extract"`. The `total` argument is the frame count for the stage, or zero if the count is not yet known.

`on_stage_done(stage)` fires when a named stage has finished all its work.

### Frame progress

`on_frame_written(frame_index, total)` fires after a frame has been fully postprocessed and written to disk. This is the primary callback for a progress bar. `frame_index` is zero-based; `total` is the clip's frame count.

`on_frame_error(stage, frame_index, error)` fires when a frame is skipped due to an error. The frame is skipped rather than aborting the clip, so processing continues. The interface should log or display these so the user knows which frames failed and why.

### Assembly line visibility

`on_preprocess_queued(frame_index)` fires when a preprocessed frame is pushed onto the inference input queue.

`on_inference_start(frame_index)` fires when the inference worker picks up a frame.

`on_inference_queued(frame_index)` fires when an inference result is pushed onto the postwrite queue.

`on_queue_depth(preprocess_queue, postwrite_queue)` fires after every frame transition and reports the current number of items waiting in each inter-stage queue. Both values are point-in-time snapshots. A GUI can use these to render a live assembly-line view showing how many frames are at each stage simultaneously.

`on_extract_frame(frame_index, total)` fires once per frame during video extraction, before inference begins.

### Scan events

`on_clip_found(clip_name, clip_root)` fires as each valid clip is discovered during scanning. This allows a GUI to populate its clip list incrementally rather than waiting for the full scan to complete.

`on_clip_skipped(reason, path)` fires when a path is encountered during scanning but cannot be used. The `reason` string is human-readable and suitable for display.

### Clip lifecycle

`on_clip_complete(clip_name, frames_written)` fires after all frames for a clip have been written and all worker threads have exited cleanly. `frames_written` is the manifest frame count. Use this to update a batch progress list in a GUI.

`on_clip_error(clip_name, error)` fires when a clip does not complete -- either because it was cancelled (`JobCancelledError`) or because an unhandled exception occurred. It fires before the exception propagates to the caller, so the interface can update the clip's status before handling the error itself.

## Passing Events to the Scanner

`scan()` also accepts a `PipelineEvents` instance. When provided, `on_clip_found` and `on_clip_skipped` fire as the scan progresses. This is useful for interfaces that want to show clips appearing in the UI in real time rather than all at once after the scan finishes.

## Related

- [Engine](runner.md) - How the whole-clip orchestrator uses these callbacks.
- [Reference - events](../reference/events.md) - Full symbol reference.
