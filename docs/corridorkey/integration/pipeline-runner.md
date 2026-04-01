# PipelineRunner

`PipelineRunner` is the standard way to process a clip. It wires the three pipeline workers together, manages the queues between them, and blocks until every frame has been written to disk.

Source: [`corridorkey/runtime/runner.py`](https://github.com/hyp3r00t/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/runtime/runner.py)

## Purpose

Most interfaces do not need to think about individual frames. They need to hand a clip to the pipeline and be notified when it is done. `PipelineRunner` is that hand-off point. The interface provides a manifest and a config; the runner handles everything in between.

## How It Works

When `run()` is called, the runner starts three daemon threads and blocks until all three finish.

The first thread is the preprocessor. It reads frames from disk one at a time, converts them to tensors, and pushes them onto a bounded input queue. When it has processed every frame in the clip's range, it signals the queue that no more frames are coming.

The second thread is the inference worker. It pulls tensors from the input queue, runs the model forward pass, and pushes the results onto a bounded output queue. It exits when it receives the shutdown signal from the preprocessor.

The third thread is the postwrite worker. It pulls inference results from the output queue, runs postprocessing (resize, despill, despeckle, composite), and writes all enabled output files to disk. It exits when it receives the shutdown signal from the inference worker.

The two queues between the threads are bounded. When a queue is full, the producing thread blocks. This is backpressure: the pipeline runs at the speed of its slowest stage without accumulating unbounded memory.

## Configuration

The runner is configured through `PipelineConfig`, which is built by calling `config.to_pipeline_config(device=device)` on the loaded `CorridorKeyConfig`. This method resolves VRAM once and derives all stage configs consistently from that single measurement. The interface should always use this method rather than constructing stage configs individually.

`PipelineConfig` also accepts an optional `events` field for progress callbacks. See [Events](events.md) for the full callback reference.

The `input_queue_depth` and `output_queue_depth` fields on `PipelineConfig` control how many frames can be buffered between stages. The default of 2 keeps one frame in flight and one buffered per stage. Each buffered preprocessed frame occupies roughly 64 MB of GPU memory at 2048 resolution, so increasing depth trades VRAM for throughput.

## Scanning and Loading

Before the runner can be used, the interface must scan for clips and load each one.

`scan()` accepts a path to a clips directory, a single clip folder, or a single video file. It returns a `ScanResult` containing a tuple of valid `Clip` objects and a tuple of `SkippedPath` objects. The interface should present both to the user. Skipped paths carry a human-readable reason explaining why they could not be used.

`load()` accepts a `Clip` and returns a `ClipManifest`. The manifest contains the resolved frame paths, output directory, frame count, and a `needs_alpha` flag.

When `needs_alpha` is `True`, the clip has no alpha hint frames. The interface is responsible for generating them using an external tool. Once generated, `resolve_alpha()` is called with the path to the new alpha frames. It validates the count matches the input frame count and returns an updated manifest with `needs_alpha` set to `False`. The runner cannot be started until `needs_alpha` is `False`.

## When to Use PipelineRunner

`PipelineRunner` is the right choice when:

- The interface does not need to inspect or modify individual frame results before they are written.
- The interface manages one GPU or wants the pipeline to handle threading internally.
- Progress reporting through callbacks is sufficient.

It is not the right choice when the host application manages its own threading model and expects to call into the pipeline one frame at a time. For that case, see [Frame Loop](frame-loop.md).

## Related

- [MultiGPURunner](multi-gpu-runner.md) - Running the same pipeline across multiple GPUs in parallel.
- [Frame Loop](frame-loop.md) - Per-frame control for host-managed threading.
- [Events](events.md) - Progress callbacks for the runner.
- [Job Queue](../job-queue.md) - How the bounded queues and shutdown signalling work internally.
