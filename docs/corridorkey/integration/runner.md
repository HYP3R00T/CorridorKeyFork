# Engine

`Engine` is the standard way to process clips. It wires the pipeline workers together, manages the queues between them, and blocks until every frame has been written to disk. It works identically whether you have one GPU or twenty.

Source: [`corridorkey/runtime/runner.py`](https://github.com/hyp3r00t/CorridorKeyFork/blob/main/packages/corridorkey/src/corridorkey/runtime/runner.py)

## Purpose

`Engine` processes one clip at a time, but a single `Engine` instance can scan and process multiple clip paths in one `run()` call.

Most interfaces do not need to think about individual frames. They need to hand a clip to the pipeline and be notified when it is done. `Engine` is that hand-off point. The interface provides a config; the engine handles scanning, loading, alpha generation, and frame processing.

## How It Works

When `Engine.run()` is called, the engine scans the input paths, loads each clip, and then starts one `PreprocessWorker`, one `PostWriteWorker`, and one `InferenceWorker` per device for each clip. It blocks until all threads finish.

The preprocessor reads frames from disk one at a time, converts them to tensors, and pushes them onto a bounded input queue. When it has processed every frame, it signals the queue that no more frames are coming.

Each inference worker pulls tensors from the shared input queue, runs the model forward pass, and pushes results onto a shared output queue. When a worker sees the shutdown signal, it re-inserts it so sibling workers also see it, then decrements a shared counter. The last worker to exit sends the shutdown signal downstream.

The postwriter pulls inference results from the output queue, runs postprocessing (resize, despill, despeckle, composite), and writes all enabled output files to disk.

The two queues are bounded. When a queue is full, the producing thread blocks. This is backpressure: the pipeline runs at the speed of its slowest stage without accumulating unbounded memory. Queue depths scale automatically with the number of devices so each device always has at least one frame buffered.

## Single GPU vs Multiple GPUs

The assembly line is the same regardless of device count. With one device there is one inference worker; with N devices there are N inference workers, all pulling from the same input queue and pushing to the same output queue. The preprocessor and postwriter are always single-threaded.

When multiple GPUs are used, each gets its own model copy loaded in a parallel thread at startup, so the startup cost does not multiply with the number of GPUs. Frame ordering is not guaranteed when multiple workers run at different speeds — the postwriter writes frames as they arrive.

## Configuration

`Engine` is constructed from a loaded `CorridorKeyConfig`. Internally it builds a `PipelineConfig` by calling `config.to_pipeline_config()`.

```python
from pathlib import Path

from corridorkey import Engine, load_config

config = load_config()
engine = Engine(config)
engine.run([Path("/clips")])
```

When the config resolves to a single device, one inference worker runs. When it resolves to multiple devices, one worker per device runs. The code path is the same in both cases.

For accepted device string values, see [Configuration — Device values](../configuration.md#device-values).

`Engine.on()` provides the clip-level callback surface for interfaces. The lower-level `PipelineConfig.events` field is used internally by the frame loop for stage and frame progress. See [Events](events.md) for the full callback reference.

The `input_queue_depth` and `output_queue_depth` fields control how many frames can be buffered between stages. Each buffered preprocessed frame occupies roughly 64 MB of GPU memory at 2048 resolution.

## Scanning and Loading

Before the engine can process a clip, it scans the input paths and loads each clip internally.

`scan()` accepts a path to a clips directory, a single clip folder, or a single video file. It returns a `ScanResult` containing a tuple of valid `Clip` objects and a tuple of `SkippedPath` objects. The interface should present both to the user.

`load()` accepts a `Clip` and returns a `ClipManifest`. When `needs_alpha` is `True`, the clip has no alpha hint frames. The interface is responsible for generating them externally. Once generated, `attach_alpha()` returns an updated manifest with `needs_alpha` set to `False`. The engine cannot proceed until `needs_alpha` is `False`.

## When to Use Engine

`Engine` is the right choice when:

- The interface does not need to inspect or modify individual frame results before they are written.
- Progress reporting through callbacks is sufficient.

It is not the right choice when the host application manages its own threading model and expects to call into the pipeline one frame at a time. For that case, see [Frame Loop](frame-loop.md).

## Related

- [Frame Loop](frame-loop.md) - Per-frame control for host-managed threading.
- [Events](events.md) - Progress callbacks.
- [Job Queue](../job-queue.md) - How the bounded queues and shutdown signalling work internally.
