# Frame Loop

The frame loop is a direct integration pattern where the interface calls each pipeline stage function individually, one frame at a time. The interface controls the loop, the threading model, and the order of operations.

Source: [`corridorkey/__init__.py`](https://github.com/hyp3r00t/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/__init__.py)

## Purpose

Host applications like DaVinci Resolve Fusion and Adobe Premiere Pro have their own rendering engines. They call into plugins one frame at a time, on their own threads, according to their own scheduling. `PipelineRunner` is not compatible with this model because it manages its own threads and blocks until the entire clip is done.

The frame loop pattern exposes each pipeline stage as a standalone function call. The interface calls them in sequence for each frame, in whatever threading context the host provides.

## How It Works

The pipeline is broken into four function calls per frame.

`preprocess_frame()` reads one frame pair (image and alpha) from disk, converts it to a tensor, and moves it to the configured device. It returns a `PreprocessedFrame` containing the tensor and the original frame dimensions.

`backend.run()` runs the model forward pass on the preprocessed frame and returns an `InferenceResult` containing the raw alpha and foreground predictions, still on the device.

`postprocess_frame()` takes the inference result, resizes the predictions back to source resolution, applies despill and despeckle, and returns a `PostprocessedFrame` containing NumPy arrays ready to write.

`write_frame()` takes the postprocessed frame and writes all enabled output files to disk.

Each function is stateless. None of them hold references to previous frames or accumulate state between calls. This makes them safe to call from any thread.

## The Backend

The frame loop uses `load_backend()` rather than `load_model()`. `load_backend()` is the preferred entry point because it handles the backend selection decision automatically. On Apple Silicon with the optional MLX package installed, it returns an `MLXBackend`. On all other platforms, it returns a `TorchBackend`. Both satisfy the same `ModelBackend` protocol, so the interface does not need to know which one it received.

The backend is loaded once per session, not once per frame. Loading involves reading the checkpoint from disk, moving weights to the device, and optionally compiling GPU kernels. This takes several seconds on first run. The interface should load the backend during startup and hold the reference for the lifetime of the session.

After loading, `backend.resolved_config` returns a flat dictionary of strings describing what was actually resolved at runtime: the backend name, device, refiner mode, precision, and image size. This is useful for populating a status panel or writing a startup log entry.

## File Lists

`preprocess_frame()` accepts optional pre-built file lists for the image and alpha directories. When these are provided, the function uses them directly instead of scanning the directory on every call. For a clip with hundreds of frames, scanning the directory on every frame call adds significant overhead. The interface should build these lists once when the clip is loaded and pass them on every frame call.

`scan_frames()` performs a single directory scan and returns a `FrameScan` containing the sorted file list and a flag indicating whether the frames are in linear colour space. `get_frame_files()` is a convenience wrapper that returns just the file list.

## Memory Management

In long-running host applications, GPU memory can fragment over time. Calling `clear_device_cache(device)` between clips releases the PyTorch memory cache and returns unused memory to the driver. This is a no-op on CPU. On CUDA and MPS it calls the appropriate cache-clearing function.

This call is not needed when processing a single clip in a short-lived process. It matters for GUI applications and plugins that process many clips in a single session.

## When to Use the Frame Loop

The frame loop is the right choice when:

- The host application manages its own threading and calls into the plugin one frame at a time.
- The interface needs to inspect or modify the result of each stage before passing it to the next.
- The interface needs to interleave pipeline calls with other work on the same thread.

It is not the right choice for a CLI or GUI that processes whole clips. For those cases, `PipelineRunner` is simpler and handles threading automatically.

## Related

- [Runner](runner.md) - Whole-clip processing with internal threading.
- [Startup](startup.md) - How to load the backend and build stage configs before the loop begins.
- [Reference - pipeline](../reference/pipeline.md) - Full signatures for all stage functions.
