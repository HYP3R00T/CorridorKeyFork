# MultiGPURunner

`MultiGPURunner` processes a clip across multiple CUDA GPUs simultaneously. Each GPU runs its own inference worker, and all workers share a single frame queue. The preprocessor and postwriter remain single-threaded.

Source: [`corridorkey/runtime/runner.py`](https://github.com/edenaion/CorridorKey/blob/main/packages/corridorkey/src/corridorkey/runtime/runner.py)

## Purpose

A single GPU processes frames sequentially. When multiple GPUs are available, the pipeline can dispatch frames to whichever GPU finishes first, reducing total processing time roughly proportionally to the number of GPUs.

## How It Works

The structure is the same as `PipelineRunner` with one difference: instead of one inference worker, there are N inference workers, one per GPU. All N workers pull from the same input queue and push to the same output queue. The preprocessor feeds all of them; the postwriter drains all of them.

Each GPU gets its own copy of the model, loaded in a parallel thread at startup. Loading happens concurrently so the startup cost does not multiply with the number of GPUs.

The queue depths scale automatically with the number of GPUs. The runner ensures each GPU always has at least one frame buffered ahead of it, so no GPU sits idle waiting for the preprocessor.

### Frame ordering

When multiple GPUs run at different speeds, inference results arrive on the output queue out of order. The postwriter writes frames as they arrive. If the downstream compositor requires frames in strict order, the interface must sort the output files by frame index after the run completes. The `stem` field on each output file encodes the frame index.

### Shutdown

When the preprocessor finishes, it sends a single shutdown signal into the input queue. Each inference worker, upon receiving it, re-inserts the signal so the next worker also sees it, then exits. The last worker to exit sends a shutdown signal downstream to the postwriter. This ensures the postwriter receives exactly one shutdown signal regardless of how many GPU workers are running.

## Configuration

`MultiGPURunner` takes a `MultiGPUConfig` instead of `PipelineConfig`. The key difference is the `devices` field, which is a list of PyTorch device strings.

`resolve_devices("all")` returns a list of every available CUDA device on the machine. Pass a subset explicitly to use only specific GPUs, for example when one GPU is reserved for display output.

The `inference` field on `MultiGPUConfig` is a base `InferenceConfig`. The device field within it is overridden per worker - all other fields (checkpoint path, precision, refiner mode) are shared across all GPUs.

## When to Use MultiGPURunner

`MultiGPURunner` is the right choice when:

- The machine has two or more CUDA GPUs.
- The clip is long enough that the parallel startup cost (loading N model copies) is worth the throughput gain.
- Frame ordering in the output is either not required or can be handled by sorting after the run.

It is not useful for ROCm or MPS devices. Multi-GPU dispatch is CUDA-only.

## Related

- [PipelineRunner](pipeline-runner.md) - Single-GPU pipeline runner.
- [Events](events.md) - Progress callbacks, which work identically for both runners.
- [API Reference - device-utils](../../../api/corridorkey/device-utils.md) - `resolve_devices`.
