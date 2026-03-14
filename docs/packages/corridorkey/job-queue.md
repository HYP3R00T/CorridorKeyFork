# Job Queue

`GPUJobQueue` ensures only one GPU job runs at a time. It is designed for GUI and async consumers where inference, alpha generation, and video extraction are submitted from one thread and processed on a worker thread.

CLI and batch consumers do not need the job queue - `process_directory` handles sequencing internally.

## Why One Job at a Time

CorridorKey's inference model requires approximately 22.7 GB of VRAM on a 24 GB card. Running two jobs concurrently would exhaust VRAM. The queue serialises all GPU work through a single lock.

## Job Types

| Type | Description |
|---|---|
| `INFERENCE` | Full inference pass on a clip |
| `ALPHA_GEN` | Alpha hint generation via any `AlphaGenerator` |
| `PREVIEW_REPROCESS` | Single-frame reprocess for live GUI preview |
| `VIDEO_EXTRACT` | FFmpeg frame extraction from a source video |
| `VIDEO_STITCH` | FFmpeg frame stitching back to a video |

## Job Lifecycle

A job moves through these states:

```text
QUEUED -> RUNNING -> COMPLETED
                  -> CANCELLED
                  -> FAILED
```

## Deduplication vs Replacement

Submitting the same clip and job type twice is rejected by default. The second `submit` call returns `False` and logs a warning. This prevents double-processing when the user clicks a button twice.

`PREVIEW_REPROCESS` is the exception: it uses replacement semantics instead of rejection. A new preview job for any clip replaces all existing preview jobs in the queue. This keeps the preview responsive - only the latest request matters.

## Cancellation Contract

When a running job is cancelled, the queue sets `job.is_cancelled = True` but does not interrupt the processing function. The processing function is responsible for checking this flag between frames and raising `JobCancelledError` to stop cleanly. This cooperative model avoids leaving GPU state in an inconsistent condition.

## Using the Queue vs process_directory

The job queue is an optional layer. `CorridorKeyService` exposes a lazy-initialised `job_queue` property for GUI consumers that want to manage jobs themselves. Batch and CLI consumers should call `process_directory` directly - it handles sequencing without the overhead of the queue.

## Related

- [job-queue reference](../../api/corridorkey/job-queue.md)
- [service reference](../../api/corridorkey/service.md)
- [errors reference](../../api/corridorkey/errors.md)
