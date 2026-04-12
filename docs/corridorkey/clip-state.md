# Clip State Machine

The clip state machine tracks the processing lifecycle of a single clip from discovery through completed inference. It lives in `corridorkey.runtime.clip_state` and is consumed by any interface layer.

## States

A clip moves through five states:

| State | Meaning |
|---|---|
| `EXTRACTING` | Video source present, frame sequence not yet extracted. |
| `RAW` | Frame sequence present, no alpha hint. |
| `READY` | Alpha hint present - clip can be submitted for inference. |
| `COMPLETE` | Inference has run and all output frames are written. |
| `ERROR` | A stage failed. `ClipRecord.error_message` contains the detail. |

## Valid Transitions

| From | To |
|---|---|
| `EXTRACTING` | `RAW`, `ERROR` |
| `RAW` | `READY`, `ERROR` |
| `READY` | `COMPLETE`, `ERROR` |
| `COMPLETE` | `READY` |
| `ERROR` | `RAW`, `READY`, `EXTRACTING` |

Calling `transition_to()` with an invalid target raises `InvalidStateTransitionError`.

## ClipRecord

`ClipRecord` wraps a `Clip` (scanner output) and optionally a `ClipManifest` (loader output). It does not re-scan the filesystem - that is the scanner's job.

State is resolved from disk at construction time via `from_clip()`. The resolution priority is:

1. `COMPLETE` - output frames cover all input frames
2. `READY` - alpha frames cover all input frames
3. `EXTRACTING` - input is a video file
4. `RAW` - frame sequence present, no alpha

## Processing Lock

`_processing` is a soft lock set by the job queue while a GPU job is active. Filesystem watchers should skip reclassification while `is_processing` is `True`.

## FrameRange

`FrameRange` represents an inclusive in/out frame range for sub-clip processing. Both indices are zero-based and inclusive. Use `to_frame_range()` to convert to a half-open `(start, end)` tuple for `ClipManifest.frame_range`.

## Related

- [Job Queue](job-queue.md) - How the queue interacts with the processing lock.
- [Reference - clip-state](reference/clip-state.md) - Full symbol reference.
