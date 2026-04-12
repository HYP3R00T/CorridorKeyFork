# Error Handling

All errors raised by the `corridorkey` package are typed exceptions that inherit from `CorridorKeyError`. The interface catches them and presents them to the user.

Source: [`corridorkey/errors.py`](https://github.com/hyp3r00t/CorridorKeyFork/blob/main/packages/corridorkey/src/corridorkey/errors.py)

## Purpose

Typed exceptions carry structured data about what went wrong. Rather than parsing an error message string, the interface can read fields like `clip_name`, `input_count`, or `required_gb` directly and construct a meaningful, actionable message for the user.

## How It Works

Every exception in the hierarchy inherits from `CorridorKeyError`. The interface can catch the base class for a single catch-all handler, or catch specific subclasses to handle different failure modes differently.

The package never raises bare `Exception` or `RuntimeError` for expected failure conditions. If an unexpected exception propagates out of the package, it is a bug.

## Exception Reference

### CorridorKeyError

The base class. Catching this handles any error the package raises. Use it as a fallback after more specific handlers.

### DeviceError

Raised during startup when the requested compute device is unavailable. This happens when the config specifies `"cuda"` on a machine with no NVIDIA GPU, or when a specific device index is out of range. The interface should catch this early, during the startup sequence, and offer the user a way to change the device setting.

### ModelError

Raised when the model checkpoint cannot be downloaded or fails checksum verification. The interface should offer to retry the download.

### ClipScanError

Raised by `scan()` when the provided path does not exist or is a file type the scanner does not recognise. The interface should validate the path before calling `scan()`, but this error is the fallback for cases that slip through.

### ExtractionError

Raised when video frame extraction fails. It carries `clip_name` and a `detail` string describing the failure. The interface should display both and offer the user a way to skip the clip or retry.

### FrameMismatchError

Raised by `load()` or `resolve_alpha()` when the number of alpha frames does not match the number of input frames. It carries `clip_name`, `input_count`, and `alpha_count`. The interface should display all three so the user can diagnose whether the alpha generation produced the wrong number of frames.

### FrameReadError

Raised by `preprocess_frame()` when a frame file cannot be read or decoded. In the context of the internal frame loop, the worker catches this per-frame and fires `on_frame_error` rather than aborting the clip. In the frame loop pattern, the interface receives it directly and decides whether to skip the frame or abort.

### WriteFailureError

Raised by `write_frame()` when `cv2.imwrite` returns a failure. It carries the path that failed. This typically indicates a permissions problem or a full disk.

### VRAMInsufficientError

Raised when there is not enough GPU VRAM for the requested operation. It carries `required_gb` and `available_gb`. The interface should suggest reducing `img_size` in the configuration or switching to a smaller model resolution.

### InvalidStateTransitionError

Raised by `ClipEntry.transition_to()` when the requested state transition is not permitted by the state machine. It carries `clip_name`, `current_state`, and `target_state`. This is a programming error in the interface, not a user-facing condition.

### JobCancelledError

Raised when a pipeline job is cancelled by the caller. It carries `clip_name` and optionally `frame_index`. The interface raises this itself (by cancelling the job) and then catches it to update the UI state.

## Handling Strategy

The recommended approach is to catch specific subclasses first, in order of how actionable they are, and fall back to the base class for anything unexpected.

`DeviceError` and `ModelError` should be caught during startup and resolved before any clip processing begins. Both are recoverable: the user can change the device setting or trigger a model download.

`FrameMismatchError` and `ExtractionError` are clip-level failures. The interface should mark the affected clip as errored, display the details, and continue processing other clips.

`FrameReadError` and `WriteFailureError` are frame-level failures. In `Engine.run()` they are handled internally and reported through the lower-level frame-loop callbacks. In the frame loop, the interface decides whether to skip the frame or abort the clip.

`CorridorKeyError` as a catch-all should log the full exception and display a generic error message. It should never be silently swallowed.

## Related

- [Startup](startup.md) - Where `DeviceError` and `ModelError` are most likely to surface.
- [Engine](runner.md) - How frame-level errors are reported through events.
- [Reference - errors](../reference/errors.md) - Full exception hierarchy reference.
