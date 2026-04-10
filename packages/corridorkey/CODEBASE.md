# Codebase Map

Quick reference for every folder and file in `src/corridorkey/`.

## Top level

- `__init__.py` — Single public import surface. Everything a wrapper (GUI, CLI, plugin) needs comes from here. Documents both integration paths with examples.
- `errors.py` — All 10 typed exceptions, all inheriting `CorridorKeyError`. Carry structured fields (`clip_name`, `required_gb`, etc.) so interfaces can build meaningful messages without parsing strings.
- `events.py` — `PipelineEvents` dataclass with 13 optional callbacks covering frame progress, stage lifecycle, clip completion, scan events, and errors. Zero UI dependencies; callbacks fire on worker threads.
- `py.typed` — PEP 561 marker. Tells type checkers this package ships inline types.

## `infra/` — startup concerns

Everything called once at application startup before any clip is processed.

- `__init__.py` — Re-exports the infra public API: config loading, device utils, logging, model hub.
- `device_utils.py` — `detect_gpu()` probes hardware (CUDA/ROCm/MPS/CPU). `resolve_device()` validates a device string. `resolve_devices()` expands `"all"` to every CUDA GPU. `clear_device_cache()` frees VRAM between clips.
- `model_hub.py` — `ensure_model()` downloads the checkpoint if absent, verifies SHA-256, and moves it into place atomically. `default_checkpoint_path()` returns the expected local path without downloading.
- `logging.py` — `setup_logging()` attaches a timestamped file handler. The interface adds its own console or GUI handler on top.
- `colorspace.py` — Shared sRGB/linear LUT used by both the preprocessor and postprocessor. Centralised here so both stages use identical transfer functions.
- `utils.py` — `natural_sort_key()` sorts `frame_10.png` after `frame_9.png`, not before. Used by every frame directory scan.

### `infra/config/` — one file per stage settings model

- `pipeline.py` — `CorridorKeyConfig`, the top-level Pydantic model. Nests all stage settings and provides bridge methods (`to_pipeline_config`, `to_inference_config`, `to_preprocess_config`, `to_postprocess_config`, `to_writer_config`) that build stage runtime configs consistently, probing VRAM at most once.
- `_loader.py` — `load_config()` and `load_config_with_metadata()`. Resolves config from defaults → global file → project file → env vars → overrides.
- `inference.py` — `InferenceSettings`: checkpoint path, precision, refiner mode, backend choice.
- `preprocess.py` — `PreprocessSettings`: img_size, upsample mode, half precision, source passthrough.
- `postprocess.py` — `PostprocessSettings`: despill, despeckle, upsample modes, hint sharpen, debug dump.
- `writer.py` — `WriterSettings`: which outputs to write (alpha, fg, processed, comp) and in what format.
- `logging.py` — `LoggingSettings`: log level and output directory.

## `runtime/` — stateful coordination

Manages threading, queuing, and clip lifecycle. Sits above the stateless stages.

- `runner.py` — `Runner` processes one clip. Spawns one `PreprocessWorker`, N `InferenceWorker`s (one per device), and one `PostWriteWorker`, then blocks until all frames are written. `cancel()` signals all workers to stop cleanly and raises `JobCancelledError`. Also contains `PipelineConfig`, `_AtomicCounter` (coordinates multi-GPU shutdown), and `_InferenceWorker`.
- `worker.py` — `PreprocessWorker` reads frames and pushes tensors onto the input queue. `PostWriteWorker` pulls inference results, postprocesses, and writes to disk. Both accept a `cancel_event`.
- `queue.py` — `BoundedQueue`, a thread-safe FIFO with fixed capacity for backpressure. Shutdown uses a `STOP` sentinel that propagates downstream automatically. `put_unless_cancelled()` unblocks on cancellation.
- `clip_state.py` — `ClipEntry`, a mutable wrapper around a `Clip` for session management. Tracks `ClipState` (EXTRACTING → RAW → READY → COMPLETE / ERROR), holds the manifest and optional `InOutRange`, exposes `completed_frame_count()` and `has_outputs`. `resolve_clip_state()` derives state from what is on disk.

## `stages/` — stateless transformations

Six stages, each in its own sub-package. Every stage follows the same internal layout: `contracts.py` for data types, `orchestrator.py` for the entry point, and optional helpers for each concern.

### `stages/scanner/` — discovers clips on disk

- `orchestrator.py` — `scan(path)` walks a directory, identifies valid clip folders and video files, returns a `ScanResult`.
- `normaliser.py` — Normalises raw video files into the expected `Input/` folder structure before scanning.
- `contracts.py` — `Clip`, `ScanResult`, `SkippedPath`: the output types of this stage.

### `stages/loader/` — validates a clip and extracts video frames if needed

- `orchestrator.py` — `load(clip)` validates frame counts, extracts video to `Frames/` if needed, and returns a `ClipManifest`. `resolve_alpha(manifest, dir)` updates the manifest after external alpha generation.
- `extractor.py` — Calls ffmpeg to extract video frames to PNG. Reads and writes a `video_meta.json` sidecar for re-encoding.
- `validator.py` — `scan_frames()`, `list_clip_frames()`, `count_frames()`, `validate()`: all frame directory scanning in one place.
- `contracts.py` — `ClipManifest`: the output of this stage, consumed by all downstream stages.

### `stages/preprocessor/` — reads one frame pair, converts it to a model-ready tensor on device

- `orchestrator.py` — `preprocess_frame()` entry point. Runs all 9 steps in order. Also defines `PreprocessConfig`.
- `reader.py` — Reads image and alpha files with `cv2.imread`, normalises to float32.
- `tensor.py` — Concatenates image + alpha into `[4, H, W]`, transfers to device in one DMA call, reorders BGR to RGB on-device.
- `colorspace.py` — Linear-to-sRGB conversion for EXR inputs, runs on-device.
- `resize.py` — Resizes to `img_size x img_size` (square stretch, no padding).
- `normalise.py` — ImageNet mean/std normalisation in-place. Caches mean/std tensors per `(dtype, device)`.
- `contracts.py` — `PreprocessedFrame` and `FrameMeta`: the output types.

### `stages/inference/` — runs the neural network forward pass

- `factory.py` — `load_backend()` resolves which backend to use (torch or MLX), loads the model, and returns a `ModelBackend`.
- `backend.py` — `ModelBackend` protocol. `TorchBackend` wraps the PyTorch model. `MLXBackend` wraps the optional `corridorkey-mlx` package. Both expose `.run(frame) -> InferenceResult`.
- `orchestrator.py` — `run_inference()` runs the model forward pass under autocast, handles tiled vs full-frame refiner dispatch.
- `loader.py` — `load_model()` loads checkpoint weights onto device, optionally compiles with `torch.compile`.
- `model.py` — GreenFormer model architecture (transformer backbone + CNN refiner).
- `config.py` — `InferenceConfig` runtime config. Also `adaptive_img_size()` for VRAM-based auto-selection.
- `contracts.py` — `InferenceResult`: pass-through output type carrying on-device tensors at model resolution.

### `stages/postprocessor/` — resizes predictions back to source resolution and applies quality corrections

- `orchestrator.py` — `postprocess_frame()` entry point. Runs all postprocessing steps in order, returns a `PostprocessedFrame`.
- `resize.py` — Resizes alpha and fg from model resolution back to source resolution using configurable interpolation.
- `despill.py` — Green spill suppression: reduces green contamination in the foreground colour.
- `despeckle.py` — Removes small disconnected alpha islands via connected-component analysis.
- `hint_sharpen.py` — Applies a hard binary mask derived from the alpha hint to eliminate soft edge tails from upscaling.
- `composite.py` — Builds the premultiplied RGBA output and the checkerboard preview composite. Handles source passthrough (replaces model FG in opaque interior regions with original source pixels).
- `config.py` — `PostprocessConfig`: all postprocessing knobs.
- `contracts.py` — `PostprocessedFrame`: numpy arrays at source resolution, ready to write or inspect directly.

### `stages/writer/` — writes output files to disk

- `orchestrator.py` — `write_frame()` writes whichever outputs are enabled (alpha, fg, processed, comp) in the configured format (PNG or EXR).
- `contracts.py` — `WriteConfig`: output directory, per-output enable flags, format choices, EXR compression codec.
