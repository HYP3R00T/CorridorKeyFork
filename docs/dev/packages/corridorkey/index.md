# corridorkey

The `corridorkey` package is the pipeline library. It exposes the full processing pipeline as a set of composable functions and types that any interface (CLI, GUI, TUI, web) can orchestrate.

## Purpose

This package owns the pipeline logic, configuration, runtime, and all stage implementations. It has no dependency on any interface layer. Interfaces import from `corridorkey` - `corridorkey` never imports from them.

## Package Layout

```text
corridorkey/
  __init__.py          # single public import surface
  errors.py            # typed exception hierarchy
  events.py            # PipelineEvents callbacks
  infra/               # configuration, device, logging
    config/            # CorridorKeyConfig and per-stage settings
    device_utils.py    # GPU detection and device resolution
    logging.py         # file logging setup
    model_hub.py       # model download and checksum
  runtime/             # pipeline orchestration
    clip_state.py      # ClipEntry state machine
    queue.py           # BoundedQueue with sentinel shutdown
    runner.py          # PipelineRunner and MultiGPURunner
    worker.py          # PreprocessWorker, InferenceWorker, PostWriteWorker
  stages/              # one folder per pipeline stage
    scanner/           # scan() - clip discovery
    loader/            # load(), resolve_alpha() - clip loading
    preprocessor/      # preprocess_frame() - tensor preparation
    inference/         # load_model(), run_inference()
    postprocessor/     # postprocess_frame()
    writer/            # write_frame()
```

## Public API

All public symbols are exported from `corridorkey.__init__`. Do not import from submodules directly.

```python
from corridorkey import (
    scan, load, resolve_alpha, preprocess_frame,
    load_model, run_inference, postprocess_frame, write_frame,
    load_config, setup_logging, resolve_device, detect_gpu,
    ClipState, ClipEntry, PipelineRunner,
)
```

See the [API Reference](../../api/corridorkey/index.md) for the full symbol list.

## Documents in This Section

- [Clip State Machine](clip-state.md) - How clip lifecycle states are tracked and transitioned.
- [Job Queue](job-queue.md) - Bounded queue and sentinel-based shutdown used between pipeline workers.
- [Configuration](configuration.md) - All configuration fields, defaults, and sources.
- [Stages](stages/index.md) - Step-by-step breakdown of all six pipeline stages.
