# Pipeline Internals

The `corridorkey` package processes footage through six sequential stages. Each stage has a single orchestrator that owns the step sequence, and delegates all transformation logic to focused sub-modules.

## Stages

| Stage | Name | Entry point | Input | Output |
|---|---|---|---|---|
| 0 | [Scanner](scanner.md) | `scan(path)` | Filesystem path | `ScanResult` containing `Clip` objects |
| 1 | [Loader](loader.md) | `load(clip)` | `Clip` | `ClipManifest` |
| 2 | [Preprocessor](preprocessor.md) | `preprocess_frame(manifest, i, config)` | `ClipManifest` + frame index | `PreprocessedFrame` |
| 3 | [Inference](inference.md) | `run_inference(frame, model, config)` | `PreprocessedFrame` | `InferenceResult` |
| 4 | [Postprocessor](postprocessor.md) | `postprocess_frame(result, config)` | `InferenceResult` | `PostprocessedFrame` |
| 5 | [Writer](writer.md) | `write_frame(frame, config)` | `PostprocessedFrame` | Files on disk |

## Stage Layout Convention

Every stage follows the same internal layout:

```text
stages/<name>/
  __init__.py       # re-exports the public entry point
  orchestrator.py   # step sequence - no transformation logic
  contracts.py      # input/output data types for this stage
  <step>.py         # one module per transformation step
```

The orchestrator owns the order of steps and calls into the step modules. It never implements transformation logic itself.

## Related

- [Clip State Machine](../clip-state.md) - How clip lifecycle states are tracked across stages.
- [Job Queue](../job-queue.md) - How stages are wired together in the pipeline runner.
