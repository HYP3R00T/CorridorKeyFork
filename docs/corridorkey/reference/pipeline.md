# pipeline

Pipeline stage functions, runner classes, backend protocol, and all stage contracts exported from `corridorkey`.

## Stage Functions

::: corridorkey.scan

::: corridorkey.load

::: corridorkey.resolve_alpha

::: corridorkey.preprocess_frame

::: corridorkey.load_backend

::: corridorkey.postprocess_frame

::: corridorkey.write_frame

## Runner Classes

::: corridorkey.runtime.runner.Runner

::: corridorkey.runtime.runner.PipelineConfig

## Backend Protocol

::: corridorkey.stages.inference.backend.ModelBackend

## Stage Contracts

### Scanner

::: corridorkey.Clip

::: corridorkey.ScanResult

::: corridorkey.SkippedPath

### Loader

::: corridorkey.ClipManifest

::: corridorkey.list_clip_frames

### Preprocessor

::: corridorkey.PreprocessConfig

::: corridorkey.PreprocessedFrame

::: corridorkey.FrameMeta

### Inference

::: corridorkey.InferenceConfig

::: corridorkey.InferenceResult

`BackendChoice` - Type alias for the backend field: `Literal["auto", "torch", "mlx"]`.

`RefinerMode` - Type alias for the refiner_mode field: `Literal["auto", "full_frame", "tiled"]`.

### Postprocessor

::: corridorkey.PostprocessConfig

::: corridorkey.PostprocessedFrame

### Writer

::: corridorkey.WriteConfig
