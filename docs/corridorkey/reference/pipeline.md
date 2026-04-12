# pipeline

Pipeline stage functions, Engine, JobStats, backend protocol, and all stage contracts exported from `corridorkey`.

## Stage Functions

::: corridorkey.scan

::: corridorkey.load

::: corridorkey.attach_alpha

::: corridorkey.preprocess_frame

::: corridorkey.load_model_backend

::: corridorkey.postprocess_frame

::: corridorkey.write_frame

## Engine

::: corridorkey.Engine

::: corridorkey.JobStats

## Backend Protocol

::: corridorkey.stages.inference.backend.ModelBackend

## Stage Contracts

### Scanner

::: corridorkey.Clip

::: corridorkey.ScanResult

::: corridorkey.SkippedClip

### Loader

::: corridorkey.ClipManifest

::: corridorkey.list_frames

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

::: corridorkey.ProcessedFrame

### Writer

::: corridorkey.WriteConfig
