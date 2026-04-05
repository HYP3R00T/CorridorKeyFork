# pipeline

Pipeline stage functions, runner classes, backend protocol, and all stage contracts exported from `corridorkey`.

## Stage Functions

::: corridorkey.scan

::: corridorkey.load

::: corridorkey.resolve_alpha

::: corridorkey.load_video_metadata

::: corridorkey.preprocess_frame

::: corridorkey.load_backend

::: corridorkey.load_model

::: corridorkey.run_inference

::: corridorkey.postprocess_frame

::: corridorkey.write_frame

## Runner Classes

::: corridorkey.runtime.runner.Runner

::: corridorkey.runtime.runner.PipelineConfig

## Backend Protocol

::: corridorkey.stages.inference.backend.ModelBackend

::: corridorkey.stages.inference.backend.TorchBackend

::: corridorkey.stages.inference.factory.discover_checkpoint

## Stage Contracts

### Scanner

::: corridorkey.Clip

::: corridorkey.ScanResult

::: corridorkey.SkippedPath

### Loader

::: corridorkey.ClipManifest

::: corridorkey.VideoMetadata

::: corridorkey.stages.loader.validator.FrameScan

::: corridorkey.stages.loader.validator.scan_frames

::: corridorkey.stages.loader.validator.get_frame_files

### Preprocessor

::: corridorkey.PreprocessConfig

::: corridorkey.PreprocessedFrame

::: corridorkey.FrameMeta

### Inference

::: corridorkey.InferenceConfig

::: corridorkey.InferenceResult

`BackendChoice` - Type alias for the backend field: `Literal["auto", "torch", "mlx"]`.

`RefinerMode` - Type alias for the refiner_mode field: `Literal["auto", "full_frame", "tiled"]`.

`VALID_IMG_SIZES` - Tuple of accepted img_size values: `(0, 512, 1024, 1536, 2048)`. `0` means auto-select based on VRAM.

::: corridorkey.stages.inference.config.adaptive_img_size

### Postprocessor

::: corridorkey.PostprocessConfig

::: corridorkey.PostprocessedFrame

### Writer

::: corridorkey.WriteConfig
