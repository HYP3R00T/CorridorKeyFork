# configuration

Configuration models for the pipeline and each stage.

::: corridorkey.CorridorKeyConfig

## Bridge Methods

`CorridorKeyConfig` provides bridge methods that build stage runtime configs from the loaded top-level config. Always use these rather than constructing stage configs manually — they resolve VRAM once and derive all settings consistently.

::: corridorkey.infra.config.pipeline.CorridorKeyConfig.to_pipeline_config

::: corridorkey.infra.config.pipeline.CorridorKeyConfig.to_inference_config

::: corridorkey.infra.config.pipeline.CorridorKeyConfig.to_preprocess_config

::: corridorkey.infra.config.pipeline.CorridorKeyConfig.to_postprocess_config

::: corridorkey.infra.config.pipeline.CorridorKeyConfig.to_writer_config

::: corridorkey.PreprocessSettings

::: corridorkey.InferenceSettings

::: corridorkey.PostprocessSettings

::: corridorkey.WriterSettings

::: corridorkey.LoggingSettings

## Loader Functions

::: corridorkey.load_config

::: corridorkey.load_config_with_metadata

The following functions are re-exported from [`utilityhub_config`](https://pypi.org/project/utilityhub_config/): `write_config`, `ensure_config_file`, `get_config_path`. Refer to that package's documentation for their full signatures.
