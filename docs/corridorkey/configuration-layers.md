# Configuration Layers

The pipeline uses two distinct configuration layers: user-facing settings and
runtime configs. Understanding the difference matters depending on what you are
building.

## The Two Layers

**Settings** (`InferenceSettings`, `PreprocessSettings`, etc.) are what end
users touch. They live in `corridorkey.toml`, can be overridden by environment
variables, and use human-readable types like `"auto"`, `"float16"`, `"bicubic"`.
Every field has a description written for a non-developer audience.

**Runtime configs** (`InferenceConfig`, `PreprocessConfig`, etc.) are what the
pipeline workers consume. They use resolved, concrete types -- `torch.dtype`
instead of `"float16"`, actual `Path` objects, concrete integers instead of
`0`/auto. They are never loaded from a file. They are constructed by
`CorridorKeyConfig` bridge methods that translate settings into pipeline-ready
values.

```
corridorkey.toml
    -> CorridorKeyConfig  (holds Settings objects, validates on load)
        -> to_pipeline_config()
        -> to_inference_config()     (resolves "auto", probes VRAM)
        -> to_preprocess_config()
        -> to_postprocess_config()
        -> to_writer_config()
            -> PipelineConfig / InferenceConfig / ...  <- workers consume these
```

`CorridorKeyConfig` is the bridge. It owns the settings and its `to_*` methods
do the translation -- resolving `"auto"` to a concrete value, probing VRAM,
converting string dtypes to `torch.dtype`, and so on.

## If You Are Configuring for End Users

You only need to work with `CorridorKeyConfig` and the Settings classes. Load
the config with `load_config()`, present the fields to the user, and pass the
result to `to_pipeline_config()`.

???+ tip "What to expose"
    Expose the Settings fields -- `device`, `img_size`, `refiner_mode`,
    `model_precision`, and so on. These are designed to be user-facing. Do not
    expose runtime config fields directly; they contain resolved internal values
    that are not meaningful to end users.

```python
from corridorkey.infra import load_config

config = load_config()          # reads corridorkey.toml + env vars
pipeline_config = config.to_pipeline_config(device=resolved_device)
Runner(manifest, pipeline_config).run()
```

## If You Are Building an Interface

You work with both layers. The Settings layer gives you the user's intent; the
runtime config layer is what you pass to the pipeline.

???+ note "Key point for interface builders"
    Never construct `InferenceConfig`, `PreprocessConfig`, or `PipelineConfig`
    manually from scratch. Always go through `CorridorKeyConfig.to_*_config()`.
    Those methods handle VRAM probing, dtype resolution, and consistency between
    stages. Constructing runtime configs directly bypasses all of that.

The typical interface flow is:

1. Load `CorridorKeyConfig` with `load_config()`.
2. Let the user adjust settings (device, quality preset, etc.).
3. Call `config.to_pipeline_config()` once to get a `PipelineConfig`.
4. Reuse that `PipelineConfig` across all clips in the session -- it holds the
   pre-loaded model and resolved values so they are not recomputed per clip.
5. For each clip, create a new `Runner(manifest, pipeline_config)` and call
   `run()`.

???+ warning "Do not reload the model per clip"
    `to_pipeline_config()` accepts an optional `model` argument. Load the model
    once before your clip loop and pass it in. If you omit it, the runner loads
    from disk on every clip, which adds several seconds of startup time per clip.

## Why the Split Exists

Settings need to survive serialisation -- they go into TOML files and
environment variables, so they must use plain Python types. Runtime configs need
to be efficient and type-safe for the workers, so they use resolved concrete
types. Keeping them separate means neither layer has to compromise.

It also means the bridge (`to_*_config()`) is the single place where
resolution logic lives. VRAM is probed once, `"auto"` is resolved once, and the
result flows consistently into every stage config.

## Related

- [Configuration](configuration.md) - All user-facing settings fields and accepted values.
- [Runner](integration/runner.md) - How `PipelineConfig` is used at runtime.
- [Reference - configuration](reference/configuration.md) - Full symbol reference.
