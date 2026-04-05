# corridorkey Reference

Auto-generated symbol reference for all public exports of the `corridorkey` package.

## Modules

- [pipeline](pipeline.md) - Pipeline stage functions, `Runner`, `PipelineRunner`, `MultiGPURunner`, backend protocol, and all stage contracts.
- [clip-state](clip-state.md) - `ClipState`, `ClipEntry`, `InOutRange`.
- [job-queue](job-queue.md) - `BoundedQueue` and the `STOP` sentinel.
- [errors](errors.md) - Typed exception hierarchy.
- [events](events.md) - `PipelineEvents` callbacks.
- [device-utils](device-utils.md) - `detect_gpu`, `resolve_device`, `resolve_devices`, `clear_device_cache`, `GPUInfo`.
- [model-hub](model-hub.md) - `ensure_model`, `default_checkpoint_path`, download constants.
- [configuration](configuration.md) - `CorridorKeyConfig`, bridge methods (`to_pipeline_config`, `to_inference_config`, etc.), and all per-stage settings models.
