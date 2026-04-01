# Interface Guide

The `corridorkey` package is a library. It has no user interface of its own. Every interface - a CLI, a desktop GUI, a DaVinci Resolve Fusion node, an Adobe Premiere Pro extension - is a separate layer that calls into the package.

This section explains what that boundary means, what each side owns, and how to build on top of the package correctly.

## The Boundary

The package owns all pipeline logic: reading frames, running the model, postprocessing, and writing outputs. It knows nothing about terminals, windows, panels, or host application APIs.

The interface owns everything the user sees and interacts with: presenting scan results, showing progress, accepting configuration input, and reporting errors. It knows nothing about tensors, VRAM, or frame sequences.

This boundary is enforced by the import direction. The interface imports from `corridorkey`. The package never imports from the interface.

## What Every Interface Must Do

Regardless of the integration pattern chosen, every interface has the same set of responsibilities before processing can begin.

1. Load and validate configuration from the user's config file and environment.
2. Set up file logging for the session.
3. Resolve and validate the compute device.
4. Detect the available GPU and present that information to the user.
5. Ensure the model checkpoint is present, downloading it if needed.
6. Scan the clips directory and present the results, including any skipped paths.
7. Handle the case where a clip has no alpha hint, by generating alpha externally and then informing the package.
8. Choose an integration pattern and run the pipeline.
9. Catch typed errors from the package and present them meaningfully.

The [Startup](startup.md) document covers steps 1 through 5. The integration pattern documents cover the rest.

## Integration Patterns

Three patterns are available. The right choice depends on how much control the interface needs over individual frames and whether it manages its own threading.

| Pattern | Document | Best for |
|---|---|---|
| PipelineRunner | [pipeline-runner.md](pipeline-runner.md) | CLI, GUI, batch tools, most plugins |
| MultiGPURunner | [multi-gpu-runner.md](multi-gpu-runner.md) | Workstations with multiple CUDA GPUs |
| Frame loop | [frame-loop.md](frame-loop.md) | Host-managed threading (Fusion node, Premiere extension) |

## Cross-Cutting Concerns

Two concerns apply regardless of which pattern is used.

- [Events](events.md) - How to receive per-frame and per-stage progress notifications.
- [Error handling](error-handling.md) - The typed error hierarchy and how to recover from each class of failure.

## Related

- [Clip State Machine](../clip-state.md) - Tracking clip lifecycle across sessions.
- [Configuration](../configuration.md) - All configuration fields and their defaults.
- [API Reference](../../../api/corridorkey/index.md) - Full symbol reference.
