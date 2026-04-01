# corridorkey

`corridorkey` is the pipeline library. It exposes the full AI green screen keying pipeline as a set of composable functions and types that any interface can build on top of.

This section is for integrators: anyone building a CLI, GUI, DaVinci Resolve Fusion node, Adobe Premiere extension, or any other interface that calls into the package.

## Documents in This Section

- [Integration Guide](integration/index.md) - How to build an interface on top of the package. Covers startup, all three integration patterns, events, and error handling.
- [Pipeline Internals](pipeline/index.md) - How each of the six pipeline stages works internally.
- [Clip State Machine](clip-state.md) - How clip lifecycle states are tracked and transitioned.
- [Job Queue](job-queue.md) - Bounded queue and sentinel-based shutdown used between pipeline workers.
- [Configuration](configuration.md) - All configuration fields, defaults, and sources.
- [Reference](reference/index.md) - Auto-generated symbol reference for all public exports.
