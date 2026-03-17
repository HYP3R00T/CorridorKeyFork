"""CorridorKey core library - pure compute, no filesystem dependencies.

Public API
-----
    create_engine(checkpoint_dir, ...) -> engine

The engine exposes a single method:
    engine.process_frame(image, mask_linear, ...) -> dict

That's the complete public contract. Stages 3, 4, and 5 run inside the engine.
Internal modules (engine, compositing, model_transformer, contracts, stages) are
implementation details and are not part of the public API.
"""

from corridorkey_core.engine_factory import create_engine

__all__ = ["create_engine"]
