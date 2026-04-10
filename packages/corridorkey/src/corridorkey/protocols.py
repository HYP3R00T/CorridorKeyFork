"""CorridorKey extension protocols.

Implement these to extend the pipeline. The Engine checks conformance at
registration time — no inheritance required, just match the method signature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from corridorkey.infra.config.pipeline import CorridorKeyConfig
    from corridorkey.stages.loader.contracts import ClipManifest


@runtime_checkable
class AlphaGenerator(Protocol):
    """Protocol for alpha hint frame generators.

    Fills the ``alpha`` slot in the Engine pipeline. Receives a clip that
    needs alpha hint frames and returns an updated manifest with alpha present.

    The Engine calls ``generate()`` synchronously and waits for it to complete
    before proceeding. If it raises, the Engine fires ``clip_error("alpha", exc)``
    and skips the clip.

    Optional config
    ---------------
    Declare a ``Config`` inner class (Pydantic ``BaseModel``) to receive
    validated settings from the ``[plugins.<name>]`` TOML section::

        class MyAlphaGenerator:
            class Config(BaseModel):
                sensitivity: float = 0.7
                model_path: Path = Path("~/models/roto.pth")

            def generate(self, manifest: ClipManifest, config: CorridorKeyConfig) -> ClipManifest: ...

    The Engine wires ``self.config`` from the TOML section at registration time.
    If no section exists, schema defaults are used.

    Contract
    --------
    - ``manifest.needs_alpha`` is guaranteed ``True`` on entry.
    - Must return a manifest with ``needs_alpha=False`` and ``alpha_frames_dir`` set.
    - If the returned manifest still has ``needs_alpha=True``, the Engine raises
      ``AlphaGeneratorError`` immediately.
    """

    def generate(self, manifest: ClipManifest, config: CorridorKeyConfig) -> ClipManifest:
        """Generate alpha hint frames for the given clip.

        Args:
            manifest: ClipManifest with needs_alpha=True.
            config: Full CorridorKeyConfig — use what you need.

        Returns:
            Updated ClipManifest with needs_alpha=False and alpha_frames_dir set.
        """
        ...
