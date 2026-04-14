"""Model cache — reuse loaded GPU models across Engine.run() calls.

When the inference configuration hasn't changed between runs, the loaded
model is reused instead of reloaded. This eliminates the 10–60 second
model load cost for long-running hosts (GUI, plugin, service).

Cache key: SHA-256 of (device, img_size, model_precision, refiner_mode,
flash_attention, use_refiner). If any of these change, the old model is
unloaded and a new one is loaded.

Usage::

    cache = ModelCache()
    model, refiner_mode = cache.get(config, resolved_refiner_mode)
    # ... use model ...
    # On process exit or explicit reset:
    cache.clear()
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import threading
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from corridorkey.stages.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


def _config_hash(config: InferenceConfig, resolved_refiner_mode: str | None) -> str:
    """Compute a stable hash of the inference config fields that affect the model."""
    parts = {
        "device": config.device,
        "img_size": config.img_size,
        "model_precision": str(config.model_precision),
        "use_refiner": config.use_refiner,
        "flash_attention": config.flash_attention,
        "refiner_mode": resolved_refiner_mode or config.refiner_mode,
        "checkpoint": str(config.checkpoint_path),
    }
    raw = json.dumps(parts, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class ModelCache:
    """Thread-safe cache for a single loaded inference model.

    Only one model is kept in memory at a time. If the config changes,
    the old model is properly unloaded before the new one is loaded.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: nn.Module | None = None
        self._hash: str = ""
        self._resolved_refiner_mode: str = ""

    def get(
        self,
        config: InferenceConfig,
        resolved_refiner_mode: str | None,
    ) -> tuple[nn.Module, str]:
        """Return a loaded model, reusing the cached one if config matches.

        Args:
            config: InferenceConfig for the current run.
            resolved_refiner_mode: The concrete refiner mode after "auto"
                resolution.

        Returns:
            (model, resolved_refiner_mode) — model is in eval mode on device.
        """
        from corridorkey.stages.inference.loader import load_model

        key = _config_hash(config, resolved_refiner_mode)
        effective_mode = resolved_refiner_mode or config.refiner_mode

        with self._lock:
            if self._model is not None and self._hash == key:
                logger.debug("model_cache: reusing cached model (hash=%s)", key)
                return self._model, self._resolved_refiner_mode

            # Config changed or first load — unload old model first.
            if self._model is not None:
                logger.info("model_cache: config changed, unloading previous model")
                self._unload()

            logger.info("model_cache: loading model (hash=%s, device=%s)", key, config.device)
            model = load_model(config, resolved_refiner_mode=resolved_refiner_mode)

            self._model = model
            self._hash = key
            self._resolved_refiner_mode = effective_mode
            return model, effective_mode

    def clear(self) -> None:
        """Unload the cached model and free GPU memory."""
        with self._lock:
            if self._model is not None:
                self._unload()

    def _unload(self) -> None:
        """Unload the model. Caller must hold _lock."""
        model = self._model
        self._model = None
        self._hash = ""
        self._resolved_refiner_mode = ""

        if model is None:
            return

        # Move weights to CPU before dropping the reference so the CUDA
        # allocator can reclaim the memory immediately.
        try:
            raw = getattr(model, "_orig_mod", model)  # unwrap torch.compile
            raw.cpu()
        except Exception as e:
            logger.debug("model_cache: cpu() failed during unload — %s", e)

        # Reset torch.compile / dynamo state so stale compiled kernels
        # referencing freed CUDA addresses are not reused.
        try:
            import torch._dynamo

            torch._dynamo.reset()
        except Exception:
            pass

        del model
        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass

        logger.info("model_cache: model unloaded, VRAM released")


# Module-level singleton — shared across all Engine instances in the process.
_default_cache = ModelCache()


def get_default_cache() -> ModelCache:
    """Return the process-level model cache singleton."""
    return _default_cache
