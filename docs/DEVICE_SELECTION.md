# Device Selection — Design & Changes

## Problem

Device selection was hardcoded as `"cuda" if torch.cuda.is_available() else "cpu"` in 3 separate places in `clip_manager.py`. This caused several issues:

1. **No MPS support** — Apple Silicon Macs were forced to CPU even though PyTorch supports MPS acceleration
2. **No user override** — no CLI flag or env var to force a specific device
3. **CUDA-only calls would crash on non-CUDA devices** — `torch.cuda.empty_cache()` and `torch.cuda.manual_seed_all()` scattered across modules
4. **Unsafe defaults** — helper constructors (`GVMProcessor`, `CorridorKeyEngine`, `load_videomama_model`) defaulted to `device="cuda"`, which would fail on any non-CUDA machine if called without an explicit device arg

## Solution

### `device_utils.py` (new)

Single module, three functions:

| Function | Purpose |
|----------|---------|
| `detect_best_device()` | Auto-detect priority: CUDA > MPS > CPU |
| `resolve_device(requested)` | Resolve from CLI arg > `CORRIDORKEY_DEVICE` env var > auto-detect. Validates explicit requests with clear error messages |
| `clear_device_cache(device)` | Calls `torch.cuda.empty_cache()` only when device is CUDA. No-op otherwise |

`resolve_device()` distinguishes between "PyTorch not built with MPS" vs "MPS not available on this machine" when MPS is explicitly requested.

### `clip_manager.py`

- Added `--device` argparse arg (choices: `auto`, `cuda`, `mps`, `cpu`, default: `auto`)
- Device resolved once at startup, logged, then threaded through all functions
- Removed 3 inline `device = "cuda" if torch.cuda.is_available() else "cpu"` blocks
- Functions now accept optional `device` param: `generate_alphas()`, `run_videomama()`, `run_inference()`, `interactive_wizard()`
- Each function falls back to `resolve_device()` if `device=None` (for standalone use)
- `get_gvm_processor()` and `get_corridor_key_engine()` defaults changed from `"cuda"` to `"cpu"` — callers always pass explicit device now, so the default is a safe fallback rather than a crash

### CUDA-only call guards

| File | Change |
|------|--------|
| `gvm_core/wrapper.py` | `seed_all()`: `torch.cuda.manual_seed_all()` guarded behind `torch.cuda.is_available()` |
| `gvm_core/wrapper.py` | `GVMProcessor.__init__` default: `"cuda"` -> `"cpu"` |
| `gvm_core/gvm/pipelines/pipeline_gvm.py` | `torch.cuda.empty_cache()` guarded behind `rgb.device.type == "cuda"` |
| `VideoMaMaInferenceModule/inference.py` | `torch.cuda.empty_cache()` guarded behind `pipeline.device.type == "cuda"` |
| `VideoMaMaInferenceModule/inference.py` | `load_videomama_model` default: `"cuda"` -> `"cpu"` |
| `CorridorKeyModule/inference_engine.py` | `CorridorKeyEngine.__init__` default: `"cuda"` -> `"cpu"` |

### Test infrastructure

`tests/conftest.py` — the `gpu` marker now skips when **neither** CUDA nor MPS is available (previously CUDA-only).

### Documentation

`README.md` — new "Device Selection" section documenting `--device`, `CORRIDORKEY_DEVICE`, auto-detect behavior, and `PYTORCH_ENABLE_MPS_FALLBACK=1` for Mac users.

## Resolution priority

```
--device flag  >  CORRIDORKEY_DEVICE env var  >  auto-detect (CUDA > MPS > CPU)
```

## Not changed

- `test_vram.py` — standalone CUDA-specific utility, not part of main flow
- `VideoMaMaInferenceModule/pipeline.py` — has its own `self.device` logic that respects the device string passed to it
