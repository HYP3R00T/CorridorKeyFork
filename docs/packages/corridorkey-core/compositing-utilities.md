# Compositing Utilities

`corridorkey_core.compositing` provides pure math functions for color space conversion, alpha compositing, green spill removal, and matte cleanup. All functions are used internally by `process_frame` but are also importable for custom pipelines.

All functions accept both NumPy arrays and PyTorch tensors unless noted otherwise. `clean_matte` is NumPy only.

For full parameter documentation and usage, see the [compositing reference](../../api/corridorkey-core/compositing.md).

## Related

- [compositing reference](../../api/corridorkey-core/compositing.md)
- [Output contract](output-contract.md)
