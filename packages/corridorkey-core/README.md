# corridorkey-core

<div align="center">
  <a href="https://pypi.org/project/corridorkey-core/">
    <img src="https://img.shields.io/pypi/v/corridorkey-core?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.13%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://nikopueringer.github.io/CorridorKey/api/corridorkey-core/">
    <img src="https://img.shields.io/badge/docs-mkdocs-4dabf7?style=for-the-badge&logo=readthedocs&logoColor=white" alt="Docs">
  </a>
</div>

<div align="center">Core inference engine and compositing utilities for the CorridorKey AI chroma keying model.</div>

## What's in this package

- `CorridorKeyEngine` - loads a GreenFormer checkpoint and runs per-frame alpha matte prediction
- Compositing utilities - color space conversion, alpha compositing, green spill removal, and matte cleanup

## Installation

```bash
uv add corridorkey-core
```

For CUDA support:

```bash
uv add corridorkey-core --extra cuda
```

## Usage

```python
from corridorkey_core import CorridorKeyEngine

engine = CorridorKeyEngine(checkpoint_path="path/to/checkpoint.pt", device="cuda")

result = engine.process_frame(image=frame_rgb, mask_linear=trimap)

alpha = result["alpha"]      # [H, W, 1] linear float
processed = result["processed"]  # [H, W, 4] linear premultiplied RGBA
```

## Architecture

This package is the Core Layer of the CorridorKey architecture. It has no filesystem,
pipeline, or UI dependencies and can be embedded in any workflow.

See the [API documentation](https://nikopueringer.github.io/CorridorKey/api/corridorkey-core/)
for the full reference.

## License

See [LICENSE](https://github.com/nikopueringer/CorridorKey/blob/main/LICENSE).
