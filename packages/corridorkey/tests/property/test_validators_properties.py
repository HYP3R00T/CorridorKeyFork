"""Property-based tests for validators.py.

normalize_mask_dtype must produce float32 output in [0, 1] for any uint8
or uint16 input, and must return float32 arrays unchanged. Hypothesis
generates random arrays of each dtype to verify these contracts hold across
the full value range, not just the boundary values tested in unit tests.
"""

from __future__ import annotations

import numpy as np
from corridorkey.validators import normalize_mask_dtype
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


@given(arrays(dtype=np.uint8, shape=st.tuples(st.integers(1, 8), st.integers(1, 8))))
def test_uint8_output_in_unit_range(mask: np.ndarray) -> None:
    """uint8 masks must normalise to [0.0, 1.0]."""
    result = normalize_mask_dtype(mask)
    assert result.dtype == np.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


@given(arrays(dtype=np.uint16, shape=st.tuples(st.integers(1, 8), st.integers(1, 8))))
def test_uint16_output_in_unit_range(mask: np.ndarray) -> None:
    """uint16 masks must normalise to [0.0, 1.0]."""
    result = normalize_mask_dtype(mask)
    assert result.dtype == np.float32
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0


@given(
    arrays(
        dtype=np.float32,
        shape=st.tuples(st.integers(1, 8), st.integers(1, 8)),
        elements=st.floats(0.0, 1.0, allow_nan=False),
    )
)
def test_float32_passthrough_identity(mask: np.ndarray) -> None:
    """float32 masks must be returned as-is (same object)."""
    result = normalize_mask_dtype(mask)
    assert result is mask
