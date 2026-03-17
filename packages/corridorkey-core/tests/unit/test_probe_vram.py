"""Unit tests for _probe_vram_gb in pipeline/engine.py.

Tests all three resolution paths: pynvml success, pynvml unavailable -> torch
fallback, and both unavailable -> 0.0. No GPU required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from corridorkey_core.engine import _probe_vram_gb


class TestProbeVramGb:
    """_probe_vram_gb - VRAM detection with pynvml and torch fallbacks."""

    def test_pynvml_success(self):
        """When pynvml is available and succeeds, must return total VRAM in GB."""
        mock_pynvml = MagicMock()
        mock_mem = MagicMock()
        mock_mem.total = 8 * 1024**3  # 8 GB
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_mem

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _probe_vram_gb()

        assert abs(result - 8.0) < 0.01

    def test_pynvml_failure_falls_back_to_torch(self):
        """When pynvml raises, must fall back to torch.cuda."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("pynvml unavailable")

        mock_props = MagicMock()
        mock_props.total_memory = 12 * 1024**3  # 12 GB

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", return_value=mock_props),
        ):
            result = _probe_vram_gb()

        assert abs(result - 12.0) < 0.01

    def test_both_unavailable_returns_zero(self):
        """When both pynvml and torch.cuda fail, must return 0.0."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("no pynvml")

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch("torch.cuda.is_available", return_value=False),
        ):
            result = _probe_vram_gb()

        assert result == 0.0

    def test_torch_cuda_raises_returns_zero(self):
        """When torch.cuda.get_device_properties raises, must return 0.0."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("no pynvml")

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_properties", side_effect=RuntimeError("no device")),
        ):
            result = _probe_vram_gb()

        assert result == 0.0

    def test_returns_float(self):
        """Return value must always be a float."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlInit.side_effect = Exception("no pynvml")

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch("torch.cuda.is_available", return_value=False),
        ):
            result = _probe_vram_gb()

        assert isinstance(result, float)
