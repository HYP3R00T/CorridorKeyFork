"""Unit tests for corridorkey.infra.device_utils."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from corridorkey.errors import DeviceError
from corridorkey.infra.device_utils import resolve_device, resolve_devices


class TestResolveDeviceCpu:
    def test_cpu_returns_cpu(self):
        assert resolve_device("cpu") == "cpu"

    def test_cpu_case_insensitive(self):
        assert resolve_device("CPU") == "cpu"


class TestResolveDeviceCuda:
    def test_cuda_bare_returns_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_device("cuda") == "cuda"

    def test_cuda_index_zero(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            assert resolve_device("cuda:0") == "cuda:0"

    def test_cuda_index_one(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            assert resolve_device("cuda:1") == "cuda:1"

    def test_cuda_index_out_of_range_raises(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
            pytest.raises(DeviceError, match="out of range"),
        ):
            resolve_device("cuda:5")

    def test_cuda_unavailable_raises(self):
        with patch("torch.cuda.is_available", return_value=False), pytest.raises(DeviceError):
            resolve_device("cuda")

    def test_cuda_invalid_index_raises(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            pytest.raises(DeviceError, match="Invalid device index"),
        ):
            resolve_device("cuda:abc")

    def test_rocm_bare_returns_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_device("rocm") == "cuda"

    def test_rocm_index(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=2),
        ):
            assert resolve_device("rocm:1") == "cuda:1"


class TestResolveDeviceUnknown:
    def test_unknown_device_raises(self):
        with pytest.raises(DeviceError, match="Unknown device"):
            resolve_device("tpu")

    def test_none_triggers_auto(self):
        # auto-detect falls back to CPU when no GPU is available
        with (
            patch("torch.version.hip", None),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            result = resolve_device(None)
            assert result == "cpu"

    def test_auto_triggers_auto(self):
        with (
            patch("torch.version.hip", None),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.backends.mps.is_available", return_value=False),
        ):
            result = resolve_device("auto")
            assert result == "cpu"


class TestResolveDevices:
    def test_single_cpu(self):
        assert resolve_devices("cpu") == ["cpu"]

    def test_single_cuda(self):
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_devices("cuda") == ["cuda"]

    def test_list_of_devices(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=4),
        ):
            result = resolve_devices(["cuda:0", "cuda:1"])
            assert result == ["cuda:0", "cuda:1"]

    def test_all_expands_to_all_gpus(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=3),
        ):
            result = resolve_devices("all")
            assert result == ["cuda:0", "cuda:1", "cuda:2"]

    def test_all_no_cuda_raises(self):
        with (
            patch("torch.cuda.is_available", return_value=False),
            pytest.raises(DeviceError, match="no CUDA devices"),
        ):
            resolve_devices("all")

    def test_all_zero_devices_raises(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=0),
            pytest.raises(DeviceError, match="device_count.*0"),
        ):
            resolve_devices("all")
