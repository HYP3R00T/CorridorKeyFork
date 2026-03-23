"""Package-level conftest — marker auto-skip.

Options are declared in the root conftest. This file only adds the skip
markers so tests run correctly when pytest is invoked from this directory
directly (without the root conftest in scope).
"""

from __future__ import annotations

import contextlib

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    # Guard against duplicate registration when the root conftest is also loaded
    # (e.g. when `mise run test-cov` runs all packages together).
    for opt, help_text in (
        ("--run-slow", "Run slow tests"),
        ("--run-gpu", "Run GPU tests"),
        ("--run-mlx", "Run MLX tests"),
    ):
        with contextlib.suppress(ValueError):
            parser.addoption(opt, action="store_true", default=False, help=help_text)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(pytest.mark.skip(reason="Pass --run-slow to run"))
        if "gpu" in item.keywords and not config.getoption("--run-gpu"):
            item.add_marker(pytest.mark.skip(reason="Pass --run-gpu to run"))
        if "mlx" in item.keywords and not config.getoption("--run-mlx"):
            item.add_marker(pytest.mark.skip(reason="Pass --run-mlx to run"))
