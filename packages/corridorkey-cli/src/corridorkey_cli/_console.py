"""Shared Rich console instances."""

from __future__ import annotations

from rich.console import Console

console = Console()
err_console = Console(stderr=True)
