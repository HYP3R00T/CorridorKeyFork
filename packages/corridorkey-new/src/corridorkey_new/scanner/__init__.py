"""Scanner stage (stage 0).

Public API::

    from corridorkey_new.scanner import scan, Clip
"""

from corridorkey_new.scanner.contracts import Clip
from corridorkey_new.scanner.orchestrator import scan

__all__ = ["scan", "Clip"]
