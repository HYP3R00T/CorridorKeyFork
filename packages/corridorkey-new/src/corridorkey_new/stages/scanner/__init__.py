"""Scanner stage (stage 0).

Public API::

    from corridorkey_new.stages.scanner import scan, Clip
"""

from corridorkey_new.stages.scanner.contracts import Clip
from corridorkey_new.stages.scanner.orchestrator import scan

__all__ = ["scan", "Clip"]
