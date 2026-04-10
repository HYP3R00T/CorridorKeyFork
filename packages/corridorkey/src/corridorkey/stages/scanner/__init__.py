"""Scanner stage (stage 0).

Public API::

    from corridorkey.stages.scanner import scan, Clip, ScanResult, SkippedClip
"""

from corridorkey.stages.scanner.contracts import Clip, ScanResult, SkippedClip
from corridorkey.stages.scanner.orchestrator import scan

__all__ = ["scan", "Clip", "ScanResult", "SkippedClip"]
