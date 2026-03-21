"""Writer stage — public surface.

Entry point:
    write_frame(frame, config)

Contracts:
    WriteConfig — output_dir, enabled flags, formats, EXR compression
"""

from corridorkey_new.writer.contracts import WriteConfig
from corridorkey_new.writer.orchestrator import write_frame

__all__ = [
    "write_frame",
    "WriteConfig",
]
