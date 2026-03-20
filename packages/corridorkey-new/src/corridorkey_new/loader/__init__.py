"""Stage 1 — loader.

Validates a Clip from stage 0, extracts video to frames if needed,
and returns a ClipManifest ready for stage 2 or stage 3.

Public API
----------
load(clip) -> ClipManifest
    The single entry point for stage 1. Pass a Clip from scan(), get back
    a ClipManifest with resolved frame paths, output directory, frame count,
    and linear flag.

ClipManifest
    The output contract of stage 1. All downstream stages receive this.
    Contains everything needed — no stage needs to re-read the filesystem
    or re-validate what stage 1 already checked.
"""

from corridorkey_new.loader.contracts import ClipManifest
from corridorkey_new.loader.manifest import load

__all__ = ["load", "ClipManifest"]
