"""Stage 1 — loader.

Validates a Clip from stage 0, extracts video to frames if needed,
and returns a ClipManifest ready for preprocessing (or for the interface
to generate alpha externally via resolve_alpha() if needs_alpha is True).

Public API
----------
load(clip) -> ClipManifest
    The single entry point for stage 1. Pass a Clip from scan(), get back
    a ClipManifest with resolved frame paths, output directory, frame count,
    and linear flag. Check needs_alpha — if True, alpha generation is the
    interface's responsibility before proceeding.

resolve_alpha(manifest, alpha_frames_dir) -> ClipManifest
    Called by the interface after external alpha generation completes.
    Returns an updated manifest with needs_alpha=False, ready for preprocessing.

ClipManifest
    The output contract of stage 1. All downstream stages receive this.
    Contains everything needed — no stage needs to re-read the filesystem
    or re-validate what stage 1 already checked.
"""

from corridorkey_new.loader.contracts import ClipManifest
from corridorkey_new.loader.extractor import VideoMetadata, load_video_metadata
from corridorkey_new.loader.orchestrator import load, resolve_alpha

__all__ = ["load", "resolve_alpha", "ClipManifest", "VideoMetadata", "load_video_metadata"]
