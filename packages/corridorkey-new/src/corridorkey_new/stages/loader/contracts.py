"""Stage 1 contracts."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, model_validator


class ClipManifest(BaseModel):
    """Output contract of stage 1. Input to all downstream stages.

    Downstream stages only receive what they need — resolved frame paths,
    output destination, and clip metadata. All discovery, validation, and
    extraction decisions are made in stage 1 and are not repeated.

    Attributes:
        clip_name: Clip name, carried through for logging and output naming.
        clip_root: Absolute path to the clip folder. Parent of Input/, Output/,
            Frames/, AlphaHint/, etc.
        frames_dir: Directory containing the input frame sequence.
            Points to ``Input/`` for image sequence inputs, or ``Frames/``
            for video inputs (extracted by stage 1).
        alpha_frames_dir: Directory containing the alpha hint frame sequence.
            Points to ``AlphaHint/`` for image sequence inputs, or
            ``AlphaFrames/`` for video inputs. None if absent — the interface
            layer is responsible for generating alpha externally and calling
            ``resolve_alpha()`` to update the manifest before proceeding.
        output_dir: Directory where stage 6 writes all output images.
            Created by stage 1 at ``clip/Output/``.
        needs_alpha: True if alpha is absent. The interface layer must generate
            alpha externally and call ``resolve_alpha()`` before proceeding.
        frame_count: Total number of input frames.
        frame_range: Half-open range ``(start, end)`` of frames to process.
            Defaults to ``(0, frame_count)`` — the full sequence.
            Can be narrowed for partial runs or testing.
        is_linear: True if input frames are in linear light (e.g. .exr).
        video_meta_path: Path to the ``video_meta.json`` sidecar file written
            by stage 1 during video extraction. None for image sequence inputs.
            Stage 6 reads this to re-encode output with matching properties.
    """

    clip_name: str
    clip_root: Path
    frames_dir: Path
    alpha_frames_dir: Path | None
    output_dir: Path
    needs_alpha: bool
    frame_count: int
    frame_range: tuple[int, int]
    is_linear: bool
    video_meta_path: Path | None = None

    @model_validator(mode="after")
    def validate_frame_range(self) -> ClipManifest:
        start, end = self.frame_range
        if start < 0:
            raise ValueError(f"frame_range start must be >= 0, got {start}")
        if end > self.frame_count:
            raise ValueError(f"frame_range end ({end}) exceeds frame_count ({self.frame_count})")
        if start >= end:
            raise ValueError(f"frame_range start ({start}) must be less than end ({end})")
        return self
