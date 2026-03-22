"""Clip state machine.

Tracks the processing lifecycle of a single clip from discovery through
completed inference. Consumed by any interface layer (CLI, GUI, web).

State machine transitions
-------------------------

    EXTRACTING -> RAW        video extraction completes
    EXTRACTING -> ERROR      video extraction fails
    RAW        -> READY      alpha frames generated externally
    RAW        -> ERROR      alpha generation fails
    READY      -> COMPLETE   inference succeeds
    READY      -> ERROR      inference fails
    COMPLETE   -> READY      reprocess with different params
    ERROR      -> RAW        retry from scratch
    ERROR      -> READY      retry inference only (alpha already present)
    ERROR      -> EXTRACTING retry extraction

Design notes
------------
- ``ClipEntry`` wraps a ``Clip`` (scanner output) and optionally a
  ``ClipManifest`` (loader output). It does not re-scan the filesystem —
  that is the scanner's job.
- State is resolved from disk at construction time via ``resolve_state()``.
- ``transition_to()`` validates every transition against the table and raises
  ``InvalidStateTransitionError`` on illegal moves.
- ``_processing`` is a soft lock set by the job queue while a GPU job is
  active. Watchers should skip reclassification while it is True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from corridorkey_new.errors import InvalidStateTransitionError
from corridorkey_new.stages.loader.validator import count_frames, get_frame_files
from corridorkey_new.stages.scanner.contracts import Clip

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class ClipState(Enum):
    """Processing lifecycle state of a single clip."""

    EXTRACTING = "EXTRACTING"
    """Video source present but frame sequence not yet extracted."""

    RAW = "RAW"
    """Frame sequence present, no alpha hint."""

    READY = "READY"
    """Alpha hint present — clip can be submitted for inference."""

    COMPLETE = "COMPLETE"
    """Inference has run and all output frames are written."""

    ERROR = "ERROR"
    """A stage failed. ``ClipEntry.error_message`` contains the detail."""


# Valid transitions: from_state -> allowed to_states.
_TRANSITIONS: dict[ClipState, set[ClipState]] = {
    ClipState.EXTRACTING: {ClipState.RAW, ClipState.ERROR},
    ClipState.RAW: {ClipState.READY, ClipState.ERROR},
    ClipState.READY: {ClipState.COMPLETE, ClipState.ERROR},
    ClipState.COMPLETE: {ClipState.READY},
    ClipState.ERROR: {ClipState.RAW, ClipState.READY, ClipState.EXTRACTING},
}


# ---------------------------------------------------------------------------
# InOutRange
# ---------------------------------------------------------------------------


@dataclass
class InOutRange:
    """Inclusive in/out frame range for sub-clip processing.

    Both indices are zero-based and inclusive.

    Attributes:
        in_point: First frame index to process.
        out_point: Last frame index to process (inclusive).
    """

    in_point: int
    out_point: int

    @property
    def frame_count(self) -> int:
        """Number of frames in the range."""
        return self.out_point - self.in_point + 1

    def contains(self, index: int) -> bool:
        """True if ``in_point <= index <= out_point``."""
        return self.in_point <= index <= self.out_point

    def to_frame_range(self) -> tuple[int, int]:
        """Convert to a half-open ``(start, end)`` tuple for ``ClipManifest.frame_range``."""
        return self.in_point, self.out_point + 1


# ---------------------------------------------------------------------------
# ClipEntry
# ---------------------------------------------------------------------------


@dataclass
class ClipEntry:
    """A single clip with its processing state.

    Construct via :func:`from_clip` (preferred) or directly for testing.

    Attributes:
        clip: Scanner output — resolved paths and alpha presence.
        state: Current lifecycle state.
        manifest: Loader output, set after ``load()`` succeeds.
            None until the clip has been loaded.
        in_out_range: Optional sub-clip range. None means process all frames.
        warnings: Non-fatal messages accumulated during scanning or processing.
        error_message: Set when state is ERROR.
    """

    clip: Clip
    state: ClipState = ClipState.RAW
    manifest: object | None = None  # ClipManifest — avoid circular import at module level
    in_out_range: InOutRange | None = None
    warnings: list[str] = field(default_factory=list)
    error_message: str | None = None
    _processing: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_clip(cls, clip: Clip) -> ClipEntry:
        """Create a ClipEntry from a scanner Clip and resolve its initial state.

        Args:
            clip: Clip produced by the scanner stage.

        Returns:
            ClipEntry with state resolved from what is present on disk.
        """
        entry = cls(clip=clip)
        entry.state = _resolve_state(clip)
        return entry

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable clip name."""
        return self.clip.name

    @property
    def root(self) -> Path:
        """Absolute path to the clip folder."""
        return self.clip.root

    @property
    def output_dir(self) -> Path:
        """Absolute path to the Output subdirectory."""
        return self.root / "Output"

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    @property
    def is_processing(self) -> bool:
        """True while a GPU job is actively working on this clip."""
        return self._processing

    def set_processing(self, value: bool) -> None:
        """Acquire or release the processing lock.

        The job queue sets this to True before dispatching a job and False
        when the job completes or fails. Filesystem watchers should skip
        reclassification while this is True.

        Args:
            value: True to lock, False to release.
        """
        self._processing = value

    def transition_to(self, new_state: ClipState) -> None:
        """Attempt a validated state transition.

        Args:
            new_state: Target state.

        Raises:
            InvalidStateTransitionError: If the transition is not in the table.
        """
        allowed = _TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise InvalidStateTransitionError(self.name, self.state.value, new_state.value)
        old = self.state
        self.state = new_state
        if new_state != ClipState.ERROR:
            self.error_message = None
        logger.debug("clip '%s': %s -> %s", self.name, old.value, new_state.value)

    def set_error(self, message: str) -> None:
        """Transition to ERROR and record a message.

        Args:
            message: Human-readable description of the failure.
        """
        self.transition_to(ClipState.ERROR)
        self.error_message = message

    def refresh_state(self) -> None:
        """Re-resolve state from disk.

        Call this after an external operation (alpha generation, inference)
        completes to bring the entry back in sync with the filesystem.
        Does nothing if ``is_processing`` is True.
        """
        if self._processing:
            return
        self.state = _resolve_state(self.clip)

    # ------------------------------------------------------------------
    # Output inspection
    # ------------------------------------------------------------------

    @property
    def has_outputs(self) -> bool:
        """True if the Output directory contains at least one written frame."""
        if not self.output_dir.is_dir():
            return False
        for subdir in ("alpha", "fg", "comp", "processed"):
            d = self.output_dir / subdir
            if d.is_dir() and any(d.iterdir()):
                return True
        return False

    def completed_frame_count(self) -> int:
        """Number of frames that have all enabled outputs written."""
        return len(self.completed_stems())

    def completed_stems(self) -> set[str]:
        """Frame stems that have outputs in every enabled output subdirectory.

        Checks ``alpha/``, ``fg/``, ``comp/``, and ``processed/`` — whichever
        exist under ``Output/``. Returns the intersection so a stem is only
        counted when all present output types have written it.

        Returns:
            Set of stem strings (e.g. ``{'frame_000001', 'frame_000002'}``).
        """
        stem_sets: list[set[str]] = []
        for subdir in ("alpha", "fg", "comp", "processed"):
            d = self.output_dir / subdir
            if d.is_dir():
                stems = {p.stem for p in get_frame_files(d)}
                if stems:
                    stem_sets.append(stems)

        if not stem_sets:
            return set()

        result = stem_sets[0]
        for s in stem_sets[1:]:
            result &= s
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_state(clip: Clip) -> ClipState:
    """Derive the initial ClipState from what is present on disk.

    Priority (highest first):
        COMPLETE   — output frames cover all input frames
        READY      — alpha frames cover all input frames
        EXTRACTING — input is a video file (frames not yet extracted)
        RAW        — frame sequence present, no alpha

    Args:
        clip: Scanner Clip with resolved paths.

    Returns:
        The most advanced state the clip has reached.
    """
    from corridorkey_new.stages.loader.extractor import is_video

    # Input is a raw video — frames haven't been extracted yet.
    if is_video(clip.input_path):
        return ClipState.EXTRACTING

    input_count = count_frames(clip.input_path)
    if input_count == 0:
        return ClipState.RAW

    # Alpha present — check if it covers all input frames.
    if clip.alpha_path is not None:
        alpha_count = count_frames(clip.alpha_path)
        if alpha_count >= input_count:
            # Check if outputs also cover all frames → COMPLETE.
            # All present output subdirs must have sufficient frames.
            output_dir = clip.root / "Output"
            output_subdirs = [d for sub in ("alpha", "fg", "comp", "processed") if (d := output_dir / sub).is_dir()]
            if output_subdirs and all(count_frames(d) >= input_count for d in output_subdirs):
                return ClipState.COMPLETE
            return ClipState.READY
        else:
            logger.info(
                "clip '%s': partial alpha (%d/%d frames), treating as RAW",
                clip.name,
                alpha_count,
                input_count,
            )

    return ClipState.RAW
