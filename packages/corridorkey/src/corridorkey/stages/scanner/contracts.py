"""Scanner stage — contracts.

Output contract of stage 0. Consumed by the loader stage (stage 1).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Clip(BaseModel):
    """A clip ready for stage 1. Output contract of stage 0.

    Frozen — all fields are immutable after construction. Downstream stages
    must never mutate a Clip; create a new one if a field needs to change.

    Attributes:
        name: Human-readable clip name derived from the folder name.
        root: Absolute path to the clip folder.
        input_path: Path to the input asset. Either the Input/ directory (for
            pre-structured clips) or a video file inside Input/ (for normalised
            videos).
        alpha_path: Path to the alpha hint asset. None if absent — the interface
            must generate alpha externally and call resolve_alpha() before
            proceeding.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    root: Path
    input_path: Path
    alpha_path: Path | None

    @field_validator("root", "input_path", "alpha_path")
    @classmethod
    def must_exist(cls, v: Path | None) -> Path | None:
        if v is not None and not v.exists():
            raise ValueError(f"Path does not exist: {v}")
        return v

    @model_validator(mode="after")
    def root_must_be_directory(self) -> Clip:
        if not self.root.is_dir():
            raise ValueError(f"Clip root is not a directory: {self.root}")
        return self

    def __repr__(self) -> str:
        return f"Clip(name={self.name!r}, input={self.input_path}, alpha={self.alpha_path})"


class SkippedPath(BaseModel):
    """A path that was encountered during scanning but could not be used.

    Attributes:
        path: The path that was skipped.
        reason: Human-readable explanation of why it was skipped.
    """

    model_config = ConfigDict(frozen=True)

    path: Path
    reason: str

    def __repr__(self) -> str:
        return f"SkippedPath(path={self.path}, reason={self.reason!r})"


class ScanResult(BaseModel):
    """Complete output of the scanner stage.

    Wraps both the valid clips and any paths that were skipped, so the
    interface can report exactly what was found and what was ignored.

    Attributes:
        clips: Valid clips ready for the loader stage.
        skipped: Paths that were encountered but could not be used, with
            reasons. Empty list if nothing was skipped.
    """

    model_config = ConfigDict(frozen=True)

    clips: tuple[Clip, ...]
    skipped: tuple[SkippedPath, ...]

    @property
    def clip_count(self) -> int:
        return len(self.clips)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    def __repr__(self) -> str:
        return f"ScanResult(clips={self.clip_count}, skipped={self.skipped_count})"
