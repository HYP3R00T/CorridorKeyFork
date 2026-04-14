from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from corridorkey.errors import ClipLoadError, FrameMismatchError
from corridorkey.infra.utils import natural_sort_key

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".exr", ".png", ".jpg", ".jpeg", ".tiff", ".tif"})
LINEAR_EXTENSIONS: frozenset[str] = frozenset({".exr"})


@dataclass(frozen=True)
class FrameScan:
    """Result of a single directory scan.

    Attributes:
        files: Naturally sorted list of image file paths.
        is_linear: True if the first frame has a linear extension (e.g. .exr).
    """

    files: tuple[Path, ...]
    is_linear: bool

    @property
    def count(self) -> int:
        return len(self.files)


def scan_frames(path: Path) -> FrameScan:
    """Scan a directory for image frames in a single pass.

    Returns a FrameScan with the sorted file list and linearity flag.
    This is the single entry point for all frame discovery — call it once
    and pass the result to ``validate_scan()`` and other helpers.

    Args:
        path: Directory to scan.

    Returns:
        FrameScan with sorted files and is_linear flag.
    """
    if not path.is_dir():
        return FrameScan(files=(), is_linear=False)

    files = sorted(
        (p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: natural_sort_key(p.name),
    )
    is_linear = bool(files) and files[0].suffix.lower() in LINEAR_EXTENSIONS
    return FrameScan(files=tuple(files), is_linear=is_linear)


def list_frames(path: Path) -> list[Path]:
    """Return naturally sorted image files in a clip frame directory.

    Convenience wrapper around :func:`scan_frames` for callers that only need
    the file list. Prefer :func:`scan_frames` directly when you also need the
    count or linearity flag to avoid redundant attribute access.

    Typical use in the frame loop::

        imgs = list_frames(manifest.frames_dir)
        alps = list_frames(manifest.alpha_frames_dir)
        for i in range(*manifest.frame_range):
            preprocessed = preprocess_frame(manifest, i, config, image_files=imgs, alpha_files=alps)

    Args:
        path: Directory containing the frame sequence.

    Returns:
        Naturally sorted list of image file paths.
    """
    return list(scan_frames(path).files)


def count_frames(path: Path) -> int:
    return scan_frames(path).count


def validate(
    clip_name: str,
    frames_dir: Path,
    alpha_frames_dir: Path | None,
    expected_frame_count: int | None = None,
) -> tuple[FrameScan | None, FrameScan | None]:
    """Validate resolved frame sequence directories.

    Performs a single iterdir() pass per directory. Returns the FrameScan
    results so callers can reuse them without re-scanning.

    Args:
        clip_name: Clip name for error messages.
        frames_dir: Resolved input frames directory.
        alpha_frames_dir: Resolved alpha frames directory, or None.
        expected_frame_count: If provided, skip scanning frames_dir and use
            this count instead (used by attach_alpha when the count is already
            known from the manifest). The returned input_scan will be None.

    Returns:
        (input_scan, alpha_scan). input_scan is None when expected_frame_count
        is provided. alpha_scan is None if alpha_frames_dir is None.

    Raises:
        ClipLoadError: If input has no frames, or alpha directory is empty.
        FrameMismatchError: If alpha frame count mismatches input.
    """
    if expected_frame_count is not None:
        # Input already validated — only scan alpha.
        input_scan = None
        input_count = expected_frame_count
    else:
        input_scan = scan_frames(frames_dir)
        input_count = input_scan.count
        if input_count == 0:
            raise ClipLoadError(clip_name, f"no image frames found in {frames_dir}")

    alpha_scan: FrameScan | None = None
    if alpha_frames_dir is not None:
        alpha_scan = scan_frames(alpha_frames_dir)
        if alpha_scan.count == 0:
            raise ClipLoadError(clip_name, f"alpha directory is empty: {alpha_frames_dir}")
        if alpha_scan.count != input_count:
            raise FrameMismatchError(clip_name, input_count, alpha_scan.count)

    return input_scan, alpha_scan
