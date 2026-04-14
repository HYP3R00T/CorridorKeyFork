"""Job-level statistics returned via the job_complete event."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class JobStats:
    """Summary of a completed Engine run.

    Attributes:
        clips_processed: Clips that completed successfully.
        clips_failed: Clips that raised a runtime error.
        clips_skipped: Clips skipped during scan.
        clips_cancelled: Clips interrupted by cancel().
        total_frames: Frames written across all completed clips.
        elapsed_seconds: Wall time from job_started to job_complete.
        frames_per_second: Sustained throughput excluding the first-frame
            Triton JIT warmup cost. 0.0 if not enough frames were processed
            to compute a meaningful rate.
    """

    clips_processed: int = 0
    clips_failed: int = 0
    clips_skipped: int = 0
    clips_cancelled: int = 0
    total_frames: int = 0
    elapsed_seconds: float = 0.0
    frames_per_second: float = 0.0
