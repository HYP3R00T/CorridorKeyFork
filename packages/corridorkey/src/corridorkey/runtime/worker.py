"""Pipeline — worker threads.

Each worker runs in its own thread, pulling from an input queue and pushing
to an output queue. Workers exit cleanly when they receive the STOP sentinel
or when the shared ``cancel_event`` is set.

Workers:
    PreprocessWorker   — reads frames from disk, preprocesses, pushes tensors
    PostWriteWorker    — pulls InferenceResult, postprocesses, writes to disk
                         Uses a ThreadPoolExecutor for parallel postprocess+write
                         so multiple GPUs can drain at full throughput.

The inference worker lives in runner.py as ``_InferenceWorker`` because it
needs the shared remaining counter for coordinated shutdown across N devices.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from corridorkey.errors import FrameReadError, PostprocessError, WriteFailureError
from corridorkey.events import PipelineEvents
from corridorkey.runtime.queue import STOP, BoundedQueue
from corridorkey.stages.inference import InferenceResult
from corridorkey.stages.loader.contracts import ClipManifest
from corridorkey.stages.loader.validator import list_frames
from corridorkey.stages.postprocessor import PostprocessConfig, postprocess_frame
from corridorkey.stages.preprocessor import PreprocessConfig, preprocess_frame
from corridorkey.stages.writer import WriteConfig, write_frame

logger = logging.getLogger(__name__)


@dataclass
class PreprocessWorker:
    """Reads and preprocesses frames, pushing PreprocessedFrame onto the queue.

    Attributes:
        manifest: ClipManifest for the clip being processed.
        config: Preprocessing configuration (img_size, device, strategy).
        preprocess_queue: Queue to push PreprocessedFrame objects onto.
        inference_queue: Inference queue reference — used only for queue depth
            snapshots fired via events.
        events: Optional pipeline event callbacks.
        cancel_event: Optional event that signals the worker to stop early.
            When set, the worker exits after the current frame and sends STOP
            downstream so the rest of the pipeline drains cleanly.
    """

    manifest: ClipManifest
    config: PreprocessConfig
    preprocess_queue: BoundedQueue
    inference_queue: BoundedQueue | None = None
    events: PipelineEvents | None = None
    cancel_event: threading.Event | None = None

    def run(self) -> None:
        """Entry point for the preprocess thread."""
        import os
        import sys

        total = self.manifest.frame_count
        if self.events:
            self.events.stage_start("preprocess", total)

        image_files = list_frames(self.manifest.frames_dir)
        alpha_files = list_frames(self.manifest.alpha_frames_dir)  # type: ignore[arg-type]

        # Estimate frame size for RAM throttling (updated after first read).
        _frame_bytes_estimate: list[int] = [0]
        _mem_headroom = 1 * 1024 * 1024 * 1024  # 1 GB

        def _available_ram() -> int:
            """Available system RAM in bytes. Returns a large value on failure."""
            try:
                if sys.platform == "linux":
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if line.startswith("MemAvailable:"):
                                return int(line.split()[1]) * 1024
                elif sys.platform == "win32":
                    import ctypes

                    class _MEMSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]

                    stat = _MEMSTATUSEX()
                    stat.dwLength = ctypes.sizeof(stat)
                    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                        return stat.ullAvailPhys
            except Exception:
                pass
            # macOS / fallback
            try:
                return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")  # type: ignore[attr-defined]
            except (AttributeError, ValueError):
                return _mem_headroom * 8  # conservative: assume plenty of RAM

        def _mem_ok() -> bool:
            """True if there is enough RAM to decode another frame."""
            est = _frame_bytes_estimate[0]
            if est == 0:
                return True  # no estimate yet — allow first reads
            return _available_ram() - est > _mem_headroom

        try:
            for i in range(*self.manifest.frame_range):
                if self.cancel_event and self.cancel_event.is_set():
                    logger.debug("preprocess_worker: cancelled at frame %d", i)
                    break

                # RAM throttle: pause if available memory minus one frame's
                # estimated size would drop below 1 GB. Resumes when the
                # inference/postwrite pipeline drains and frees memory.
                if not _mem_ok():
                    logger.debug("preprocess_worker: RAM pressure, waiting before frame %d", i)
                    while not _mem_ok():
                        if self.cancel_event and self.cancel_event.is_set():
                            break
                        import time as _time

                        _time.sleep(0.1)

                try:
                    frame = preprocess_frame(
                        self.manifest,
                        i,
                        self.config,
                        image_files=image_files,
                        alpha_files=alpha_files,
                    )
                    # Update frame size estimate from the first successful read.
                    if _frame_bytes_estimate[0] == 0:
                        _frame_bytes_estimate[0] = frame.tensor.nbytes
                    # Use put_unless_cancelled so a full queue doesn't block
                    # indefinitely when the user cancels mid-run.
                    if self.cancel_event is not None:
                        enqueued = self.preprocess_queue.put_unless_cancelled(frame, self.cancel_event)
                        if not enqueued:
                            logger.debug("preprocess_worker: cancelled while waiting to enqueue frame %d", i)
                            break
                    else:
                        self.preprocess_queue.put(frame)
                    logger.debug("preprocess_worker: queued frame %d", i)
                    if self.events:
                        self.events.preprocess_queued(i)
                        self.events.queue_depth(
                            len(self.preprocess_queue),
                            len(self.inference_queue) if self.inference_queue else 0,
                        )
                except FrameReadError as e:
                    logger.error("preprocess_worker: skipping frame %d — %s", i, e)
                    if self.events:
                        self.events.frame_error("preprocess", i, e)
        finally:
            self.preprocess_queue.put_stop()
            logger.debug("preprocess_worker: sent STOP")
            if self.events:
                self.events.stage_done("preprocess")

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name="preprocess-worker", daemon=True)
        t.start()
        return t


@dataclass
class PostWriteWorker:
    """Pulls InferenceResult, postprocesses, and writes frames to disk.

    Uses a ThreadPoolExecutor so multiple GPUs can drain at full throughput.
    The pool size defaults to max(2, n_gpus * 2) — enough threads to keep
    every GPU's output draining without unbounded memory growth.

    Attributes:
        inference_queue: Queue to pull InferenceResult objects from.
        output_dir: Directory to write output frames into.
        postprocess_config: Postprocessing options (despill, despeckle, checkerboard).
        write_config: Writer options (formats, enabled outputs).
            If None, a default WriteConfig pointing at output_dir is used.
        total_frames: Total frame count passed to frame_written events.
        n_write_workers: Thread pool size. 0 = auto (max(2, cpu_count // 4)).
        events: Optional pipeline event callbacks.
        cancel_event: Optional event that signals the worker to stop early.
    """

    inference_queue: BoundedQueue
    output_dir: Path
    postprocess_config: PostprocessConfig = field(default_factory=PostprocessConfig)
    write_config: WriteConfig | None = None
    total_frames: int = 0
    n_write_workers: int = 0
    events: PipelineEvents | None = None
    cancel_event: threading.Event | None = None

    def run(self) -> None:
        """Entry point for the postprocess+write thread."""
        import os

        if self.events:
            self.events.stage_start("postwrite", self.total_frames)

        write_cfg = self.write_config or WriteConfig(output_dir=self.output_dir)
        n_workers = self.n_write_workers or max(2, (os.cpu_count() or 4) // 4)

        futures: list = []

        with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="postwrite") as pool:
            while True:
                if self.cancel_event and self.cancel_event.is_set():
                    logger.debug("postwrite_worker: cancelled, draining queue")
                    self.inference_queue.put_stop()
                    break
                item = self.inference_queue.get()
                if item is STOP:
                    self.inference_queue.put_stop()
                    break
                assert isinstance(item, InferenceResult)
                frame_index = item.meta.frame_index
                future = pool.submit(self._process_one, item, write_cfg)
                futures.append((frame_index, future))
            # pool.__exit__ waits for all submitted futures to complete

        # Collect results and fire events after all futures are done.
        for frame_index, future in futures:
            self._collect(frame_index, future)

        logger.debug("postwrite_worker: done")
        if self.events:
            self.events.stage_done("postwrite")

    def _process_one(self, item: InferenceResult, write_cfg: WriteConfig) -> int:
        """Postprocess and write one frame. Returns frame_index on success."""
        processed = postprocess_frame(item, self.postprocess_config, output_dir=self.output_dir)
        write_frame(processed, write_cfg)
        return item.meta.frame_index

    def _collect(self, frame_index: int, future: object) -> None:
        """Handle a completed future — fire events or log errors."""
        try:
            future.result()  # type: ignore[union-attr]
            logger.debug("postwrite_worker: wrote frame %d", frame_index)
            if self.events:
                self.events.frame_written(frame_index, self.total_frames)
        except WriteFailureError as e:
            logger.error("postwrite_worker: write failed frame %d — %s", frame_index, e)
            if self.events:
                self.events.frame_error("postwrite", frame_index, e)
        except OSError as e:
            typed = WriteFailureError(str(getattr(e, "filename", "unknown")), str(e))
            logger.error("postwrite_worker: write failed frame %d — %s", frame_index, typed)
            if self.events:
                self.events.frame_error("postwrite", frame_index, typed)
        except Exception as e:
            typed = PostprocessError(frame_index, str(e))
            logger.error("postwrite_worker: postprocess failed frame %d — %s", frame_index, typed)
            if self.events:
                self.events.frame_error("postwrite", frame_index, typed)

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name="postwrite-worker", daemon=True)
        t.start()
        return t
