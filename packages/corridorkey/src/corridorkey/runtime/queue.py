"""Pipeline — bounded queue with sentinel-based shutdown.

Every inter-stage queue in the pipeline is a BoundedQueue. Capacity is fixed
at construction time — when full, producers block. This is backpressure: the
pipeline naturally throttles to the speed of the slowest stage without
unbounded memory growth.

Shutdown uses a sentinel object. When a producer is done it puts STOP on the
queue. The consumer pulls STOP, re-puts it (so any other consumer on the same
queue also sees it), and exits. This propagates shutdown downstream
automatically without any shared flags or events.
"""

from __future__ import annotations

import queue
import threading

# Sentinel — a unique object that signals no more items will be produced.
# Identity comparison (``item is STOP``) is used, never equality.
STOP = object()


class BoundedQueue[T]:
    """A thread-safe FIFO queue with a fixed capacity.

    Wraps ``queue.Queue`` with a typed interface and the sentinel pattern
    for clean shutdown.

    Args:
        capacity: Maximum number of items the queue can hold. Producers
            block on ``put()`` when the queue is full.
    """

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError(f"Queue capacity must be >= 1, got {capacity}.")
        self._q: queue.Queue = queue.Queue(maxsize=capacity)
        self.capacity = capacity

    def put(self, item: T) -> None:
        """Put an item. Blocks if the queue is full."""
        self._q.put(item)

    def put_unless_cancelled(self, item: T, cancel_event: threading.Event, poll_interval: float = 0.05) -> bool:
        """Put an item, returning False immediately if ``cancel_event`` is set.

        Polls in short intervals so a cancellation signal is never missed even
        when the queue is full and the normal ``put()`` would block indefinitely.

        Args:
            item: The item to enqueue.
            cancel_event: Event that signals cancellation.
            poll_interval: Seconds between attempts when the queue is full.

        Returns:
            True if the item was enqueued, False if cancelled before it could be.
        """
        while not cancel_event.is_set():
            try:
                self._q.put(item, timeout=poll_interval)
                return True
            except queue.Full:
                continue
        return False

    def put_stop(self) -> None:
        """Signal that no more items will be produced.

        Puts the STOP sentinel. Blocks if the queue is full — this ensures
        the sentinel is never lost even under backpressure.
        """
        self._q.put(STOP)

    def get(self) -> T | object:
        """Get the next item. Blocks until one is available.

        Returns:
            The next item, or the ``STOP`` sentinel if the producer is done.
            Always check ``item is STOP`` before using the value.
        """
        return self._q.get()

    def task_done(self) -> None:
        """Mark the last item returned by ``get()`` as processed."""
        self._q.task_done()

    def __len__(self) -> int:
        return self._q.qsize()
