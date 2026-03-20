"""Unit tests for corridorkey_new.pipeline.queue."""

from __future__ import annotations

import threading

import pytest
from corridorkey_new.pipeline.queue import STOP, BoundedQueue


class TestBoundedQueue:
    def test_put_and_get(self):
        q: BoundedQueue[int] = BoundedQueue(4)
        q.put(42)
        assert q.get() == 42

    def test_fifo_order(self):
        q: BoundedQueue[int] = BoundedQueue(4)
        for i in range(3):
            q.put(i)
        assert [q.get() for _ in range(3)] == [0, 1, 2]

    def test_stop_sentinel_received(self):
        q: BoundedQueue[int] = BoundedQueue(4)
        q.put_stop()
        assert q.get() is STOP

    def test_stop_is_unique_object(self):
        # STOP must be compared by identity, not equality
        assert STOP is not None
        assert STOP != 0  # noqa: E712
        assert STOP != ""

    def test_capacity_zero_raises(self):
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            BoundedQueue(0)

    def test_capacity_negative_raises(self):
        with pytest.raises(ValueError, match="capacity must be >= 1"):
            BoundedQueue(-1)

    def test_len_reflects_queue_size(self):
        q: BoundedQueue[int] = BoundedQueue(4)
        assert len(q) == 0
        q.put(1)
        q.put(2)
        assert len(q) == 2

    def test_blocks_when_full(self):
        """A full queue blocks the producer until a consumer reads."""
        q: BoundedQueue[int] = BoundedQueue(1)
        q.put(1)  # fills the queue

        results = []

        def producer():
            q.put(2)  # should block until consumer reads
            results.append("produced")

        def consumer():
            import time

            time.sleep(0.05)
            q.get()  # unblocks producer
            results.append("consumed")

        t_prod = threading.Thread(target=producer)
        t_cons = threading.Thread(target=consumer)
        t_prod.start()
        t_cons.start()
        t_prod.join(timeout=2)
        t_cons.join(timeout=2)

        assert "consumed" in results
        assert "produced" in results
        assert results.index("consumed") < results.index("produced")

    def test_stop_propagates_to_multiple_consumers(self):
        """put_stop() followed by re-put in consumer allows second consumer to see STOP."""
        q: BoundedQueue[int] = BoundedQueue(4)
        q.put_stop()

        item1 = q.get()
        assert item1 is STOP
        q.put_stop()  # re-put for next consumer

        item2 = q.get()
        assert item2 is STOP
