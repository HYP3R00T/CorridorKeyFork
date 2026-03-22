"""Property-based tests for corridorkey_new.runtime.queue."""

from __future__ import annotations

import threading

from corridorkey_new.runtime.queue import STOP, BoundedQueue
from hypothesis import given, settings
from hypothesis import strategies as st


class TestBoundedQueueProperties:
    @given(st.integers(1, 64))
    def test_capacity_stored(self, cap: int):
        """capacity attribute matches constructor argument."""
        assert BoundedQueue(cap).capacity == cap

    @given(st.integers(min_value=1, max_value=32), st.lists(st.integers(), min_size=1, max_size=32))
    def test_fifo_order_preserved(self, capacity: int, items: list[int]):
        """Items come out in the same order they went in."""
        q: BoundedQueue[int] = BoundedQueue(max(capacity, len(items)))
        for item in items:
            q.put(item)
        received = [q.get() for _ in items]
        assert received == items

    @given(st.integers(min_value=1, max_value=32), st.lists(st.integers(), min_size=0, max_size=32))
    def test_len_tracks_puts(self, capacity: int, items: list[int]):
        """len() reflects the number of items currently in the queue."""
        q: BoundedQueue[int] = BoundedQueue(max(capacity, len(items), 1))
        for i, item in enumerate(items):
            q.put(item)
            assert len(q) == i + 1

    @given(st.integers(min_value=1, max_value=16))
    def test_stop_always_last(self, n: int):
        """STOP is always the last item received after n items + put_stop."""
        q: BoundedQueue[int] = BoundedQueue(n + 1)
        for i in range(n):
            q.put(i)
        q.put_stop()
        for _ in range(n):
            item = q.get()
            assert item is not STOP
        assert q.get() is STOP

    @given(st.integers(min_value=1, max_value=8))
    @settings(max_examples=20)
    def test_producer_consumer_roundtrip(self, n: int):
        """All items produced by a thread are received by the consumer."""
        q: BoundedQueue[int] = BoundedQueue(n)
        produced = list(range(n))
        received: list[int] = []

        def producer() -> None:
            for item in produced:
                q.put(item)
            q.put_stop()

        t = threading.Thread(target=producer)
        t.start()

        while True:
            item = q.get()
            if item is STOP:
                break
            assert isinstance(item, int)
            received.append(item)

        t.join(timeout=5)
        assert received == produced

    @given(st.integers(min_value=1, max_value=32))
    def test_invalid_capacity_raises(self, n: int):
        """Negative or zero capacity always raises ValueError."""
        import pytest

        with pytest.raises(ValueError):
            BoundedQueue(-n)
        with pytest.raises(ValueError):
            BoundedQueue(0)
