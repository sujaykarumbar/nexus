"""
NEXUS Event Bus
Lightweight asyncio-based pub/sub broker. Topics are string keys; producers
publish() records; consumers subscribe() and receive records via async iteration.

Architecture note: The interface deliberately mirrors the confluent-kafka
Producer/Consumer API so that dropping in a real Kafka broker only requires
swapping the backend, not the calling code.
"""

import asyncio
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional


class EventBus:
    """
    In-process async event bus backed by asyncio.Queue per topic per subscriber.

    Thread-safety: All methods are coroutine-safe when called from a single
    event loop. Use asyncio.run_coroutine_threadsafe() for cross-thread bridging.
    """

    def __init__(self) -> None:
        # topic -> list of subscriber queues
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def publish(self, topic: str, record: Any) -> None:
        """Push a record to all subscribers of the given topic."""
        async with self._lock:
            queues = list(self._subscribers.get(topic, []))
        for q in queues:
            try:
                q.put_nowait(record)
            except asyncio.QueueFull:
                # Drop oldest message to make room for new one (ring-buffer behaviour)
                try:
                    q.get_nowait()
                    q.put_nowait(record)
                except Exception:
                    pass

    async def subscribe(self, topic: str, maxsize: int = 500) -> "Subscription":
        """
        Returns a Subscription context-manager / async iterator.
        Caller must call .unsubscribe() or use 'async with' to clean up.
        """
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        async with self._lock:
            self._subscribers.setdefault(topic, []).append(q)
        return Subscription(bus=self, topic=topic, queue=q)

    async def _unsubscribe(self, topic: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            subs = self._subscribers.get(topic, [])
            if queue in subs:
                subs.remove(queue)

    def topic_subscriber_count(self, topic: str) -> int:
        return len(self._subscribers.get(topic, []))


class Subscription:
    """Async iterator that receives records published to a topic."""

    _SENTINEL = object()

    def __init__(self, bus: EventBus, topic: str, queue: asyncio.Queue) -> None:
        self._bus = bus
        self._topic = topic
        self._queue = queue
        self._closed = False

    async def get(self, timeout: Optional[float] = 1.0) -> Optional[Any]:
        """Get next record, returning None on timeout."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def unsubscribe(self) -> None:
        if not self._closed:
            self._closed = True
            await self._bus._unsubscribe(self._topic, self._queue)

    def __aiter__(self) -> "Subscription":
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        record = await self.get()
        if record is self._SENTINEL:
            raise StopAsyncIteration
        return record

    async def __aenter__(self) -> "Subscription":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.unsubscribe()


# Global singleton bus shared across all streaming sessions
_global_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Return the process-global EventBus singleton."""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus
