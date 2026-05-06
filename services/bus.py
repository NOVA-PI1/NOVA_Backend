import asyncio
from collections import defaultdict
from typing import Awaitable, Callable

from schemas import BusEvent


Subscriber = Callable[[BusEvent], Awaitable[None]]


class InMemoryMessageBus:
    def __init__(self) -> None:
        self._events: dict[str, list[BusEvent]] = defaultdict(list)
        self._subscribers: list[Subscriber] = []
        self._lock = asyncio.Lock()

    async def publish(self, event: BusEvent) -> None:
        async with self._lock:
            self._events[event.session_id].append(event)
            subscribers = list(self._subscribers)

        for subscriber in subscribers:
            await subscriber(event)

    async def events_for_session(self, session_id: str) -> list[BusEvent]:
        async with self._lock:
            return list(self._events.get(session_id, []))

    def subscribe(self, subscriber: Subscriber) -> None:
        self._subscribers.append(subscriber)
