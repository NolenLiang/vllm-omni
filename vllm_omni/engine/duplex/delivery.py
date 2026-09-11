# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded, single-consumer delivery across the engine and caller threads.

Only queued, unsequenced audio can be removed. The producer still rejects
new stale model output; this buffer does not retain a growing response-id
tombstone table. It tracks just the most recently dequeued event until the
next ``get()`` so a consumer can recheck it immediately before delivery.

Limits cover queued events, with a separate small termination reserve. A
consumer may additionally hold one dequeued event, at most one queue budget
in size. Already journaled or delivered events are outside this buffer.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from vllm_omni.engine.duplex.events import AudioDelta, DuplexEvent, ErrorEvent, ResponseDone, SessionClosed


class DuplexOutputOverflowError(RuntimeError):
    """One session exceeded its pending output budget; valid output was not dropped."""


@dataclass(frozen=True, slots=True)
class _PendingEvent:
    event: DuplexEvent
    size: int
    reserved: bool


class DuplexOutputBuffer:
    """Nonblocking producer, one async consumer, and response-scoped audio removal.

    ``get()`` releases tracking of its previous result. The consumer must
    finish delivering that result before requesting another. ``guard()``
    makes the final validity check and synchronous journal recording atomic
    with invalidation; never await while holding that guard.
    """

    def __init__(
        self,
        *,
        max_bytes: int,
        max_events: int,
        reserve_bytes: int = 64 * 1024,
        reserve_events: int = 8,
    ) -> None:
        if min(max_bytes, max_events, reserve_bytes, reserve_events) <= 0:
            raise ValueError("output buffer limits must be positive")
        self._max_bytes = max_bytes
        self._max_events = max_events
        self._reserve_bytes = reserve_bytes
        self._reserve_events = reserve_events
        self._lock = threading.Lock()
        self._pending: deque[_PendingEvent] = deque()
        self._bytes = self._events = 0
        self._reserved_bytes = self._reserved_events = 0
        self._held: DuplexEvent | None = None
        self._held_valid = True
        self._waiter: tuple[asyncio.AbstractEventLoop, asyncio.Future[None]] | None = None
        self._closed = False

    @property
    def pending_bytes(self) -> int:
        with self._lock:
            return self._bytes + self._reserved_bytes

    @property
    def pending_events(self) -> int:
        with self._lock:
            return self._events + self._reserved_events

    @staticmethod
    def _can_use_reserve(event: DuplexEvent) -> bool:
        return isinstance(event, ErrorEvent | SessionClosed | ResponseDone)

    def put(self, event: DuplexEvent) -> None:
        """Append without blocking; overflow leaves the queue unchanged.

        Error and response/session terminals can use the finite reserve.
        They retain FIFO order and cannot overtake still-valid media. The
        caller must stop the affected session after ordinary overflow instead
        of repeatedly generating errors until the reserve also overflows.
        """
        size = len(json.dumps(event.to_realtime(), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        with self._lock:
            if self._closed:
                raise RuntimeError("duplex output buffer is closed")
            reserved = self._bytes + size > self._max_bytes or self._events >= self._max_events
            if reserved and (
                not self._can_use_reserve(event)
                or self._reserved_bytes + size > self._reserve_bytes
                or self._reserved_events >= self._reserve_events
            ):
                raise DuplexOutputOverflowError("duplex session pending output limit exceeded")
            self._pending.append(_PendingEvent(event=event, size=size, reserved=reserved))
            if reserved:
                self._reserved_bytes += size
                self._reserved_events += 1
            else:
                self._bytes += size
                self._events += 1
            waiter = self._waiter
            self._waiter = None
        self._notify(waiter)

    @staticmethod
    def _matches(event: DuplexEvent, response_id: str, through_epoch: int) -> bool:
        return (
            isinstance(event, AudioDelta)
            and event.response_id == response_id
            and event.epoch is not None
            and event.epoch <= through_epoch
        )

    def invalidate(self, response_id: str, through_epoch: int) -> int:
        """Remove queued audio for one accepted cancellation, returning its count.

        Non-audio events and other responses retain their exact order. An
        already dequeued matching event becomes invalid too. The producer
        must not enqueue further stale events after this call.
        """
        with self._lock:
            kept: deque[_PendingEvent] = deque()
            removed = 0
            for pending in self._pending:
                if self._matches(pending.event, response_id, through_epoch):
                    self._release(pending)
                    removed += 1
                else:
                    kept.append(pending)
            self._pending = kept
            if self._held is not None and self._matches(self._held, response_id, through_epoch):
                self._held_valid = False
            return removed

    def _is_valid(self, event: DuplexEvent) -> bool:
        return not isinstance(event, AudioDelta) or (event is self._held and self._held_valid)

    def is_valid(self, event: DuplexEvent) -> bool:
        """Check the current dequeued event; use ``guard`` for an atomic handoff."""
        with self._lock:
            return self._is_valid(event)

    @contextmanager
    def guard(self, event: DuplexEvent) -> Iterator[bool]:
        """Hold validity stable during synchronous sequencing; never await here."""
        with self._lock:
            yield self._is_valid(event)

    def _release(self, pending: _PendingEvent) -> None:
        if pending.reserved:
            self._reserved_bytes -= pending.size
            self._reserved_events -= 1
        else:
            self._bytes -= pending.size
            self._events -= 1

    async def get(self) -> DuplexEvent | None:
        """Return the next event, or ``None`` once a closed buffer is drained."""
        loop = asyncio.get_running_loop()
        with self._lock:
            self._held = None
        while True:
            with self._lock:
                if self._pending:
                    pending = self._pending.popleft()
                    self._release(pending)
                    self._held = pending.event
                    self._held_valid = True
                    return pending.event
                if self._closed:
                    return None
                if self._waiter is not None:
                    raise RuntimeError("duplex output buffer already has a waiting consumer")
                future: asyncio.Future[None] = loop.create_future()
                self._waiter = (loop, future)
            try:
                await future
            finally:
                with self._lock:
                    if self._waiter is not None and self._waiter[1] is future:
                        self._waiter = None

    def close(self) -> None:
        """Reject further output and wake the consumer, preserving queued events."""
        with self._lock:
            self._closed = True
            waiter = self._waiter
            self._waiter = None
        self._notify(waiter)

    @staticmethod
    def _notify(waiter: tuple[asyncio.AbstractEventLoop, asyncio.Future[None]] | None) -> None:
        if waiter is None:
            return
        loop, future = waiter

        def wake() -> None:
            if not future.done():
                future.set_result(None)

        try:
            loop.call_soon_threadsafe(wake)
        except RuntimeError:
            if not loop.is_closed():
                raise
