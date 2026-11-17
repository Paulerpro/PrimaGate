from __future__ import annotations

import asyncio
import threading
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Final

from primagate.core import steps
from primagate.core.preparation import may_start
from primagate.core.profile import PREP_PROMPT_VERSION
from primagate.schema.errors import PrimaGateError
from primagate.schema.events import PreparationFailed, PreparationStarted, PrepFailureReason
from primagate.schema.profile import PrepState, Profile

# Provisional: long enough that a broken provider or store is not hammered by every request,
# short enough that a passing fault does not leave the app unprepared for long.
PREP_RESTART_COOLDOWN: Final = 600.0


class PreparationStopped(PrimaGateError):
    """The instance was closed while a preparation ran, so it stopped before its next action.

    Nothing further was sent or written, and no final event follows. A `PrimaGateError`, as every
    other use of a closed instance raises.
    """

    def __init__(self) -> None:
        super().__init__("PrimaGate instance is closed")


class StoreFailed(Exception):
    """The Profile store raised during a background preparation; the store's error is the cause.

    Only the background job uses it: a foreground `prepare()` lets the store's error through.
    """


@dataclass
class Progress:
    """What a background preparation has shown so far, so a failure the flow itself could not
    report can still name its id and attempt, and the runner can learn why it failed."""

    prep_id: str | None = None
    attempts: int = 0
    reason: PrepFailureReason | None = None

    def observe(self, action: steps.Action) -> None:
        if isinstance(action, steps.Analyze):
            self.attempts += 1
        elif isinstance(action, steps.Emit):
            for event in action.events:
                if isinstance(event, PreparationStarted):
                    self.prep_id = event.request_id
                elif isinstance(event, PreparationFailed) and event.final:
                    self.reason = event.reason

    def failed(self, fingerprint: str, reason: PrepFailureReason) -> PreparationFailed:
        # A store read fails before the flow has asked for an id, so the job makes its own.
        prep_id = self.prep_id or f"prep-{uuid.uuid4().hex}"
        return PreparationFailed(
            prep_id, fingerprint, PREP_PROMPT_VERSION, reason, self.attempts, True
        )


class PreparationRunner:
    """Where each fingerprint's preparation stands, and the background work running it.

    One per instance, shared by both drivers, so a sync call and an async call never both start
    one. The states live in memory only: a restart forgets them, so it can never find a stale
    PREPARING. One lock guards every check and change together, so of many first requests for
    a fingerprint exactly one starts its preparation.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[str, PrepState] = {}
        # Each READY fingerprint's Profile, kept beside its state so a call is handed both at once
        # and never reads the store.
        self._profiles: dict[str, Profile] = {}
        # When each FAILED fingerprint failed, and whether it may ever start again.
        self._failures: dict[str, tuple[float, bool]] = {}
        self._closed = False
        # Held until done: the loop keeps only a weak reference, so an unheld task could vanish.
        self._tasks: set[asyncio.Task[None]] = set()
        self._threads: set[threading.Thread] = set()

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def state(self, fingerprint: str) -> PrepState | None:
        with self._lock:
            return self._states.get(fingerprint)

    def try_start(self, fingerprint: str) -> bool:
        """Mark `fingerprint` PREPARING and return True, if a preparation may start now."""
        with self._lock:
            if self._closed:
                return False
            state = self._states.get(fingerprint)
            if not may_start(state):
                return False
            if state is PrepState.FAILED:
                failed_at, restartable = self._failures[fingerprint]
                if not restartable or self._clock() - failed_at < PREP_RESTART_COOLDOWN:
                    return False
            self._states[fingerprint] = PrepState.PREPARING
            return True

    def outcome(self, fingerprint: str, started: bool) -> steps.StartOutcome:
        """What a call starting now is served by: the state, and the Profile when READY.

        A call that just started the preparation is PREPARING whatever happens next, so it never
        depends on how fast the job runs. A fingerprint with no state, because the instance is
        closed or the start was abandoned, is served as PREPARING too: no Profile is coming yet.
        """
        if started:
            return steps.StartOutcome(True, PrepState.PREPARING)
        with self._lock:
            state = self._states.get(fingerprint, PrepState.PREPARING)
            profile = self._profiles.get(fingerprint) if state is PrepState.READY else None
        return steps.StartOutcome(False, state, profile)

    def finish(
        self,
        fingerprint: str,
        state: PrepState,
        reason: PrepFailureReason | None = None,
        profile: Profile | None = None,
    ) -> None:
        """Record how a preparation ended, with the Profile it wrote or loaded when READY.

        READY needs its Profile: without one, every call would be served as preparing, and a
        READY fingerprint is never prepared again.
        """
        if state is PrepState.READY and profile is None:
            raise ValueError("a READY preparation needs its Profile")
        with self._lock:
            self._states[fingerprint] = state
            if profile is not None and state is PrepState.READY:
                self._profiles[fingerprint] = profile
            else:
                self._profiles.pop(fingerprint, None)
            if state is PrepState.FAILED:
                # Sources too large stay too large, so trying again could only pay twice.
                self._failures[fingerprint] = (self._clock(), reason != "overflow")
            else:
                self._failures.pop(fingerprint, None)

    def abandon(self, fingerprint: str) -> None:
        """Forget a preparation that ended with no result, so a later request can start it."""
        with self._lock:
            if self._states.get(fingerprint) is PrepState.PREPARING:
                del self._states[fingerprint]

    def start_thread(self, fingerprint: str, job: Callable[[], None]) -> bool:
        """Start `job` on a daemon thread if a preparation may start; never waits on it."""
        if not self.try_start(fingerprint):
            return False

        def run() -> None:
            try:
                job()
            finally:
                with self._lock:
                    self._threads.discard(thread)

        # Daemon, so a preparation still running never keeps the process from exiting.
        thread = threading.Thread(target=run, name="primagate-preparation", daemon=True)
        with self._lock:
            self._threads.add(thread)
        try:
            thread.start()
        except RuntimeError:
            # No thread to be had: the request goes on, and a later one may try again.
            with self._lock:
                self._threads.discard(thread)
            self.abandon(fingerprint)
            return False
        return True

    def start_task(
        self,
        fingerprint: str,
        loop: asyncio.AbstractEventLoop,
        job: Callable[[], Coroutine[Any, Any, None]],
    ) -> bool:
        """Start `job` as a task on `loop` if a preparation may start; never waits on it."""
        if not self.try_start(fingerprint):
            return False
        work = job()
        try:
            task = loop.create_task(work)
        except RuntimeError:
            work.close()
            self.abandon(fingerprint)
            return False
        with self._lock:
            self._tasks.add(task)

        def done(finished: asyncio.Task[None]) -> None:
            with self._lock:
                self._tasks.discard(finished)
            # Cut off by its loop ending rather than by close: a failure like any other, so an
            # app that runs a fresh loop per request does not pay for a lost preparation on
            # every request, and a later one may still try again once the cooldown has passed.
            if finished.cancelled() and not self.is_closed:
                self.finish(fingerprint, PrepState.FAILED)

        task.add_done_callback(done)
        return True

    def close(self) -> None:
        """Start no more preparations; a running one stops at its next action boundary."""
        with self._lock:
            self._closed = True

    async def aclose(self) -> None:
        """Close, then cancel the running tasks and wait for those on this loop to finish."""
        self.close()
        current = asyncio.get_running_loop()
        with self._lock:
            running = [task for task in self._tasks if not task.done()]
        here: list[asyncio.Task[None]] = []
        for task in running:
            if task.get_loop() is current:
                task.cancel()
                here.append(task)
                continue
            # A task on another loop can only be asked to stop from that loop's own thread.
            try:
                task.get_loop().call_soon_threadsafe(task.cancel)
            except RuntimeError:
                pass
        if here:
            await asyncio.gather(*here, return_exceptions=True)
