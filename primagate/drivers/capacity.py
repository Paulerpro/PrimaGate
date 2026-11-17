from __future__ import annotations

import threading
from typing import Final

# Provisional: the evaluation spike measures real Analyzer load, and this is tuned from that.
DEFAULT_MAX_CONCURRENT_ANALYZER_CALLS: Final = 8


class AnalyzerCap:
    """A count of Analyzer requests in flight, with a limit and no waiting.

    A call that cannot get a permit goes to the Interim Binding at once, so nothing queues behind
    a slow Analyzer. One lock guards the check and the change together, so two callers can never
    both take the last permit.
    """

    def __init__(self, limit: int) -> None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a whole number of 1 or more")
        self._limit = limit
        self._in_flight = 0
        self._lock = threading.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    def try_acquire(self) -> bool:
        with self._lock:
            if self._in_flight >= self._limit:
                return False
            self._in_flight += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._in_flight < 1:
                # A release with no permit held would let more through than the limit allows.
                raise RuntimeError("release without a permit")
            self._in_flight -= 1
