from __future__ import annotations

import threading


class SpendTracker:
    """A running total of estimated spend, and whether it has passed a ceiling yet.

    The ceiling is advice, so this only ever reports: it never refuses anything. One lock
    guards the add and the check together, so two calls cannot both be the one that crosses.
    """

    def __init__(self) -> None:
        self._total = 0.0
        self._warned = False
        self._lock = threading.Lock()

    @property
    def total(self) -> float:
        with self._lock:
            return self._total

    def add(self, amount: float, ceiling: float) -> tuple[float, bool]:
        """Add `amount`; return the new total and whether this add first passed `ceiling`."""
        with self._lock:
            self._total += amount
            crossed = self._total > ceiling and not self._warned
            if crossed:
                # Once only, so a total that stays over the ceiling is not repeated on every call.
                self._warned = True
            return self._total, crossed
