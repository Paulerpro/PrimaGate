from __future__ import annotations

import copy
import threading

from primagate.schema.types import ChatCompletion


class ResponseCache:
    """Answers kept in memory for identical requests. One per instance, shared by both drivers.

    It copies on the way in and on the way out, so a caller who changes a response cannot change
    what the next caller gets, and two callers never hold the same object.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, ChatCompletion] = {}

    def get(self, key: str) -> ChatCompletion | None:
        with self._lock:
            found = self._entries.get(key)
            return copy.deepcopy(found) if found is not None else None

    def put(self, key: str, response: ChatCompletion) -> None:
        snapshot = copy.deepcopy(response)
        with self._lock:
            self._entries[key] = snapshot
