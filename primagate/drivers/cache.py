from __future__ import annotations

import copy
import secrets
import threading
from collections import OrderedDict

from primagate.schema.types import ChatCompletion

DEFAULT_MAX_ENTRIES = 256


class ResponseCache:
    """Answers kept in memory for identical requests. One per instance, shared by both drivers.

    It copies on the way in and on the way out, so a caller who changes a response cannot change
    what the next caller gets, and two callers never hold the same object. It keeps at most
    `max_entries` answers and drops the one used least recently, so a hot answer survives. The
    bound counts answers, not bytes. A maximum of 0 keeps nothing.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES, salt: bytes | None = None) -> None:
        # Keys the request hash; the drivers hand it to core, which never makes a secret itself.
        # A cache built without one still gets its own, so no path logs a guessable hash.
        self.salt = secrets.token_bytes(16) if salt is None else salt
        self._lock = threading.Lock()
        self._max = max_entries
        self._entries: OrderedDict[str, ChatCompletion] = OrderedDict()

    def get(self, key: str) -> ChatCompletion | None:
        with self._lock:
            found = self._entries.get(key)
            if found is None:
                return None
            self._entries.move_to_end(key)
            return copy.deepcopy(found)

    def put(self, key: str, response: ChatCompletion) -> None:
        if self._max == 0:
            return
        snapshot = copy.deepcopy(response)
        with self._lock:
            self._entries[key] = snapshot
            self._entries.move_to_end(key)
            while len(self._entries) > self._max:
                self._entries.popitem(last=False)
