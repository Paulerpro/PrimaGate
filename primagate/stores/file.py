from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from pathlib import Path

from primagate.core.profile import parse_profile
from primagate.schema.errors import BadRequestError
from primagate.schema.profile import Profile


class FileProfileStore:
    """The latest Profile, kept in one JSON file so it outlives the process.

    The file holds exactly `Profile.to_dict()`, versioned by its own `schema_version`. Content
    that cannot be read as a Profile is a miss, so the next write simply replaces it. An OS error
    is not, so a wrong path or missing permission is noticed rather than read as a cold start.
    """

    non_blocking = False

    def __init__(self, path: str | os.PathLike[str]) -> None:
        # Fixed now, so a later change of working directory cannot move the file.
        self._path = Path(_text_path(path)).absolute()
        # Serialises this store's own reads and writes: Windows refuses to replace a file that is
        # still open, so a read on another thread would otherwise make a write fail.
        self._lock = threading.Lock()

    def read(self) -> Profile | None:
        with self._lock:
            try:
                data = self._path.read_bytes()
            except FileNotFoundError:
                return None
        try:
            return parse_profile(json.loads(data.decode("utf-8")))
        # ValueError covers bad UTF-8, bad JSON, an over-long number and InvalidProfileError;
        # RecursionError is JSON nested too deeply to parse. All are content, not the disk.
        except (ValueError, RecursionError):
            return None

    def write(self, profile: Profile) -> None:
        # Escaped to ASCII, so any text a Profile can hold, even a lone surrogate, reads back.
        data = json.dumps(profile.to_dict(), sort_keys=True, indent=2, ensure_ascii=True) + "\n"
        encoded = data.encode("utf-8")
        with self._lock:
            directory = self._path.parent
            directory.mkdir(parents=True, exist_ok=True)
            # Same directory as the target, so the replace never crosses a file system and stays
            # atomic: a reader sees the old file or the new one, never part of either.
            fd, temp = tempfile.mkstemp(dir=directory, prefix=f".{self._path.name}.", suffix=".tmp")
            try:
                with os.fdopen(fd, "wb") as file:
                    file.write(encoded)
                    file.flush()
                    os.fsync(file.fileno())
                os.replace(temp, self._path)
            except BaseException:
                # Even an interrupt must not leave a stray temp file beside the Profile.
                with contextlib.suppress(OSError):
                    os.unlink(temp)
                raise


def _text_path(path: object) -> str:
    problem = "profile store path must be non-empty text or a path"
    if not isinstance(path, str | os.PathLike):
        raise BadRequestError(problem)
    try:
        text = os.fspath(path)
    except TypeError:
        raise BadRequestError(problem) from None
    if not isinstance(text, str) or not text:
        raise BadRequestError(problem)
    return text
