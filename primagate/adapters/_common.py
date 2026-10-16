from __future__ import annotations

import json


def error_details(body: bytes, code_field: str) -> tuple[str, str | None]:
    """Pull (message, code) from a provider error body, whose code lives under `code_field`."""
    try:
        err = json.loads(body)["error"]
        message = err["message"]
        code = err.get(code_field)
        if isinstance(message, str) and message:
            return message, code if isinstance(code, str) else None
    except (ValueError, KeyError, TypeError, AttributeError):
        pass
    # Non-JSON bodies (gateway pages) are still worth surfacing, but bounded.
    return body.decode("utf-8", errors="replace")[:500] or "Unknown error", None
