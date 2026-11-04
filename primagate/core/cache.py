from __future__ import annotations

import hashlib
import hmac
import json

from primagate.core.steps import DropPolicy
from primagate.schema.types import ChatCompletionRequest

# Bumped when the set of hashed fields changes, so an old key can never match a new request.
_KEY_VERSION = 1


def cache_key(request: ChatCompletionRequest, policy: DropPolicy, salt: bytes = b"") -> str | None:
    """The identity of a request for the Cache: everything that can change the answer.

    The model is hashed as the caller wrote it, and `stream` is left out because streams never
    use the Cache. The effective `drop_params` is part of the key because it can change what is
    sent, so a call that allowed drops must not be answered with one that did not. A request
    holding something JSON cannot encode has no key, so it is never cached: guessing a text for
    it could give two different requests the same key.

    The hash is keyed with a secret the driver made for this instance, so a log line holding it
    cannot be used to test a guess at a short prompt. Core only receives that secret as data.
    """
    payload = {
        "v": _KEY_VERSION,
        "model": request.model,
        "messages": [message.to_dict() for message in request.messages],
        "tools": request.tools,
        "response_format": request.response_format,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stop": request.stop,
        "n": request.n,
        "seed": request.seed,
        "drop": {"hints": policy.hints, "features": sorted(policy.features)},
    }
    try:
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return hmac.new(salt, blob.encode("utf-8"), hashlib.sha256).hexdigest()
