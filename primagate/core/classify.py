from __future__ import annotations

from enum import Enum

from primagate.schema import errors


class Fate(Enum):
    RETRY = "retry"
    ADVANCE = "advance"
    RAISE = "raise"


_RETRY = (
    errors.Timeout,
    errors.RateLimitError,
    errors.ServiceUnavailableError,
    errors.APIError,
    errors.APIConnectionError,
)
_ADVANCE = (
    errors.ContextWindowExceededError,
    errors.ContentPolicyViolationError,
    errors.AuthenticationError,
    errors.PermissionDeniedError,
    errors.NotFoundError,
)


def classify(error: errors.PrimaGateError) -> Fate:
    """Decide what a failure means for the rest of the chain.

    Anything not named here stops the chain, so a new error class never retries by accident.
    """
    if isinstance(error, _RETRY):
        return Fate.RETRY
    if isinstance(error, _ADVANCE):
        return Fate.ADVANCE
    return Fate.RAISE
