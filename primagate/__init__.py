from primagate.client import PrimaGate, acompletion, completion
from primagate.schema.errors import (
    APIConnectionError,
    APIError,
    AttemptsExhaustedError,
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    ImageFetchError,
    NotFoundError,
    PermissionDeniedError,
    PrimaGateError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnsupportedParamsError,
)
from primagate.schema.events import (
    AttemptSkipped,
    CallFailed,
    CallSucceeded,
    CostUnknown,
    DispatchFailed,
    DispatchStarted,
    Event,
    RetryScheduled,
)

__version__ = "0.1.0"

# Read on every call by the module-level functions; an instance has its own `drop_params`.
drop_params: bool | list[str] | None = None

__all__ = [
    "APIConnectionError",
    "APIError",
    "AttemptSkipped",
    "AttemptsExhaustedError",
    "AuthenticationError",
    "BadRequestError",
    "CallFailed",
    "CallSucceeded",
    "CostUnknown",
    "ContentPolicyViolationError",
    "ContextWindowExceededError",
    "DispatchFailed",
    "DispatchStarted",
    "Event",
    "ImageFetchError",
    "NotFoundError",
    "PermissionDeniedError",
    "PrimaGate",
    "PrimaGateError",
    "RateLimitError",
    "RetryScheduled",
    "ServiceUnavailableError",
    "Timeout",
    "UnsupportedParamsError",
    "__version__",
    "acompletion",
    "completion",
]
