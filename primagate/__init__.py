from primagate.client import PrimaGate, acompletion, completion
from primagate.schema.errors import (
    APIConnectionError,
    APIError,
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

__version__ = "0.1.0"

# Read on every call by the module-level functions; an instance has its own `drop_params`.
drop_params: bool | list[str] | None = None

__all__ = [
    "APIConnectionError",
    "APIError",
    "AuthenticationError",
    "BadRequestError",
    "ContentPolicyViolationError",
    "ContextWindowExceededError",
    "ImageFetchError",
    "NotFoundError",
    "PermissionDeniedError",
    "PrimaGate",
    "PrimaGateError",
    "RateLimitError",
    "ServiceUnavailableError",
    "Timeout",
    "UnsupportedParamsError",
    "__version__",
    "acompletion",
    "completion",
]
