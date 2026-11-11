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
    CacheHit,
    CallFailed,
    CallSucceeded,
    CostUnknown,
    DecisionEvent,
    DispatchFailed,
    DispatchStarted,
    Event,
    LowConfidenceDowngrade,
    PremiumRoutingSelected,
    RetryScheduled,
    RoutingDecision,
)
from primagate.schema.policy import TIER_ROUTING_ALIAS
from primagate.schema.tiers import UNKNOWN, Unknown

__version__ = "0.1.0"

# Read on every call by the module-level functions; an instance has its own `drop_params`.
drop_params: bool | list[str] | None = None

__all__ = [
    "TIER_ROUTING_ALIAS",
    "UNKNOWN",
    "APIConnectionError",
    "APIError",
    "AttemptSkipped",
    "AttemptsExhaustedError",
    "AuthenticationError",
    "BadRequestError",
    "CacheHit",
    "CallFailed",
    "CallSucceeded",
    "CostUnknown",
    "ContentPolicyViolationError",
    "ContextWindowExceededError",
    "DecisionEvent",
    "DispatchFailed",
    "DispatchStarted",
    "Event",
    "ImageFetchError",
    "LowConfidenceDowngrade",
    "NotFoundError",
    "PermissionDeniedError",
    "PremiumRoutingSelected",
    "PrimaGate",
    "PrimaGateError",
    "RateLimitError",
    "RetryScheduled",
    "RoutingDecision",
    "ServiceUnavailableError",
    "Timeout",
    "Unknown",
    "UnsupportedParamsError",
    "__version__",
    "acompletion",
    "completion",
]
