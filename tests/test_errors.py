import pytest

from primagate.schema import errors

SUBCLASSES = [
    "Timeout",
    "RateLimitError",
    "ServiceUnavailableError",
    "APIError",
    "APIConnectionError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "UnsupportedParamsError",
    "ImageFetchError",
]


def test_rate_limit_error_attributes() -> None:
    err = errors.RateLimitError("m", llm_provider="openai", status_code=429)
    assert err.status_code == 429
    assert err.message == "m"
    assert err.llm_provider == "openai"
    assert isinstance(err, errors.PrimaGateError)
    assert "m" in str(err)


@pytest.mark.parametrize("name", SUBCLASSES)
def test_every_subclass_carries_attrs(name: str) -> None:
    cls = getattr(errors, name)
    assert issubclass(cls, errors.PrimaGateError)
    err = cls("boom", llm_provider="anthropic", status_code=500)
    assert (err.message, err.llm_provider, err.status_code) == ("boom", "anthropic", 500)

