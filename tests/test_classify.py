import pytest

from primagate.core.classify import Fate, classify
from primagate.schema import errors

RETRY = [
    errors.Timeout,
    errors.RateLimitError,
    errors.ServiceUnavailableError,
    errors.APIError,
    errors.APIConnectionError,
]
ADVANCE = [
    errors.ContextWindowExceededError,
    errors.ContentPolicyViolationError,
    errors.AuthenticationError,
    errors.PermissionDeniedError,
    errors.NotFoundError,
]
RAISE = [errors.BadRequestError, errors.UnsupportedParamsError, errors.ImageFetchError]


@pytest.mark.parametrize("cls", RETRY)
def test_transient_failures_are_retried(cls: type[errors.PrimaGateError]) -> None:
    assert classify(cls("x")) is Fate.RETRY


@pytest.mark.parametrize("cls", ADVANCE)
def test_failures_another_endpoint_may_avoid_advance(cls: type[errors.PrimaGateError]) -> None:
    assert classify(cls("x")) is Fate.ADVANCE


@pytest.mark.parametrize("cls", RAISE)
def test_caller_faults_stop_the_chain(cls: type[errors.PrimaGateError]) -> None:
    assert classify(cls("x")) is Fate.RAISE


def test_every_error_class_has_a_stated_fate() -> None:
    named = {*RETRY, *ADVANCE, *RAISE}
    concrete = {
        c for c in vars(errors).values() if isinstance(c, type) and c.__module__ == errors.__name__
    }
    assert concrete - named == {errors.PrimaGateError}


def test_an_unclassified_error_stops_the_chain() -> None:
    assert classify(errors.PrimaGateError("x")) is Fate.RAISE


def test_a_subclass_follows_its_parent() -> None:
    class Throttled(errors.RateLimitError):
        pass

    assert classify(Throttled("x")) is Fate.RETRY
