from collections.abc import Iterable
from typing import Any, Self

from primagate.schema.types import Attempt


class PrimaGateError(Exception):
    def __init__(
        self,
        message: str,
        *,
        llm_provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.llm_provider = llm_provider
        self.status_code = status_code
        # Filled in by core once a dispatch has been made; empty for pre-dispatch errors.
        self.attempt_chain: list[Attempt] = []

    def redact(self, secrets: Iterable[str]) -> Self:
        # Providers sometimes echo the offending key back in their error text.
        for secret in secrets:
            if secret:
                self.message = self.message.replace(secret, "***")
        self.args = (self.message,)
        return self


class Timeout(PrimaGateError):
    pass


class RateLimitError(PrimaGateError):
    pass


class ServiceUnavailableError(PrimaGateError):
    pass


class APIError(PrimaGateError):
    pass


class APIConnectionError(PrimaGateError):
    pass


class ContextWindowExceededError(PrimaGateError):
    pass


class ContentPolicyViolationError(PrimaGateError):
    pass


class AuthenticationError(PrimaGateError):
    pass


class PermissionDeniedError(PrimaGateError):
    pass


class NotFoundError(PrimaGateError):
    pass


class BadRequestError(PrimaGateError):
    pass


class UnsupportedParamsError(PrimaGateError):
    pass


class ImageFetchError(PrimaGateError):
    pass


class CancelledError(PrimaGateError):
    """The call was stopped by its cancel token or its deadline.

    Never a provider failure, so nothing retries or falls back on it. An async call whose task
    was cancelled raises `asyncio.CancelledError` instead, as asyncio expects.
    """


class AttemptsExhaustedError(PrimaGateError):
    """Every Attempt failed (or none that could be made was left), so the chain is the story.

    `attempts` are the models that were actually dispatched, in order; `attempt_chain` also has
    the candidates passed over. `last_error` is the error that ended the chain and is also the
    `__cause__`, so tracebacks show it.
    """

    def __init__(
        self,
        message: str,
        *,
        attempt_chain: list[Attempt],
        last_error: PrimaGateError,
    ) -> None:
        super().__init__(
            message, llm_provider=last_error.llm_provider, status_code=last_error.status_code
        )
        self.attempt_chain = attempt_chain
        self.attempts = [row for row in attempt_chain if row.skipped is None]
        self.last_error = last_error
        self.__cause__ = last_error

    def __reduce__(self) -> tuple[Any, ...]:
        # Exception rebuilds from `args`, which would drop the keyword-only fields.
        return (_rebuild_exhausted, (self.message, self.attempt_chain, self.last_error))


def _rebuild_exhausted(
    message: str, attempt_chain: list[Attempt], last_error: PrimaGateError
) -> AttemptsExhaustedError:
    return AttemptsExhaustedError(message, attempt_chain=attempt_chain, last_error=last_error)
