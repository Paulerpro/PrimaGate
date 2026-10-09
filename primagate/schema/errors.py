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
