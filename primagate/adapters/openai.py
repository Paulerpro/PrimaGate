from __future__ import annotations

import json
from typing import Any

from primagate.adapters._common import error_details
from primagate.schema import errors
from primagate.schema.types import (
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    Usage,
)
from primagate.schema.wire import Capabilities, ProviderRequest, RawOutcome

PROVIDER = "openai"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
KEY_ENV = "OPENAI_API_KEY"

_CONTEXT_CODES = {"context_length_exceeded"}
_POLICY_CODES = {"content_policy_violation", "content_filter"}
_BY_STATUS: dict[int, type[errors.PrimaGateError]] = {
    400: errors.BadRequestError,
    401: errors.AuthenticationError,
    403: errors.PermissionDeniedError,
    404: errors.NotFoundError,
    408: errors.Timeout,
    429: errors.RateLimitError,
}


class OpenAIAdapter:
    def build_request(
        self, request: ChatCompletionRequest, *, api_key: str, base_url: str
    ) -> ProviderRequest:
        return ProviderRequest(
            method="POST",
            url=f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json_body=request.to_dict(),
        )

    def parse_response(self, outcome: RawOutcome) -> ChatCompletion:
        try:
            data = json.loads(outcome.body)
            usage = data.get("usage")
            return ChatCompletion(
                id=data["id"],
                created=data["created"],
                model=data["model"],
                choices=[_choice(c) for c in data["choices"]],
                usage=_usage(usage) if usage else None,
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise errors.APIError(
                "Unparseable response from OpenAI",
                llm_provider=PROVIDER,
                status_code=outcome.status,
            ) from exc

    def new_stream_parser(self) -> Any:
        raise NotImplementedError("Streaming is not supported yet")

    def map_error(self, outcome: RawOutcome) -> errors.PrimaGateError:
        if outcome.failure == "timeout":
            return errors.Timeout("Request timed out", llm_provider=PROVIDER)
        if outcome.failure is not None:
            return errors.APIConnectionError("Connection to OpenAI failed", llm_provider=PROVIDER)

        message, code = error_details(outcome.body, "code")
        status = outcome.status
        if status == 400 and code in _CONTEXT_CODES:
            cls: type[errors.PrimaGateError] = errors.ContextWindowExceededError
        elif status == 400 and code in _POLICY_CODES:
            cls = errors.ContentPolicyViolationError
        else:
            cls = _BY_STATUS.get(status or 0, errors.APIError)
            if status is not None and status >= 500:
                cls = errors.ServiceUnavailableError
        return cls(message, llm_provider=PROVIDER, status_code=status)

    def capabilities(self) -> Capabilities:
        return Capabilities(tools=True)


def _choice(raw: dict[str, Any]) -> Choice:
    msg = raw["message"]
    calls = [
        ToolCall(
            id=c["id"],
            type=c.get("type", "function"),
            function=FunctionCall(name=c["function"]["name"], arguments=c["function"]["arguments"]),
        )
        for c in msg.get("tool_calls") or []
    ]
    return Choice(
        index=raw["index"],
        message=Message(role=msg["role"], content=msg.get("content"), tool_calls=calls or None),
        finish_reason=raw.get("finish_reason"),
    )


def _usage(raw: dict[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=raw["prompt_tokens"],
        completion_tokens=raw["completion_tokens"],
        total_tokens=raw["total_tokens"],
    )
