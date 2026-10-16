from __future__ import annotations

import json
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Any

from primagate.adapters._common import error_details
from primagate.schema import errors
from primagate.schema.types import (
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    Message,
    Usage,
)
from primagate.schema.wire import Capabilities, ProviderRequest, RawOutcome

PROVIDER = "anthropic"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
KEY_ENV = "ANTHROPIC_API_KEY"

API_VERSION = "2023-06-01"
# The Messages API requires max_tokens, while the unified call treats it as optional.
DEFAULT_MAX_TOKENS = 4096

_STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length"}
_BY_STATUS: dict[int, type[errors.PrimaGateError]] = {
    400: errors.BadRequestError,
    401: errors.AuthenticationError,
    403: errors.PermissionDeniedError,
    404: errors.NotFoundError,
    408: errors.Timeout,
    413: errors.BadRequestError,
    429: errors.RateLimitError,
}
_CONTEXT_PHRASES = ("prompt is too long", "context window")


class AnthropicAdapter:
    def build_request(
        self, request: ChatCompletionRequest, *, api_key: str, base_url: str
    ) -> ProviderRequest:
        # There is no equivalent, and ignoring them would change what the caller asked for.
        if request.n not in (None, 1):
            raise errors.UnsupportedParamsError(
                "Anthropic does not support n greater than 1", llm_provider=PROVIDER
            )
        if request.seed is not None:
            raise errors.UnsupportedParamsError(
                "Anthropic does not support seed", llm_provider=PROVIDER
            )
        system, messages = _split_messages(request.messages)
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": DEFAULT_MAX_TOKENS if request.max_tokens is None else request.max_tokens,
            "messages": messages,
        }
        if system:
            body["system"] = system
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.top_p is not None:
            body["top_p"] = request.top_p
        if request.stop is not None:
            body["stop_sequences"] = (
                [request.stop] if isinstance(request.stop, str) else request.stop
            )
        return ProviderRequest(
            method="POST",
            url=f"{base_url.rstrip('/')}/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            json_body=body,
        )

    def parse_response(self, outcome: RawOutcome) -> ChatCompletion:
        try:
            data = json.loads(outcome.body)
            texts = [b["text"] for b in data["content"] if b.get("type") == "text"]
            reason = data.get("stop_reason")
            usage = data["usage"]
            prompt, completion = usage["input_tokens"], usage["output_tokens"]
            return ChatCompletion(
                id=data["id"],
                created=_created(outcome.headers),
                model=data["model"],
                choices=[
                    Choice(
                        index=0,
                        message=Message(
                            role="assistant", content="".join(texts) if texts else None
                        ),
                        finish_reason=_STOP_REASONS.get(reason, reason),
                    )
                ],
                usage=Usage(
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=prompt + completion,
                ),
            )
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise errors.APIError(
                "Unparseable response from Anthropic",
                llm_provider=PROVIDER,
                status_code=outcome.status,
            ) from exc

    def new_stream_parser(self) -> Any:
        raise NotImplementedError("Streaming is not supported yet")

    def map_error(self, outcome: RawOutcome) -> errors.PrimaGateError:
        if outcome.failure == "timeout":
            return errors.Timeout("Request timed out", llm_provider=PROVIDER)
        if outcome.failure is not None:
            return errors.APIConnectionError(
                "Connection to Anthropic failed", llm_provider=PROVIDER
            )

        message, _ = error_details(outcome.body, "type")
        status = outcome.status
        if status == 400 and any(p in message.lower() for p in _CONTEXT_PHRASES):
            cls: type[errors.PrimaGateError] = errors.ContextWindowExceededError
        else:
            cls = _BY_STATUS.get(status or 0, errors.APIError)
            if status is not None and status >= 500:
                cls = errors.ServiceUnavailableError
        return cls(message, llm_provider=PROVIDER, status_code=status)

    def capabilities(self) -> Capabilities:
        return Capabilities()


def _split_messages(messages: list[Message]) -> tuple[str, list[dict[str, str]]]:
    """Anthropic takes system text as a separate field, not as a message role."""
    system: list[str] = []
    turns: list[dict[str, str]] = []
    for message in messages:
        if message.role == "system":
            if message.content:
                system.append(message.content)
        elif message.role in ("user", "assistant"):
            turns.append({"role": message.role, "content": message.content or ""})
        else:
            raise errors.BadRequestError(
                f"Unsupported message role '{message.role}' for Anthropic", llm_provider=PROVIDER
            )
    return "\n\n".join(system), turns


def _created(headers: dict[str, str]) -> int:
    # The body has no timestamp and adapters cannot read a clock, so use the server's.
    for name, value in headers.items():
        if name.lower() == "date":
            try:
                moment = parsedate_to_datetime(value)
                # A "-0000" zone parses as naive, which timestamp() would read as local time.
                return int(moment.replace(tzinfo=moment.tzinfo or UTC).timestamp())
            except (TypeError, ValueError, OverflowError, OSError):
                return 0
    return 0
