from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from primagate.adapters._common import error_details
from primagate.schema import errors
from primagate.schema.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChunkChoice,
    Delta,
    FunctionCall,
    FunctionDelta,
    Message,
    ToolCall,
    ToolCallDelta,
    Usage,
)
from primagate.schema.wire import Capabilities, ProviderRequest, RawEvent, RawOutcome

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
        body = request.to_dict()
        if request.stream:
            # Without this OpenAI sends no usage on a stream.
            body["stream_options"] = {"include_usage": True}
        return ProviderRequest(
            method="POST",
            url=f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json_body=body,
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

    def new_stream_parser(self, headers: Mapping[str, str]) -> _StreamParser:
        return _StreamParser(self)

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
        return Capabilities(tools=True, structured_output=True, streaming=True)


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


class _StreamParser:
    def __init__(self, adapter: OpenAIAdapter) -> None:
        self._adapter = adapter
        self._role: str | None = None
        # The finish chunk waits for the usage chunk that follows it, so the last chunk has both.
        self._held: ChatCompletionChunk | None = None
        self._done = False

    def feed(self, event: RawEvent) -> list[ChatCompletionChunk]:
        data = event.data.strip()
        if data == "[DONE]":
            self._done = True
            return self._release()
        try:
            raw = json.loads(data)
            if "error" in raw:
                raise self._adapter.map_error(
                    RawOutcome(status=None, headers={}, body=data.encode())
                )
            return self._chunk(raw)
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise errors.APIError(
                "Unparseable stream event from OpenAI", llm_provider=PROVIDER
            ) from exc

    def finish(self) -> list[ChatCompletionChunk]:
        if not self._done:
            raise errors.APIError("OpenAI stream ended before [DONE]", llm_provider=PROVIDER)
        return []

    def _release(self) -> list[ChatCompletionChunk]:
        held, self._held = self._held, None
        return [held] if held else []

    def _chunk(self, raw: dict[str, Any]) -> list[ChatCompletionChunk]:
        usage = _usage(raw["usage"]) if raw.get("usage") else None
        choices = [c for c in map(self._choice, raw.get("choices") or []) if c]
        out: list[ChatCompletionChunk] = []
        if usage and not choices and self._held:
            self._held.usage = usage
            return self._release()
        if not choices:
            return out
        chunk = ChatCompletionChunk(
            id=raw["id"], created=raw["created"], model=raw["model"], choices=choices, usage=usage
        )
        if any(c.finish_reason for c in choices) and usage is None:
            out = self._release()
            self._held = chunk
            return out
        return [*self._release(), chunk]

    def _choice(self, raw: dict[str, Any]) -> ChunkChoice | None:
        delta = raw.get("delta") or {}
        self._role = delta.get("role") or self._role
        calls = [
            ToolCallDelta(
                index=c["index"],
                id=c.get("id"),
                type=c.get("type"),
                function=(
                    FunctionDelta(
                        name=c["function"].get("name"), arguments=c["function"].get("arguments")
                    )
                    if c.get("function")
                    else None
                ),
            )
            for c in delta.get("tool_calls") or []
        ]
        content, finish = delta.get("content") or None, raw.get("finish_reason")
        if not (content or calls or finish):
            # Role-only events carry nothing to hand over; the role rides on the next chunk.
            return None
        role, self._role = self._role, None
        return ChunkChoice(
            index=raw["index"],
            delta=Delta(role=role, content=content, tool_calls=calls or None),
            finish_reason=finish,
        )
