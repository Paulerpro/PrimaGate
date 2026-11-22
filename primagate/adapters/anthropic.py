from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC
from email.utils import parsedate_to_datetime
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

PROVIDER = "anthropic"
DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
KEY_ENV = "ANTHROPIC_API_KEY"

API_VERSION = "2023-06-01"
# The Messages API requires max_tokens, while the unified call treats it as optional.
DEFAULT_MAX_TOKENS = 4096

_STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}
_BY_STATUS: dict[int, type[errors.PrimaGateError]] = {
    400: errors.BadRequestError,
    401: errors.AuthenticationError,
    403: errors.PermissionDeniedError,
    404: errors.NotFoundError,
    408: errors.Timeout,
    413: errors.BadRequestError,
    429: errors.RateLimitError,
}
# Statuses that stand in for in-band stream errors, which arrive without an HTTP status.
_ERROR_TYPE_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "overloaded_error": 529,
}
_CONTEXT_PHRASES = ("prompt is too long", "context window")
# `claude-<family>-<major>...`. Anthropic's version 5 models reason before answering and reject
# `temperature` and `top_p`; the older `claude-<major>-...` names and version 4 and below do
# neither.
_FAMILY_MAJOR = re.compile(r"^claude-[a-z]+-(\d{1,2})(?:-|$)")


def _is_version_5_or_later(model: str) -> bool:
    match = _FAMILY_MAJOR.match(model)
    return match is not None and int(match.group(1)) >= 5


def _model_hints(model: str) -> frozenset[str]:
    return frozenset({"temperature", "top_p"}) if _is_version_5_or_later(model) else frozenset()


def _thinks_by_default(model: str) -> bool:
    return _is_version_5_or_later(model)


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
        rejected = _model_hints(request.model)
        for name in ("temperature", "top_p"):
            if getattr(request, name) is not None and name in rejected:
                raise errors.UnsupportedParamsError(
                    f"Anthropic model '{request.model}' does not support {name}",
                    llm_provider=PROVIDER,
                )
        system, messages = _split_messages(request.messages)
        body: dict[str, Any] = {
            "model": request.model,
            "max_tokens": DEFAULT_MAX_TOKENS if request.max_tokens is None else request.max_tokens,
            "messages": messages,
        }
        if request.stream:
            body["stream"] = True
        if system:
            body["system"] = system
        if request.tools:
            body["tools"] = [_tool(t) for t in request.tools]
        if request.response_format:
            body["output_config"] = {"format": _output_format(request.response_format)}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        if request.disable_thinking and _thinks_by_default(request.model):
            body["thinking"] = {"type": "disabled"}
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
            calls = [
                ToolCall(
                    id=b["id"],
                    function=FunctionCall(name=b["name"], arguments=json.dumps(b["input"])),
                )
                for b in data["content"]
                if b.get("type") == "tool_use"
            ]
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
                            role="assistant",
                            content="".join(texts) if texts else None,
                            tool_calls=calls or None,
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

    def new_stream_parser(self, headers: Mapping[str, str]) -> _StreamParser:
        return _StreamParser(self, _created(dict(headers)))

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
        return Capabilities(
            tools=True,
            structured_output=True,
            streaming=True,
            unsupported_hints=frozenset({"n", "seed"}),
            model_hints=_model_hints,
            thinks_by_default=_thinks_by_default,
        )


def _output_format(response_format: dict[str, Any]) -> dict[str, Any]:
    if response_format.get("type") != "json_schema":
        raise errors.UnsupportedParamsError(
            "Anthropic structured output needs a JSON schema; json_object has none to enforce",
            llm_provider=PROVIDER,
        )
    return {
        "type": "json_schema",
        "schema": _closed(response_format["json_schema"]["schema"]),
    }


def _closed(schema: Any) -> Any:
    """Anthropic rejects object schemas that do not set additionalProperties; pydantic omits it."""
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if (out.get("type") == "object" or "properties" in out) and "additionalProperties" not in out:
        out["additionalProperties"] = False
    for key in ("properties", "$defs", "definitions"):
        if isinstance(out.get(key), dict):
            out[key] = {name: _closed(sub) for name, sub in out[key].items()}
    for key in ("items", "additionalProperties"):
        if isinstance(out.get(key), dict):
            out[key] = _closed(out[key])
    for key in ("anyOf", "allOf", "oneOf", "prefixItems"):
        if isinstance(out.get(key), list):
            out[key] = [_closed(sub) for sub in out[key]]
    return out


def _tool(definition: dict[str, Any]) -> dict[str, Any]:
    function = definition["function"]
    out: dict[str, Any] = {"name": function["name"]}
    if function.get("description"):
        out["description"] = function["description"]
    out["input_schema"] = function.get("parameters") or {"type": "object"}
    return out


def _tool_use(call: ToolCall) -> dict[str, Any]:
    try:
        arguments = json.loads(call.function.arguments)
    except (ValueError, TypeError):
        arguments = None
    if not isinstance(arguments, dict):
        raise errors.BadRequestError(
            f"Arguments of tool call '{call.id}' must be a JSON object", llm_provider=PROVIDER
        )
    return {"type": "tool_use", "id": call.id, "name": call.function.name, "input": arguments}


def _assistant_content(message: Message) -> str | list[dict[str, Any]]:
    if not message.tool_calls:
        return message.content or ""
    blocks: list[dict[str, Any]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    blocks.extend(_tool_use(c) for c in message.tool_calls)
    return blocks


def _split_messages(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Anthropic takes system text as a separate field, not as a message role."""
    system: list[str] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            if message.content:
                system.append(message.content)
        elif message.role == "user":
            turns.append({"role": "user", "content": message.content or ""})
        elif message.role == "assistant":
            turns.append({"role": "assistant", "content": _assistant_content(message)})
        elif message.role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content or "",
            }
            # Every result for parallel calls must arrive in the single user turn that follows.
            last = turns[-1] if turns else None
            if last and last["role"] == "user" and _is_results(last["content"]):
                last["content"].append(result)
            else:
                turns.append({"role": "user", "content": [result]})
        else:
            raise errors.BadRequestError(
                f"Unsupported message role '{message.role}' for Anthropic", llm_provider=PROVIDER
            )
    return "\n\n".join(system), turns


def _is_results(content: Any) -> bool:
    return isinstance(content, list) and all(b["type"] == "tool_result" for b in content)


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


class _StreamParser:
    def __init__(self, adapter: AnthropicAdapter, created: int) -> None:
        self._adapter = adapter
        self._created = created
        self._id = ""
        self._model = ""
        self._input_tokens = 0
        # Anthropic numbers every content block; callers see tool calls numbered from zero.
        self._tool_index = -1
        self._tool_of_block: dict[int, int] = {}
        # message_start carries nothing to hand over, so the role waits for the first real chunk.
        self._role: str | None = None
        # Tool blocks whose arguments arrived; a call with none still needs a valid "{}".
        self._tools_with_arguments: set[int] = set()
        self._final_sent = False
        self._done = False

    def feed(self, event: RawEvent) -> list[ChatCompletionChunk]:
        if not event.data.strip():
            return []
        try:
            data = json.loads(event.data)
            kind = data["type"]
            if kind == "error":
                raise self._in_band_error(event.data, data)
            if kind == "message_start":
                message = data["message"]
                self._id, self._model = message["id"], message["model"]
                self._input_tokens = message["usage"]["input_tokens"]
                self._role = message.get("role", "assistant")
                return []
            if kind == "content_block_start":
                return self._block_start(data)
            if kind == "content_block_delta":
                return self._block_delta(data)
            if kind == "message_delta":
                return self._message_delta(data)
            if kind == "content_block_stop":
                return self._block_stop(data)
            if kind == "message_stop":
                if not self._final_sent:
                    raise errors.APIError(
                        "Anthropic stream stopped without a stop reason", llm_provider=PROVIDER
                    )
                self._done = True
            return []
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise errors.APIError(
                "Unparseable stream event from Anthropic", llm_provider=PROVIDER
            ) from exc

    def finish(self) -> list[ChatCompletionChunk]:
        if not self._done:
            raise errors.APIError(
                "Anthropic stream ended before message_stop", llm_provider=PROVIDER
            )
        return []

    def _in_band_error(self, raw: str, data: dict[str, Any]) -> errors.PrimaGateError:
        status = _ERROR_TYPE_STATUS.get(data["error"].get("type"), 500)
        return self._adapter.map_error(RawOutcome(status=status, headers={}, body=raw.encode()))

    def _block_start(self, data: dict[str, Any]) -> list[ChatCompletionChunk]:
        block = data["content_block"]
        if block["type"] != "tool_use":
            return []
        self._tool_index += 1
        self._tool_of_block[data["index"]] = self._tool_index
        call = ToolCallDelta(
            index=self._tool_index,
            id=block["id"],
            type="function",
            function=FunctionDelta(name=block["name"], arguments=""),
        )
        return [self._chunk(Delta(tool_calls=[call]))]

    def _block_delta(self, data: dict[str, Any]) -> list[ChatCompletionChunk]:
        delta = data["delta"]
        if delta["type"] == "text_delta" and delta["text"]:
            return [self._chunk(Delta(content=delta["text"]))]
        if delta["type"] == "input_json_delta" and delta["partial_json"]:
            self._tools_with_arguments.add(self._tool_of_block[data["index"]])
            call = ToolCallDelta(
                index=self._tool_of_block[data["index"]],
                function=FunctionDelta(arguments=delta["partial_json"]),
            )
            return [self._chunk(Delta(tool_calls=[call]))]
        return []

    def _block_stop(self, data: dict[str, Any]) -> list[ChatCompletionChunk]:
        tool = self._tool_of_block.get(data["index"])
        if tool is None or tool in self._tools_with_arguments:
            return []
        call = ToolCallDelta(index=tool, function=FunctionDelta(arguments="{}"))
        return [self._chunk(Delta(tool_calls=[call]))]

    def _message_delta(self, data: dict[str, Any]) -> list[ChatCompletionChunk]:
        reason = data["delta"].get("stop_reason")
        usage = data["usage"]
        prompt = usage.get("input_tokens", self._input_tokens)
        completion = usage["output_tokens"]
        self._final_sent = True
        return [
            self._chunk(
                Delta(),
                finish_reason=_STOP_REASONS.get(reason, reason),
                usage=Usage(
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=prompt + completion,
                ),
            )
        ]

    def _chunk(
        self, delta: Delta, *, finish_reason: str | None = None, usage: Usage | None = None
    ) -> ChatCompletionChunk:
        delta.role, self._role = self._role, None
        return ChatCompletionChunk(
            id=self._id,
            created=self._created,
            model=self._model,
            choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
            usage=usage,
        )
