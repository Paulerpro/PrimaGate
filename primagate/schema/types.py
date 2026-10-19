from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any


def _dump(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        keep_null = value.keep_null() if isinstance(value, _Serializable) else ()
        return {
            f.name: _dump(getattr(value, f.name))
            for f in fields(value)
            if getattr(value, f.name) is not None or f.name in keep_null
        }
    if isinstance(value, list):
        return [_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _dump(v) for k, v in value.items()}
    return value


class _Serializable:
    """Mirrors the OpenAI wire shape: unset (None) fields are omitted."""

    def keep_null(self) -> tuple[str, ...]:
        """Fields sent as an explicit null instead of being omitted."""
        return ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _dump(self)
        return out

    def model_dump(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass
class FunctionCall(_Serializable):
    name: str
    # A JSON object encoded as a string, as OpenAI sends it; adapters convert for other providers.
    arguments: str


@dataclass
class ToolCall(_Serializable):
    id: str
    function: FunctionCall
    type: str = "function"


@dataclass
class Message(_Serializable):
    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def keep_null(self) -> tuple[str, ...]:
        # OpenAI rejects tool-call and tool-result messages that lack the content key.
        return ("content",) if self.tool_calls or self.role == "tool" else ()


@dataclass
class ChatCompletionRequest(_Serializable):
    model: str
    messages: list[Message]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    n: int | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None


@dataclass
class Usage(_Serializable):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Choice(_Serializable):
    index: int
    message: Message
    finish_reason: str | None = None


@dataclass
class Metadata(_Serializable):
    """What the SDK did to a call beyond relaying it; absent when there is nothing to report."""

    dropped_params: list[str] = field(default_factory=list)


@dataclass
class ChatCompletion(_Serializable):
    id: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    object: str = "chat.completion"
    metadata: Metadata | None = None


@dataclass
class FunctionDelta(_Serializable):
    name: str | None = None
    arguments: str | None = None


@dataclass
class ToolCallDelta(_Serializable):
    # Position among the response's tool calls; later deltas for one call share it.
    index: int
    id: str | None = None
    type: str | None = None
    function: FunctionDelta | None = None


@dataclass
class Delta(_Serializable):
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCallDelta] | None = None


@dataclass
class ChunkChoice(_Serializable):
    index: int
    delta: Delta
    finish_reason: str | None = None


@dataclass
class ChatCompletionChunk(_Serializable):
    id: str
    created: int
    model: str
    choices: list[ChunkChoice]
    usage: Usage | None = None
    object: str = "chat.completion.chunk"
    metadata: Metadata | None = None
