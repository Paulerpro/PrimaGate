from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any


def _dump(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _dump(getattr(value, f.name))
            for f in fields(value)
            if getattr(value, f.name) is not None
        }
    if isinstance(value, list):
        return [_dump(v) for v in value]
    if isinstance(value, dict):
        return {k: _dump(v) for k, v in value.items()}
    return value


class _Serializable:
    """Mirrors the OpenAI wire shape: unset (None) fields are omitted."""

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = _dump(self)
        return out

    def model_dump(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass
class Message(_Serializable):
    role: str
    content: str | None = None
    name: str | None = None


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
class ChatCompletion(_Serializable):
    id: str
    created: int
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    object: str = "chat.completion"


@dataclass
class Delta(_Serializable):
    role: str | None = None
    content: str | None = None


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
