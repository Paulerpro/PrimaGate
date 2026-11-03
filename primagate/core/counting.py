from __future__ import annotations

from collections.abc import Mapping

from primagate.schema.types import ChatCompletion, ChatCompletionRequest, Message

_CHARS_PER_TOKEN = 4


def heuristic_count(text: str) -> int:
    """A rough count, one token per four characters, for when no real tokenizer is available."""
    return -(-len(text) // _CHARS_PER_TOKEN)


def prompt_text(request: ChatCompletionRequest) -> str:
    return _join(request.messages)


def completion_text(response: ChatCompletion) -> str:
    return _join([choice.message for choice in response.choices])


def _join(messages: list[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        parts.extend(_content(message.content))
        for call in message.tool_calls or []:
            # Counting is best effort and must never fail a call, so odd values are skipped.
            parts += [
                s for s in (call.function.name, call.function.arguments) if isinstance(s, str)
            ]
    return "\n".join(parts)


def _content(content: object) -> list[str]:
    # Images are skipped: their cost depends on size and detail, which text cannot tell.
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [
            part["text"]
            for part in content
            if isinstance(part, Mapping)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        ]
    return []
