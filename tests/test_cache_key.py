import dataclasses
from typing import Any

import pytest

from primagate.core.cache import cache_key
from primagate.core.steps import DropPolicy
from primagate.schema.types import ChatCompletionRequest, Message

TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}
SCHEMA = {"type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "object"}}}


def _request(**changes: Any) -> ChatCompletionRequest:
    base = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="hello")],
        temperature=0.2,
        tools=[TOOL],
        response_format=SCHEMA,
    )
    return dataclasses.replace(base, **changes)


def test_the_key_is_a_stable_sha256_hex_digest() -> None:
    key = cache_key(_request(), DropPolicy())
    assert key == cache_key(_request(), DropPolicy())
    assert len(key) == 64 and int(key, 16) >= 0


@pytest.mark.parametrize(
    "change",
    [
        {"model": "openai/gpt-4o"},
        {"model": "gpt-4o-mini"},
        {"messages": [Message(role="user", content="hello!")]},
        {"messages": [Message(role="system", content="hello")]},
        {"temperature": 0.3},
        {"top_p": 0.9},
        {"max_tokens": 10},
        {"stop": ["x"]},
        {"n": 2},
        {"seed": 7},
        {"tools": None},
        {"tools": [{**TOOL, "function": {"name": "g", "parameters": {"type": "object"}}}]},
        {"response_format": None},
        {"response_format": {"type": "json_object"}},
    ],
)
def test_changing_any_field_changes_the_key(change: dict[str, Any]) -> None:
    assert cache_key(_request(**change), DropPolicy()) != cache_key(_request(), DropPolicy())


def test_the_effective_drop_params_are_part_of_the_key() -> None:
    keys = {
        cache_key(_request(), DropPolicy()),
        cache_key(_request(), DropPolicy(hints=True)),
        cache_key(_request(), DropPolicy(features=frozenset({"tools"}))),
        cache_key(_request(), DropPolicy(features=frozenset({"tools", "image"}))),
    }
    assert len(keys) == 4


def test_the_order_of_dropped_features_and_of_dict_keys_does_not_matter() -> None:
    one = DropPolicy(features=frozenset({"tools", "image"}))
    two = DropPolicy(features=frozenset({"image", "tools"}))
    assert cache_key(_request(), one) == cache_key(_request(), two)
    reordered = {"type": "json_schema", "json_schema": {"schema": {"type": "object"}, "name": "s"}}
    assert cache_key(_request(response_format=reordered), one) == cache_key(_request(), one)


def test_every_request_field_is_in_the_key_or_deliberately_left_out() -> None:
    """A new request field that changes the answer must not be forgotten by the Cache."""
    request = _request()
    left_out = {"stream"}
    for field in dataclasses.fields(ChatCompletionRequest):
        if field.name in left_out:
            continue
        value = getattr(request, field.name)
        changed = {
            "model": "other-model",
            "messages": [Message(role="user", content="other")],
            "tools": None,
            "response_format": None,
        }.get(field.name, 7 if value is None else value + 1 if isinstance(value, float) else value)
        if changed == value:
            changed = [] if isinstance(value, list) else "other"
        assert cache_key(_request(**{field.name: changed}), DropPolicy()) != cache_key(
            request, DropPolicy()
        ), field.name


def test_a_request_json_cannot_encode_has_no_key() -> None:
    odd = [{"type": "function", "function": {"name": "f", "parameters": {"x": object()}}}]
    assert cache_key(_request(tools=odd), DropPolicy()) is None


def test_streaming_is_not_part_of_the_key() -> None:
    assert cache_key(_request(stream=True), DropPolicy()) == cache_key(_request(), DropPolicy())


def test_the_salt_changes_every_key_and_the_same_salt_repeats_it() -> None:
    plain = cache_key(_request(), DropPolicy())
    one = cache_key(_request(), DropPolicy(), b"salt-one")
    two = cache_key(_request(), DropPolicy(), b"salt-two")
    assert one == cache_key(_request(), DropPolicy(), b"salt-one")
    assert len({plain, one, two}) == 3
    assert one is not None and len(one) == 64
