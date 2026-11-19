"""Live: one call shape, tools, structured output, streaming, and the counts and cost on each."""

import json
from typing import Any

import pytest

from primagate.core import prices

from .support import (
    ASK,
    ASK_TOOL,
    TOOLS,
    Live,
    LiveCaller,
    joined,
    recorded_reply,
    shape,
    stop_if_cut,
)

pytestmark = pytest.mark.live

FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "place",
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}
# Asks for the JSON object alone, so the answer starts with it and fits under the cap.
ASK_JSON = [
    {
        "role": "user",
        "content": "Reply with only the JSON object, no other text: the capital of Norway.",
    }
]
DONE = ("stop", "length")


@pytest.mark.fr("FR1", "FR13")
def test_a_plain_call_has_the_recorded_shape_and_one_direct_dispatch(
    call: LiveCaller, live: Live
) -> None:
    result = call(live.gate(), model=live.model, messages=ASK)
    choice = result.choices[0]
    assert choice.message.role == "assistant"
    assert isinstance(choice.message.content, str) and choice.message.content.strip()
    assert choice.finish_reason in DONE
    usage = result.usage
    assert usage is not None and usage.prompt_tokens > 0 and usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    # The real reply and a scripted one come back in the same unified shape.
    reference = recorded_reply(call, live.provider, live.model)
    assert shape(result.to_dict()) == shape(reference.to_dict())

    # With Provider Routing off, the call is one dispatch to the provider, never retried.
    assert result.metadata is not None
    (attempt,) = result.metadata.attempt_chain
    assert attempt.model == f"{live.provider}/{live.model}" and attempt.skipped is None
    (dispatch,) = attempt.dispatches
    assert (dispatch.endpoint, dispatch.retries, dispatch.error) == (live.provider, 0, None)
    assert live.events.types.count("dispatch_started") == 1


@pytest.mark.fr("FR2")
def test_a_tool_round_trip_completes(call: LiveCaller, live: Live) -> None:
    gate = live.gate()
    messages: list[Any] = list(ASK_TOOL)
    first = call(gate, model=live.model, messages=messages, tools=TOOLS)
    message = first.choices[0].message
    stop_if_cut(first.choices[0].finish_reason)
    assert first.choices[0].finish_reason == "tool_calls"
    assert message.tool_calls
    for tool_call in message.tool_calls:
        assert tool_call.function.name == "weather"
        assert isinstance(json.loads(tool_call.function.arguments)["city"], str)
    assert len({c.id for c in message.tool_calls}) == len(message.tool_calls)

    messages.append(message)
    for tool_call in message.tool_calls:
        messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": "cold, 2 C"})
    second = call(gate, model=live.model, messages=messages, tools=TOOLS)
    stop_if_cut(second.choices[0].finish_reason)
    assert second.choices[0].finish_reason == "stop"
    assert second.choices[0].message.content and not second.choices[0].message.tool_calls


@pytest.mark.fr("FR2")
def test_structured_output_is_json_for_the_schema(call: LiveCaller, live: Live) -> None:
    result = call(live.gate(), model=live.model, messages=ASK_JSON, response_format=FORMAT)
    stop_if_cut(result.choices[0].finish_reason)
    assert result.choices[0].finish_reason == "stop"
    parsed = json.loads(result.choices[0].message.content or "")
    assert isinstance(parsed, dict) and isinstance(parsed["city"], str)


@pytest.mark.fr("FR2", "FR33")
def test_a_text_stream_ends_with_finish_reason_usage_and_estimates(
    call: LiveCaller, live: Live
) -> None:
    chunks = call.stream(live.gate(), model=live.model, messages=ASK)
    assert chunks and joined(chunks).strip()
    first = next(c for c in chunks if c.choices)
    assert first.choices[0].delta.role == "assistant"
    last = chunks[-1]
    assert last.choices and last.choices[0].finish_reason in DONE
    assert last.usage is not None and last.usage.completion_tokens > 0
    assert all(c.usage is None and c.metadata is None for c in chunks[:-1])
    meta = last.metadata
    assert meta is not None and meta.cache_hit is False and meta.token_counts is not None
    assert meta.token_counts.total_tokens == last.usage.total_tokens


@pytest.mark.fr("FR2")
def test_streamed_tool_call_deltas_join_to_the_call(call: LiveCaller, live: Live) -> None:
    chunks = call.stream(live.gate(), model=live.model, messages=ASK_TOOL, tools=TOOLS)
    stop_if_cut(chunks[-1].choices[0].finish_reason if chunks and chunks[-1].choices else None)
    deltas = [d for c in chunks if c.choices for d in c.choices[0].delta.tool_calls or []]
    assert deltas
    index = deltas[0].index
    mine = [d for d in deltas if d.index == index]
    assert mine[0].id and mine[0].function is not None and mine[0].function.name == "weather"
    arguments = "".join(d.function.arguments or "" for d in mine if d.function)
    assert isinstance(json.loads(arguments)["city"], str)
    assert chunks[-1].choices and chunks[-1].choices[0].finish_reason == "tool_calls"


@pytest.mark.fr("FR33", "FR34")
def test_provider_usage_gives_the_counts_and_the_price_gives_the_cost(
    call: LiveCaller, live: Live
) -> None:
    asked: list[str] = []

    def counter(model: str, text: str) -> int:
        asked.append(model)
        return 1

    bundled = prices.lookup(prices.BUNDLED, live.provider, live.model)
    # A model missing from the bundled map is priced by the Caller, as a user would do.
    price = bundled or prices.Price(1.0, 5.0)
    overrides = None if bundled else {live.model: {"input": price.input, "output": price.output}}
    gate = live.gate(token_counter=counter, price_overrides=overrides)
    result = call(gate, model=live.model, messages=ASK)
    usage, meta = result.usage, result.metadata
    assert usage is not None and meta is not None
    # The provider reported usage, so the counter is never asked.
    assert asked == []
    counts = meta.token_counts
    assert counts is not None and counts.heuristic is False
    assert (counts.prompt_tokens, counts.completion_tokens, counts.total_tokens) == (
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
    )
    spent = (usage.prompt_tokens * price.input + usage.completion_tokens * price.output) / 1e6
    assert meta.cost is not None and meta.cost.total == pytest.approx(spent)
    assert meta.cost.currency == "USD" and meta.unknown_model is False
