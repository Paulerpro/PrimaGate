"""Live: the response cache, event delivery and payload retention on real replies."""

import logging
import re
import uuid

import pytest

from primagate import CallSucceeded, DispatchStarted
from primagate.core import prices

from .support import ASK, Live, LiveCaller

pytestmark = pytest.mark.live

ISO_UTC = re.compile(r"\d{4}-\d\d-\d\dT[\d:.]+(Z|\+00:00)")


def _tagged() -> tuple[str, list[dict[str, str]]]:
    # A fresh tag the model is asked to repeat, so it marks both the question and the reply.
    tag = f"LIVE{uuid.uuid4().hex[:10].upper()}"
    ask = f"Reply with exactly this code and nothing else: {tag}"
    return tag, [{"role": "user", "content": ask}]


def _logged(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "primagate"]


@pytest.mark.fr("FR31", "FR32")
def test_a_repeat_is_served_from_memory_while_opt_outs_and_streams_are_sent(
    call: LiveCaller, live: Live
) -> None:
    gate = live.gate()
    first = call(gate, model=live.model, messages=ASK)
    second = call(gate, model=live.model, messages=ASK)
    assert first.metadata is not None and first.metadata.cache_hit is False
    assert second.metadata is not None and second.metadata.cache_hit is True
    assert second.choices == first.choices and second.id == first.id
    assert live.events.types.count("dispatch_started") == 1
    assert "cache_hit" in live.events.types

    skipped = call(gate, model=live.model, messages=ASK, cache=False)
    assert skipped.metadata is not None and skipped.metadata.cache_hit is False
    assert live.events.types.count("dispatch_started") == 2

    chunks = call.stream(gate, model=live.model, messages=ASK)
    final = chunks[-1].metadata
    assert final is not None and final.cache_hit is False
    assert live.events.types.count("dispatch_started") == 3


@pytest.mark.fr("FR37", "FR38")
def test_events_reach_the_handler_and_the_log_without_the_conversation(
    call: LiveCaller, live: Live, caplog: pytest.LogCaptureFixture
) -> None:
    tag, messages = _tagged()
    with caplog.at_level(logging.INFO, logger="primagate"):
        call(live.gate(), model=live.model, messages=messages)
    seen = live.events.seen
    # A model missing from the price map also reports that its cost is unknown.
    unpriced = prices.lookup(prices.BUNDLED, live.provider, live.model) is None
    assert ("cost_unknown" in live.events.types) == unpriced
    story = [t for t in live.events.types if t != "cost_unknown"]
    assert story == ["dispatch_started", "call_succeeded"]
    assert len({e.request_id for e in seen}) == 1 and seen[0].request_id.startswith("req-")
    assert all(ISO_UTC.fullmatch(e.timestamp) for e in seen)
    lines = _logged(caplog)
    assert [line.split()[0] for line in lines] == live.events.types

    started, done = live.events.of(DispatchStarted), live.events.of(CallSucceeded)
    assert started[0].messages is None and done[0].reply_messages is None
    assert tag not in repr(seen) + "\n".join(lines)


@pytest.mark.fr("FR38")
def test_opting_in_keeps_both_sides_of_the_conversation(
    call: LiveCaller, live: Live, caplog: pytest.LogCaptureFixture
) -> None:
    tag, messages = _tagged()
    with caplog.at_level(logging.INFO, logger="primagate"):
        result = call(live.gate(payload_retention=True), model=live.model, messages=messages)
    (started,) = live.events.of(DispatchStarted)
    (done,) = live.events.of(CallSucceeded)
    assert started.messages == tuple(messages)
    assert done.reply_messages is not None
    assert done.reply_messages[0]["content"] == result.choices[0].message.content
    lines = _logged(caplog)
    assert any(line.startswith("dispatch_started") and tag in line for line in lines)
