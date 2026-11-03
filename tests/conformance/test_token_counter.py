import asyncio
import logging
import sys
import time
import types
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller
from test_attempt_chain import DATE, SSE, _json_ok, _openai_chunk
from test_events import Recorder
from test_fallback_run import CLAUDE, GPT, _fallbacks
from test_retry_routing import MSGS, PRIMARY, Run, _status

from primagate import PrimaGate
from primagate.schema import errors

OPENAI, ANTHROPIC = PRIMARY["openai"], PRIMARY["anthropic"]


def _no_usage(provider: str, stream: bool) -> httpx2.Response:
    body = _json_ok(provider)
    body.pop("usage")
    body["choices"][0]["message"]["content"] = "ok then"
    return httpx2.Response(200, json=body, headers=DATE)


def _with_usage(provider: str, stream: bool) -> httpx2.Response:
    return httpx2.Response(200, json=_json_ok(provider), headers=DATE)


class FakeEncoding:
    def __init__(self, name: str) -> None:
        self.name = name

    def encode(self, text: str, *, disallowed_special: Any = "all") -> list[int]:
        assert disallowed_special == (), "special tokens in user text must not raise"
        return list(range(len(text.split())))


class FakeTiktoken(types.ModuleType):
    def __init__(self, *, fail_to_load: bool = False) -> None:
        super().__init__("tiktoken")
        self.fail_to_load = fail_to_load
        self.asked: list[str] = []

    def encoding_for_model(self, model: str) -> FakeEncoding:
        if model != "gpt-4o":
            raise KeyError(model)
        return self.get_encoding("o200k_base")

    def get_encoding(self, name: str) -> FakeEncoding:
        self.asked.append(name)
        if self.fail_to_load:
            raise OSError("could not fetch the vocabulary")
        return FakeEncoding(name)


def _counts(result: Any) -> tuple[int, int, bool]:
    counts = result.metadata.token_counts
    assert counts is not None
    return counts.prompt_tokens, counts.completion_tokens, counts.heuristic


def test_with_tiktoken_installed_counts_come_from_it(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeTiktoken()
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    result = call(Run({OPENAI: [_no_usage]}).gate, model="gpt-4o", messages=MSGS)
    assert _counts(result) == (1, 2, False)
    assert result.metadata.cost is not None


def test_a_model_tiktoken_does_not_know_uses_the_default_encoding(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeTiktoken()
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    result = call(Run({OPENAI: [_no_usage]}).gate, model="mystery-1", messages=MSGS)
    assert _counts(result)[2] is False and fake.asked == ["o200k_base"]


def test_without_tiktoken_the_heuristic_counts_and_says_so(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    result = call(Run({OPENAI: [_no_usage]}).gate, model="gpt-4o", messages=MSGS)
    assert _counts(result) == (1, 2, True)
    assert result.to_dict()["metadata"]["token_counts"]["heuristic"] is True


def test_a_failed_load_falls_back_once_and_is_not_retried(
    call: Caller, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = FakeTiktoken(fail_to_load=True)
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    run = Run({OPENAI: [_no_usage]})
    with caplog.at_level(logging.WARNING, logger="primagate"):
        first = call(run.gate, model="gpt-4o", messages=MSGS)
        # A different request, so the Cache does not answer it and it is counted again.
        second = call(run.gate, model="gpt-4o", messages=[{"role": "user", "content": "again"}])
    assert _counts(first)[2] is True and _counts(second)[2] is True
    assert fake.asked == ["o200k_base"], "the failed load is not attempted again"
    warned = [r for r in caplog.records if r.name == "primagate" and r.levelno >= logging.WARNING]
    assert len(warned) == 1 and "tiktoken" in warned[0].getMessage()


def test_text_that_looks_like_a_special_token_is_counted_not_refused(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", FakeTiktoken())
    result = call(
        Run({OPENAI: [_no_usage]}).gate,
        model="gpt-4o",
        messages=[{"role": "user", "content": "<|endoftext|> hello"}],
    )
    assert _counts(result)[2] is False


def test_a_caller_counter_replaces_the_default(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeTiktoken()
    monkeypatch.setitem(sys.modules, "tiktoken", fake)
    seen: list[tuple[str, str]] = []

    def counter(model: str, text: str) -> int:
        seen.append((model, text))
        return 11

    run = Run({OPENAI: [_no_usage]}, token_counter=counter)
    result = call(run.gate, model="gpt-4o", messages=MSGS)
    assert _counts(result) == (11, 11, False)
    assert seen == [("gpt-4o", "hi"), ("gpt-4o", "ok then")]
    assert fake.asked == []


@pytest.mark.parametrize("bad", ["raises", -1, 2.5, True, "7"])
def test_a_bad_counter_falls_back_for_that_call_and_never_raises(
    call: Caller, caplog: pytest.LogCaptureFixture, bad: Any
) -> None:
    def counter(model: str, text: str) -> Any:
        if bad == "raises":
            raise RuntimeError("counter bug")
        return bad

    run = Run({OPENAI: [_no_usage]}, token_counter=counter)
    with caplog.at_level(logging.WARNING, logger="primagate"):
        result = call(run.gate, model="gpt-4o", messages=MSGS)
    assert _counts(result) == (1, 2, True)
    assert any(r.levelno == logging.WARNING for r in caplog.records if r.name == "primagate")


def test_provider_usage_means_the_counter_is_not_asked(call: Caller) -> None:
    asked: list[str] = []

    def counter(model: str, text: str) -> int:
        asked.append(text)
        return 99

    run = Run({OPENAI: [_with_usage]}, token_counter=counter)
    result = call(run.gate, model="gpt-4o", messages=MSGS)
    assert asked == [] and _counts(result)[2] is False


def test_a_counter_must_be_a_plain_callable() -> None:
    async def slow(model: str, text: str) -> int:
        return 1

    with pytest.raises(errors.BadRequestError, match="token_counter"):
        PrimaGate(token_counter="nope")  # type: ignore[arg-type]
    with pytest.raises(errors.BadRequestError, match="token_counter"):
        PrimaGate(token_counter=slow)  # type: ignore[arg-type]


async def test_a_slow_counter_does_not_block_the_event_loop() -> None:
    def slow(model: str, text: str) -> int:
        time.sleep(0.15)
        return 1

    run = Run({OPENAI: [_no_usage]}, token_counter=slow)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        await run.gate.acompletion(model="gpt-4o", messages=MSGS)
    finally:
        task.cancel()
    assert ticks >= 5, ticks


def test_a_stream_that_reports_no_usage_is_left_uncounted(stream_call: StreamCaller) -> None:
    def no_usage_stream(provider: str, stream: bool) -> httpx2.Response:
        body = _openai_chunk({"content": "ok"}) + _openai_chunk({}, "stop") + "data: [DONE]\n\n"
        return httpx2.Response(200, content=body.encode(), headers=SSE)

    result = stream_call(Run({OPENAI: [no_usage_stream]}).gate, model="gpt-4o", messages=MSGS)
    assert result.read_error is None and result.chunks
    final = result.chunks[-1]
    assert final.metadata is not None and final.metadata.token_counts is None


def test_a_counted_fallback_answer_keeps_its_chain_and_is_priced_as_the_fallback(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    handler = Recorder()
    run = Run(
        {ANTHROPIC: [_status(503)], OPENAI: [_no_usage]},
        event_handler=handler,
        **_fallbacks(GPT),
    )
    result = call(run.gate, model=CLAUDE, messages=MSGS)
    meta = result.metadata
    assert meta is not None and meta.cost is not None
    assert [a.model for a in meta.attempt_chain] == [f"anthropic/{CLAUDE}", f"openai/{GPT}"]
    assert meta.attempt_chain[0].dispatches[0].error is not None
    # One prompt token and two completion tokens at gpt-4o's bundled rates.
    assert meta.cost.total == pytest.approx((1 * 2.5 + 2 * 10) / 1e6)
    done = handler.seen[-1]
    assert done.event_type == "call_succeeded" and done.model == f"openai/{GPT}"  # type: ignore[attr-defined]
    assert done.attempts == 2  # type: ignore[attr-defined]


def test_a_counted_response_for_an_unpriced_model_is_flagged_and_warned(
    call: Caller, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    handler = Recorder()
    run = Run({OPENAI: [_no_usage]}, event_handler=handler)
    result = call(run.gate, model="mystery-1", messages=MSGS)
    meta = result.metadata
    assert meta is not None and meta.token_counts is not None
    assert meta.unknown_model is True and meta.cost is None
    assert handler.types == ["dispatch_started", "cost_unknown", "call_succeeded"]
