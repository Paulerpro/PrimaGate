import asyncio
import json
from collections.abc import AsyncIterator, Iterable, Iterator
from typing import Any, cast

import httpx2
import pytest
from conftest import StreamCaller
from test_attempt_chain import SSE, Dropping, _stream_body
from test_cancellation import Clock, Stopper, _one_cancel_event, _paced, stand_in_clock
from test_failover_chain import NO_MEDIUM, D, F, _rows
from test_retry_routing import B_KEY, B_URL, Reply, _answer, _in_band_error, _status
from test_tier_failover import GPT, MINI, POLICY, _too_long
from test_tier_failover import Run as FailoverRun
from test_tier_routing import MSGS

from primagate import CancelledError
from primagate.schema import errors, events
from primagate.schema.policy import TIER_ROUTING_ALIAS

ALIAS = TIER_ROUTING_ALIAS
# How the shared stream bodies name their model, whichever model was asked.
STAMP = b'"model": "gpt-4o"'


class Restamped(httpx2.SyncByteStream, httpx2.AsyncByteStream):
    """A stream body renamed to the model it was sent to, with its failures and pacing kept."""

    def __init__(self, inner: Any, model: str) -> None:
        self._inner = inner
        self._stamp = f'"model": "{model}"'.encode()

    def __iter__(self) -> Iterator[bytes]:
        for part in cast(Iterable[bytes], self._inner):
            yield part.replace(STAMP, self._stamp)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for part in cast(AsyncIterator[bytes], self._inner):
            yield part.replace(STAMP, self._stamp)


class Run(FailoverRun):
    """Each streamed chunk names the model that sent it, so a chunk leaked from another
    Attempt cannot pass for the answering one."""

    def _handle(self, request: httpx2.Request) -> Any:
        response = super()._handle(request)
        body = json.loads(request.content)
        if not isinstance(response, httpx2.Response) or response.status_code != 200:
            return response
        if not body.get("stream"):
            return response
        return httpx2.Response(200, stream=Restamped(response.stream, body["model"]), headers=SSE)


def _from(chunks: list[Any], model: str) -> bool:
    return bool(chunks) and all(c.model == model for c in chunks)


def _drops(provider: str, stream: bool) -> httpx2.Response:
    """One chunk reaches the caller, then the connection drops."""
    return httpx2.Response(200, stream=Dropping(_stream_body(provider, full=False)), headers=SSE)


def _early(failure: str) -> Reply:
    """A failure before any chunk: a refused request, or an error as the stream's first event."""
    return _status(503) if failure == "status" else _in_band_error("boom")


EARLY = pytest.mark.parametrize("failure", ["status", "first_event"])


# --- streams that fail before their first chunk ---


@EARLY
@pytest.mark.fr("FR30")
def test_a_stream_whose_tier_fails_early_opens_on_medium(
    stream_call: StreamCaller, failure: str
) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_early(failure)]}, default_fallbacks=[F])
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.read_error is None
    assert run.models == [MINI, GPT] and len(run.analyzed) == 1
    final = result.chunks[-1]
    assert final.metadata is not None and final.metadata.tier == "MEDIUM"
    assert final.metadata.analyzer is not None
    assert _rows(final.metadata.attempt_chain) == [
        (f"openai/{MINI}", True),
        (f"openai/{GPT}", True),
    ]
    assert all(c.metadata is None for c in result.chunks[:-1])
    assert _from(result.chunks, GPT)
    (failed,) = run.of(events.DispatchFailed)
    assert failed.next == "next_attempt"


@EARLY
@pytest.mark.fr("FR7", "FR30")
def test_a_stream_whose_tier_and_medium_fail_early_comes_from_the_fallback(
    stream_call: StreamCaller, failure: str
) -> None:
    script = {MINI: [_early(failure)], GPT: [_early(failure)]}
    run = Run(["SMALL 0.9"], script, default_fallbacks=[F])
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.read_error is None
    assert run.models == [MINI, GPT, F] and len(run.analyzed) == 1
    final = result.chunks[-1]
    meta = final.metadata
    assert meta is not None and meta.tier is None
    assert meta.analyzer is not None and meta.analyzer.cost is not None
    assert [row.model for row in meta.attempt_chain] == [f"openai/{m}" for m in (MINI, GPT, F)]
    assert _from(result.chunks, F)


@pytest.mark.fr("FR30")
def test_a_skipped_medium_stays_in_the_final_chunks_chain(stream_call: StreamCaller) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_status(503)]}, policy=NO_MEDIUM, default_fallbacks=[F])
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.read_error is None
    assert run.models == [MINI, F] and _from(result.chunks, F)
    final = result.chunks[-1]
    assert final.metadata is not None and final.metadata.tier is None
    assert _rows(final.metadata.attempt_chain) == [
        (f"openai/{MINI}", True),
        (f"openai/{GPT}", False),
        (f"openai/{F}", True),
    ]
    (skip,) = run.of(events.AttemptSkipped)
    assert skip.reason == "MEDIUM is not in enabled_tiers"


@EARLY
@pytest.mark.fr("FR11", "FR30")
def test_a_stream_where_every_layer_fails_early_raises_the_aggregate_from_the_call(
    stream_call: StreamCaller, failure: str
) -> None:
    down = [_early(failure)]
    run = Run(["SMALL 0.9"], {MINI: down, GPT: down, F: down}, default_fallbacks=[F])
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert isinstance(result.call_error, errors.AttemptsExhaustedError)
    assert result.chunks == []
    assert run.models == [MINI, GPT, F]
    chain = result.call_error.attempt_chain
    assert _rows(chain) == [(f"openai/{m}", True) for m in (MINI, GPT, F)]
    (failed,) = run.of(events.CallFailed)
    assert failed.attempts == 3


@pytest.mark.fr("FR30")
def test_a_context_window_error_on_a_streamed_tier_moves_to_medium(
    stream_call: StreamCaller,
) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_too_long]}, default_fallbacks=[F])
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.read_error is None
    assert run.models == [MINI, GPT]
    meta = result.chunks[-1].metadata
    assert meta is not None and meta.tier == "MEDIUM"
    first = meta.attempt_chain[0].dispatches[0]
    assert first.error is not None and first.error.type == "ContextWindowExceededError"


@pytest.mark.fr("FR30")
def test_a_context_window_error_on_a_streamed_medium_goes_to_the_fallback_list(
    stream_call: StreamCaller,
) -> None:
    run = Run(["MEDIUM 0.9"], {GPT: [_too_long]}, default_fallbacks=[F])
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.read_error is None
    assert run.models == [GPT, F] and run.of(events.AttemptSkipped) == []
    meta = result.chunks[-1].metadata
    assert meta is not None and meta.tier is None


@pytest.mark.fr("FR22", "FR30")
def test_a_degraded_stream_on_an_interim_medium_goes_straight_to_mediums_list(
    stream_call: StreamCaller,
) -> None:
    run = Run(["nonsense"], {GPT: [_status(503)]}, fallbacks={GPT: [F]}, default_fallbacks=[D])
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.read_error is None
    assert [e.reason for e in run.of(events.RoutingDegraded)] == ["malformed_output"]
    assert run.models == [GPT, F] and run.of(events.AttemptSkipped) == []
    meta = result.chunks[-1].metadata
    assert meta is not None and meta.tier is None and meta.analyzer is not None
    assert [row.model for row in meta.attempt_chain] == [f"openai/{GPT}", f"openai/{F}"]


@pytest.mark.fr("FR22", "FR30")
def test_a_degraded_stream_on_an_interim_small_fails_over_to_medium(
    stream_call: StreamCaller,
) -> None:
    policy = {**POLICY, "interim_binding": "SMALL"}
    run = Run(["nonsense"], {MINI: [_status(503)]}, policy=policy, default_fallbacks=[F])
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.read_error is None
    assert run.models == [MINI, GPT]
    meta = result.chunks[-1].metadata
    assert meta is not None and meta.tier == "MEDIUM"


# --- streams that fail after their first chunk ---


@pytest.mark.fr("FR30")
def test_a_stream_that_fails_mid_way_on_its_tier_tries_nothing_else(
    stream_call: StreamCaller,
) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_drops]}, default_fallbacks=[F])
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.chunks
    assert isinstance(result.read_error, errors.APIConnectionError)
    assert run.models == [MINI] and _from(result.chunks, MINI)
    assert _rows(result.read_error.attempt_chain) == [(f"openai/{MINI}", True)]
    assert run.of(events.AttemptSkipped) == []
    _nothing_after_the_stream_opened(run)


@pytest.mark.fr("FR30")
def test_a_stream_that_fails_mid_way_on_medium_tries_no_fallback(
    stream_call: StreamCaller,
) -> None:
    run = Run(["SMALL 0.9"], {MINI: [_status(503)], GPT: [_drops]}, default_fallbacks=[F])
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.chunks
    assert isinstance(result.read_error, errors.APIConnectionError)
    assert run.models == [MINI, GPT]
    # The chunks came from MEDIUM alone, and the earlier failure stays on record.
    assert _from(result.chunks, GPT)
    small, medium = result.read_error.attempt_chain
    assert small.model == f"openai/{MINI}" and small.dispatches[0].error is not None
    assert medium.model == f"openai/{GPT}"
    assert medium.dispatches[0].error is not None
    assert medium.dispatches[0].error.type == "APIConnectionError"
    _nothing_after_the_stream_opened(run)


@pytest.mark.fr("FR30")
def test_a_stream_that_fails_mid_way_on_a_fallback_keeps_the_skipped_medium(
    stream_call: StreamCaller,
) -> None:
    run = Run(
        ["SMALL 0.9"],
        {MINI: [_status(503)], F: [_drops]},
        policy=NO_MEDIUM,
        default_fallbacks=[F, D],
    )
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.chunks and isinstance(result.read_error, errors.APIConnectionError)
    assert run.models == [MINI, F] and _from(result.chunks, F)
    assert _rows(result.read_error.attempt_chain) == [
        (f"openai/{MINI}", True),
        (f"openai/{GPT}", False),
        (f"openai/{F}", True),
    ]
    _nothing_after_the_stream_opened(run)


def _nothing_after_the_stream_opened(run: Run) -> None:
    """A failure after the first chunk is the caller's error alone: no move to a next attempt
    is announced for it, and, as a stream reports no outcome event, no `call_failed` either."""
    assert run.handler.types[-1] == "dispatch_started"
    failed = run.of(events.DispatchFailed)
    assert len(failed) == len(run.models) - 1
    assert all(e.next == "next_attempt" for e in failed)
    assert run.of(events.CallFailed) == []


# --- final-chunk metadata ---


@pytest.mark.fr("FR2", "FR19")
def test_a_routed_stream_reports_its_tier_and_analyzer_like_a_plain_answer(
    stream_call: StreamCaller,
) -> None:
    run = Run(["SMALL 0.9"], {})
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.read_error is None
    streamed = result.chunks[-1].metadata
    answered = Run(["SMALL 0.9"], {}).gate.completion(model=ALIAS, messages=MSGS).metadata
    assert streamed is not None and answered is not None
    assert streamed.tier == answered.tier == "SMALL"
    assert streamed.analyzer == answered.analyzer and streamed.analyzer is not None


@pytest.mark.parametrize("script", [{}, {MINI: [_status(503)]}], ids=["chosen", "failover"])
@pytest.mark.fr("FR2")
def test_a_routed_streams_final_metadata_matches_the_non_streamed_answers(
    stream_call: StreamCaller, script: dict[str, list[Reply]]
) -> None:
    result = stream_call(Run(["SMALL 0.9"], script).gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.read_error is None
    streamed = result.chunks[-1].metadata
    answer = Run(["SMALL 0.9"], script).gate.completion(model=ALIAS, messages=MSGS)
    assert streamed is not None and answer.metadata is not None
    # A field set on one path must be set alike on the other; only the Cache flag is apart,
    # since a stream never touches the Cache.
    apart = {"cache_hit"}
    streamed_fields = {k: v for k, v in streamed.to_dict().items() if k not in apart}
    answered_fields = {k: v for k, v in answer.metadata.to_dict().items() if k not in apart}
    assert streamed_fields == answered_fields
    assert {"tier", "analyzer", "attempt_chain", "cost"} <= streamed_fields.keys()


@pytest.mark.fr("FR15")
def test_a_plain_stream_reports_no_tier_or_analyzer(stream_call: StreamCaller) -> None:
    run = Run(["SMALL 0.9"], {})
    result = stream_call(run.gate, model=MINI, messages=MSGS)
    assert result.call_error is None and result.read_error is None
    meta = result.chunks[-1].metadata
    assert meta is not None and meta.tier is None and meta.analyzer is None
    assert run.analyzed == []


# --- cancellation during failover ---


@pytest.fixture(params=["sync", "async"])
def stopper(request: pytest.FixtureRequest) -> Stopper:
    return Stopper(str(request.param))


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    return stand_in_clock(monkeypatch)


CAPPED = {**POLICY, "max_concurrent_analyzer_calls": 1, "spending_ceiling": 100.0}


def _analyzer_cost() -> float:
    answered = Run(["SMALL 0.9"], {}, policy=CAPPED).gate.completion(model=ALIAS, messages=MSGS)
    assert answered.metadata is not None and answered.metadata.analyzer is not None
    assert answered.metadata.analyzer.cost is not None
    return answered.metadata.analyzer.cost.total


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("held", "failing", "unsent"),
    [(GPT, [MINI], F), (F, [MINI, GPT], D)],
    ids=["medium", "fallback"],
)
@pytest.mark.fr("FR27", "FR30")
def test_cancelling_while_a_later_layer_is_held_dispatches_nothing_more(
    stopper: Stopper, stream: bool, held: str, failing: list[str], unsent: str
) -> None:
    def holding(provider: str, is_stream: bool) -> Any:
        return stopper.stop_with(_status(503)(provider, is_stream))

    script: dict[str, list[Reply]] = {m: [_status(503)] for m in failing}
    script[held] = [holding]
    run = Run(["SMALL 0.9"], script, policy=CAPPED, default_fallbacks=[F, D])
    stopper.call(run.gate, model=ALIAS, messages=MSGS, stream=stream)
    assert run.models == [*failing, held] and unsent not in run.models
    _one_cancel_event(run.handler)
    assert run.handler.types[-1] == "call_cancelled"
    cap = run.gate._analyzer_cap
    assert cap is not None and cap.in_flight == 0
    # The Analyzer was already paid for, so the ceiling still counts it.
    assert run.gate._spend.total == pytest.approx(_analyzer_cost())


@pytest.mark.fr("FR27", "FR30")
def test_a_stream_that_failed_over_and_is_cancelled_mid_way_dispatches_nothing_more(
    stopper: Stopper,
) -> None:
    script: dict[str, list[Reply]] = {MINI: [_status(503)], GPT: [_paced]}
    run = Run(["SMALL 0.9"], script, policy=CAPPED, default_fallbacks=[F, D])
    chunks = stopper.read(run.gate, model=ALIAS, messages=MSGS)
    assert len(chunks) == 1 and _from(chunks, GPT)
    assert run.models == [MINI, GPT]
    _one_cancel_event(run.handler)
    assert run.handler.types[-1] == "call_cancelled"
    cap = run.gate._analyzer_cap
    assert cap is not None and cap.in_flight == 0
    assert run.gate._spend.total == pytest.approx(_analyzer_cost())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.fr("FR27")
def test_a_deadline_that_passes_in_a_fallbacks_retry_wait_stops_the_chain(
    clock: Clock, mode: str, stream: bool
) -> None:
    routing = {F: [{"base_url": B_URL, "api_key": B_KEY}]}
    script = {MINI: [_status(503)], GPT: [_status(503)], F: [_status(503), _answer]}
    run = Run(
        ["SMALL 0.9"],
        script,
        policy=CAPPED,
        default_fallbacks=[F, D],
        provider_routing=routing,
        backoff_base=2.0,
    )
    params = {"model": ALIAS, "messages": MSGS, "stream": stream, "deadline": 0.5}
    began = clock.now
    with pytest.raises(CancelledError):
        if mode == "sync":
            run.gate.completion(**params)
        else:
            asyncio.run(run.gate.acompletion(**params))
    assert run.models == [MINI, GPT, F]
    assert run.handler.types[-2:] == ["retry_scheduled", "call_cancelled"]
    _one_cancel_event(run.handler)
    assert clock.now > began + 0.5
    cap = run.gate._analyzer_cap
    assert cap is not None and cap.in_flight == 0
    assert run.gate._spend.total == pytest.approx(_analyzer_cost())
