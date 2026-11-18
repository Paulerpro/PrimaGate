import asyncio
import math
import threading
import time
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import httpx2
import pytest
from test_attempt_chain import SSE, _openai_chunk, _openai_usage
from test_background_preparation import ROUTED, Background
from test_events import Recorder
from test_preparation import GOOD
from test_retry_routing import KEYS, MSGS, Reply, Run, _answer, _status
from test_tier_routing import POLICY, _analyzer_reply
from test_tier_routing import Run as Routed

import primagate
from primagate import CancelledError, CancelToken, PrimaGate
from primagate.drivers import cancel as cancel_module
from primagate.schema import errors, events
from primagate.schema.policy import TIER_ROUTING_ALIAS
from primagate.schema.profile import PrepState

# Taken before any test can patch it, so a held request can always really wait.
_real_sleep = asyncio.sleep
WAIT = 10.0
GPT = "gpt-4o"
OPENAI = "api.openai.com"


class Stopper:
    """Cancels a call the way its mode does: the cancel token in sync mode, the task in async.

    `trigger` is called by the mock while it holds a request. In sync mode the request then
    returns as usual, as one already in flight does; in async mode the cancellation lands while
    the request is still held.
    """

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.token = CancelToken()

    def trigger(self) -> None:
        if self.mode == "sync":
            self.token.cancel()
            return
        task = asyncio.current_task()
        assert task is not None
        task.cancel()

    def held(self, response: httpx2.Response) -> Any:
        """What the mock answers once it has triggered."""
        if self.mode == "sync":
            return response

        async def later() -> httpx2.Response:
            await _real_sleep(WAIT)
            raise AssertionError("the cancellation never landed")

        return later()

    def stop_with(self, response: httpx2.Response) -> Any:
        self.trigger()
        return self.held(response)

    def call(self, gate: PrimaGate, **params: Any) -> None:
        """Make a call and check it ends cancelled, with the error its mode raises."""
        if self.mode == "sync":
            with pytest.raises(CancelledError):
                gate.completion(cancel=self.token, **params)
            return

        async def go() -> None:
            task = asyncio.create_task(gate.acompletion(**params))
            with pytest.raises(asyncio.CancelledError):
                await task
            assert task.cancelled()

        asyncio.run(go())

    def read(self, gate: PrimaGate, **params: Any) -> list[Any]:
        """Open a stream, cancel after its first chunk, and return the chunks that came."""
        chunks: list[Any] = []
        if self.mode == "sync":
            stream = gate.completion(stream=True, cancel=self.token, **params)
            with pytest.raises(CancelledError):
                for chunk in stream:
                    chunks.append(chunk)
                    self.trigger()
            assert list(stream) == []
            return chunks

        async def consume() -> None:
            stream = await gate.acompletion(stream=True, **params)
            try:
                async for chunk in stream:
                    chunks.append(chunk)
                    self.trigger()
            finally:
                assert stream._closed
            raise AssertionError("the stream ended without being cancelled")

        async def go() -> None:
            task = asyncio.create_task(consume())
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(go())
        return chunks


@pytest.fixture(params=["sync", "async"])
def stopper(request: pytest.FixtureRequest) -> Stopper:
    return Stopper(str(request.param))


def _cancelled(handler: Recorder) -> list[events.CallCancelled]:
    return [e for e in handler.seen if isinstance(e, events.CallCancelled)]


def _one_cancel_event(handler: Recorder) -> events.CallCancelled:
    (event,) = _cancelled(handler)
    assert event.event_type == "call_cancelled" and event.to_dict().keys() == {
        "event_type",
        "request_id",
        "timestamp",
    }
    # Cancelling is not failing, so no failure event goes with it.
    assert "call_failed" not in handler.types
    return event


def _plain(script: dict[str, list[Reply]], **kwargs: Any) -> tuple[Run, Recorder]:
    handler = Recorder()
    return Run(script, event_handler=handler, **kwargs), handler


# --- routing ---


def test_cancelling_while_the_analyzer_is_asked_sends_no_worker_and_frees_the_permit(
    stopper: Stopper,
) -> None:
    run = Routed(
        [lambda: stopper.stop_with(_analyzer_reply("SMALL 0.9"))],
        policy={**POLICY, "max_concurrent_analyzer_calls": 1},
    )
    stopper.call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS)
    assert len(run.analyzed) == 1 and run.workers == []
    event = _one_cancel_event(run.handler)
    assert event.request_id.startswith("req-")
    # Nothing is decided or announced once the call is cancelled.
    assert run.handler.types == ["call_cancelled"]
    cap = run.gate._analyzer_cap
    assert cap is not None and cap.in_flight == 0


def test_a_stream_cancelled_while_routing_never_opens(stopper: Stopper) -> None:
    run = Routed([lambda: stopper.stop_with(_analyzer_reply("SMALL 0.9"))])
    stopper.call(run.gate, model=TIER_ROUTING_ALIAS, messages=MSGS, stream=True)
    assert run.workers == [] and run.handler.types == ["call_cancelled"]


def test_a_cancelled_call_does_not_stop_the_preparation_it_started(mode: str) -> None:
    run = Background(mode, [GOOD])
    token = CancelToken()

    def cancel_on_degrade(event: events.Event) -> None:
        if event.event_type == "routing_degraded":
            token.cancel()

    run.gate._event_handler = cancel_on_degrade
    if mode == "sync":
        with pytest.raises(CancelledError):
            run.gate.completion(cancel=token, **ROUTED)
        run.settle_threads()
    else:

        async def go() -> None:
            with pytest.raises(CancelledError):
                await run.gate.acompletion(cancel=token, **ROUTED)
            await run.settle_tasks()

        asyncio.run(go())
    assert run.state is PrepState.READY and run.stored is not None


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


# --- before the start, and between the steps of one model ---


def test_a_token_already_set_stops_the_call_before_anything_is_sent(mode: str) -> None:
    run, handler = _plain({OPENAI: [_answer]})
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        if mode == "sync":
            run.gate.completion(model=GPT, messages=MSGS, cancel=token)
        else:
            asyncio.run(run.gate.acompletion(model=GPT, messages=MSGS, cancel=token))
    assert run.sent == []
    # Asking for an id sends nothing, so even this call is reported under its own.
    assert _one_cancel_event(handler).request_id.startswith("req-")
    assert handler.types == ["call_cancelled"]


def test_cancelling_during_a_failed_dispatch_sends_no_retry(stopper: Stopper) -> None:
    def failing(provider: str, stream: bool) -> Any:
        return stopper.stop_with(_status(503)(provider, stream))

    run, handler = _plain(
        {OPENAI: [failing, _answer]},
        provider_routing={GPT: [{"base_url": "https://b.test/v1"}]},
    )
    stopper.call(run.gate, model=GPT, messages=MSGS)
    assert run.hosts == [OPENAI]
    assert _one_cancel_event(handler).request_id.startswith("req-")
    assert "retry_scheduled" not in handler.types


def test_cancelling_between_fallbacks_tries_no_further_entry(stopper: Stopper) -> None:
    def refused(provider: str, stream: bool) -> Any:
        return stopper.stop_with(_status(401)(provider, stream))

    run, handler = _plain(
        {OPENAI: [refused], "api.anthropic.com": [_answer]},
        fallbacks={GPT: ["claude-sonnet-5-5"]},
    )
    stopper.call(run.gate, model=GPT, messages=MSGS)
    assert run.hosts == [OPENAI]
    _one_cancel_event(handler)
    assert "attempt_skipped" not in handler.types


def test_an_answer_that_arrives_after_the_cancel_is_never_cached(stopper: Stopper) -> None:
    def answered(provider: str, stream: bool) -> Any:
        return stopper.stop_with(_answer(provider, stream))

    run, handler = _plain({OPENAI: [answered, _answer]})
    stopper.call(run.gate, model=GPT, messages=MSGS)
    _one_cancel_event(handler)
    again = run.gate.completion(model=GPT, messages=MSGS)
    assert again.metadata is not None and not again.metadata.cache_hit
    assert run.hosts == [OPENAI, OPENAI]


def test_a_token_set_after_the_answer_arrived_stops_before_the_cache_write(mode: str) -> None:
    token = CancelToken()
    handler = Recorder()

    def watch(event: events.Event) -> None:
        handler(event)
        if event.event_type == "call_succeeded":
            token.cancel()

    run = Run({OPENAI: [_answer]}, event_handler=watch)
    with pytest.raises(CancelledError):
        if mode == "sync":
            run.gate.completion(model=GPT, messages=MSGS, cancel=token)
        else:
            asyncio.run(run.gate.acompletion(model=GPT, messages=MSGS, cancel=token))
    assert handler.types == ["dispatch_started", "call_succeeded", "call_cancelled"]
    assert not run.gate._cache._entries
    again = run.gate.completion(model=GPT, messages=MSGS)
    assert again.metadata is not None and not again.metadata.cache_hit


def test_a_token_set_mid_call_in_async_raises_the_sdk_error_at_the_next_step() -> None:
    token = CancelToken()

    def answered(provider: str, stream: bool) -> httpx2.Response:
        token.cancel()
        return _answer(provider, stream)

    run, handler = _plain({OPENAI: [answered]})

    async def go() -> None:
        with pytest.raises(CancelledError) as raised:
            await run.gate.acompletion(model=GPT, messages=MSGS, cancel=token)
        assert isinstance(raised.value, errors.PrimaGateError)

    asyncio.run(go())
    assert run.hosts == [OPENAI]
    assert handler.types == ["dispatch_started", "call_succeeded", "call_cancelled"]


def _call(mode: str, gate: PrimaGate, **params: Any) -> Any:
    if mode == "sync":
        return gate.completion(**params)
    return asyncio.run(gate.acompletion(**params))


@pytest.mark.parametrize("when", ["during_the_dispatch", "on_call_failed"])
def test_a_token_set_once_the_call_has_failed_never_hides_the_error(mode: str, when: str) -> None:
    token = CancelToken()
    handler = Recorder()

    def watch(event: events.Event) -> None:
        handler(event)
        if when == "on_call_failed" and event.event_type == "call_failed":
            token.cancel()

    def refused(provider: str, stream: bool) -> httpx2.Response:
        if when == "during_the_dispatch":
            token.cancel()
        return _status(401)(provider, stream)

    run = Run({OPENAI: [refused]}, event_handler=watch)
    with pytest.raises(errors.AuthenticationError):
        _call(mode, run.gate, model=GPT, messages=MSGS, cancel=token)
    assert handler.types == ["dispatch_started", "dispatch_failed", "call_failed"]


def test_without_a_cache_write_a_token_set_on_success_still_returns_the_answer(mode: str) -> None:
    token = CancelToken()
    handler = Recorder()

    def watch(event: events.Event) -> None:
        handler(event)
        if event.event_type == "call_succeeded":
            token.cancel()

    run = Run({OPENAI: [_answer]}, event_handler=watch)
    result = _call(mode, run.gate, model=GPT, messages=MSGS, cache=False, cancel=token)
    assert result.choices[0].message.content
    assert handler.types == ["dispatch_started", "call_succeeded"]


# --- spend already billed ---

CEILING = {**POLICY, "spending_ceiling": 100.0}


def _routed_total(mode: str, cancel_on: str | None) -> tuple[float, Routed]:
    run = Routed(["SMALL 0.9"], policy=CEILING)
    token = CancelToken()

    def watch(event: events.Event) -> None:
        run.handler(event)
        if event.event_type == cancel_on:
            token.cancel()

    # Before any driver exists, so both modes are handed this handler.
    run.gate._event_handler = watch
    params = {"model": TIER_ROUTING_ALIAS, "messages": MSGS, "cancel": token}
    if cancel_on is None:
        _call(mode, run.gate, **params)
    else:
        with pytest.raises(CancelledError):
            _call(mode, run.gate, **params)
    return run.gate._spend.total, run


def test_a_call_cancelled_after_its_answer_still_counts_what_it_cost(mode: str) -> None:
    finished, _ = _routed_total(mode, None)
    cancelled, run = _routed_total(mode, "call_succeeded")
    assert finished > 0 and cancelled == pytest.approx(finished)
    assert run.handler.types[-2:] == ["call_succeeded", "call_cancelled"]


def test_a_call_cancelled_after_the_analyzer_answered_counts_the_analyzer(mode: str) -> None:
    cancelled, run = _routed_total(mode, "routing_decision")
    assert run.workers == [] and len(run.analyzed) == 1
    answer = Routed(["SMALL 0.9"], policy=CEILING).gate.completion(
        model=TIER_ROUTING_ALIAS, messages=MSGS
    )
    analyzer = answer.metadata.analyzer if answer.metadata else None
    assert analyzer is not None and analyzer.cost is not None
    assert cancelled == pytest.approx(analyzer.cost.total) and cancelled > 0


def test_one_shared_token_stops_every_call_given_it() -> None:
    token = CancelToken()
    inside = threading.Barrier(3)
    let_go = threading.Event()

    def held(provider: str, stream: bool) -> httpx2.Response:
        inside.wait(WAIT)
        assert let_go.wait(WAIT)
        return _answer(provider, stream)

    run, handler = _plain({OPENAI: [held]})
    raised: list[BaseException] = []

    def one(n: int) -> None:
        try:
            run.gate.completion(
                model=GPT, messages=[{"role": "user", "content": f"q{n}"}], cancel=token
            )
        except BaseException as err:
            raised.append(err)

    workers = [threading.Thread(target=one, args=(n,), daemon=True) for n in range(2)]
    for worker in workers:
        worker.start()
    inside.wait(WAIT)
    token.cancel()
    let_go.set()
    for worker in workers:
        worker.join(WAIT)
    assert len(raised) == 2 and all(isinstance(e, CancelledError) for e in raised)
    stopped = _cancelled(handler)
    assert len(stopped) == 2 and len({e.request_id for e in stopped}) == 2
    assert not run.gate._cache._entries
    # A cancelled token stays cancelled: a later call given it stops before sending anything.
    with pytest.raises(CancelledError):
        run.gate.completion(model=GPT, messages=MSGS, cancel=token)
    assert len(run.sent) == 2


# --- retry waits ---


class Clock:
    """Stands in for the monotonic clock; only sleeping moves it."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """The deadline's clock, moved only by retry waits or by the test itself. Only the driver's
    view of the clock is replaced, so the event loop keeps its own."""
    stand_in = Clock()

    async def asleep(seconds: float) -> None:
        stand_in.sleep(seconds)
        await _real_sleep(0)

    monkeypatch.setattr(cancel_module, "time", SimpleNamespace(monotonic=stand_in))
    monkeypatch.setattr(time, "sleep", stand_in.sleep)
    monkeypatch.setattr(asyncio, "sleep", asleep)
    return stand_in


def test_a_deadline_that_passes_during_a_retry_wait_stops_before_the_retry(
    mode: str, clock: Clock
) -> None:
    run, handler = _plain(
        {OPENAI: [_status(503), _answer]},
        provider_routing={GPT: [{"base_url": "https://b.test/v1"}]},
        backoff_base=2.0,
    )
    with pytest.raises(CancelledError):
        _call(mode, run.gate, model=GPT, messages=MSGS, deadline=0.5)
    assert run.hosts == [OPENAI]
    assert handler.types[-2:] == ["retry_scheduled", "call_cancelled"]
    assert clock.now > 1000.5


def test_a_task_cancelled_in_its_retry_wait_sends_no_retry() -> None:
    run, handler = _plain(
        {OPENAI: [_status(503), _answer]},
        provider_routing={GPT: [{"base_url": "https://b.test/v1"}]},
        backoff_base=30.0,
        backoff_max=60.0,
    )

    async def go() -> None:
        task = asyncio.create_task(run.gate.acompletion(model=GPT, messages=MSGS))
        for _ in range(int(WAIT / 0.005)):
            if "retry_scheduled" in handler.types:
                break
            await _real_sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert run.hosts == [OPENAI]
    assert handler.types[-2:] == ["retry_scheduled", "call_cancelled"]


# --- streams ---


class Paced(httpx2.SyncByteStream, httpx2.AsyncByteStream):
    """An event stream sent one event at a time; in async mode each waits its turn on the loop,
    so a cancelled reader is stopped between chunks."""

    def __init__(self, parts: list[bytes]) -> None:
        self._parts = parts

    def __iter__(self) -> Iterator[bytes]:
        yield from self._parts

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for part in self._parts:
            await _real_sleep(0)
            yield part


def _paced(provider: str, stream: bool) -> httpx2.Response:
    parts = [
        _openai_chunk({"content": "one"}),
        _openai_chunk({"content": " two"}),
        _openai_chunk({}, "stop"),
        _openai_usage(),
        "data: [DONE]\n\n",
    ]
    return httpx2.Response(200, stream=Paced([p.encode() for p in parts]), headers=SSE)


def test_a_stream_cancelled_between_chunks_yields_no_more_and_closes(stopper: Stopper) -> None:
    run, handler = _plain({OPENAI: [_paced]})
    chunks = stopper.read(run.gate, model=GPT, messages=MSGS)
    assert len(chunks) == 1
    event = _one_cancel_event(handler)
    assert event.request_id == handler.seen[0].request_id
    assert handler.types == ["dispatch_started", "call_cancelled"]


def test_a_token_set_between_chunks_stops_the_stream_in_either_mode(mode: str) -> None:
    run, handler = _plain({OPENAI: [_paced]})
    token = CancelToken()
    chunks: list[Any] = []

    def read_sync() -> None:
        stream = run.gate.completion(model=GPT, messages=MSGS, stream=True, cancel=token)
        for chunk in stream:
            chunks.append(chunk)
            token.cancel()

    async def read_async() -> None:
        stream = await run.gate.acompletion(model=GPT, messages=MSGS, stream=True, cancel=token)
        async for chunk in stream:
            chunks.append(chunk)
            token.cancel()

    with pytest.raises(CancelledError):
        if mode == "sync":
            read_sync()
        else:
            asyncio.run(read_async())
    assert len(chunks) == 1
    assert handler.types == ["dispatch_started", "call_cancelled"]


def test_a_deadline_that_passes_between_chunks_stops_the_stream(mode: str, clock: Clock) -> None:
    run, handler = _plain({OPENAI: [_paced]})
    chunks: list[Any] = []
    params = {"model": GPT, "messages": MSGS, "stream": True, "deadline": 5}

    def read_sync() -> None:
        for chunk in run.gate.completion(**params):
            chunks.append(chunk)
            clock.now += 10

    async def read_async() -> None:
        async for chunk in await run.gate.acompletion(**params):
            chunks.append(chunk)
            clock.now += 10

    with pytest.raises(CancelledError):
        if mode == "sync":
            read_sync()
        else:
            asyncio.run(read_async())
    assert len(chunks) == 1
    assert handler.types == ["dispatch_started", "call_cancelled"]


# --- without cancellation ---


@pytest.mark.parametrize("settings", [{}, {"cancel": CancelToken(), "deadline": 600}])
def test_a_call_that_is_not_cancelled_behaves_as_before(
    mode: str, settings: dict[str, Any]
) -> None:
    run, handler = _plain({OPENAI: [_answer]})
    if mode == "sync":
        result = run.gate.completion(model=GPT, messages=MSGS, **settings)
    else:
        result = asyncio.run(run.gate.acompletion(model=GPT, messages=MSGS, **settings))
    assert result.choices[0].message.content
    assert handler.types == ["dispatch_started", "call_succeeded"]


def test_one_token_can_be_cancelled_from_another_thread() -> None:
    token = CancelToken()
    assert not token.cancelled
    worker = threading.Thread(target=token.cancel)
    worker.start()
    worker.join(WAIT)
    assert token.cancelled
    token.cancel()
    assert token.cancelled


# --- settings and exports ---


@pytest.mark.parametrize(
    "bad",
    [
        {"cancel": "x"},
        {"cancel": True},
        {"deadline": 0},
        {"deadline": -1},
        {"deadline": math.nan},
        {"deadline": math.inf},
        {"deadline": True},
        {"deadline": "5"},
    ],
)
@pytest.mark.parametrize("stream", [False, True])
def test_bad_cancellation_settings_are_refused_before_anything_is_sent(
    mode: str, bad: dict[str, Any], stream: bool
) -> None:
    run, handler = _plain({OPENAI: [_answer]})
    params = {"model": GPT, "messages": MSGS, "stream": stream, **bad}
    with pytest.raises(errors.BadRequestError):
        if mode == "sync":
            run.gate.completion(**params)
        else:
            asyncio.run(run.gate.acompletion(**params))
    assert run.sent == [] and handler.seen == []


def test_the_token_and_the_error_are_exported_from_the_package_root() -> None:
    assert primagate.CancelToken is CancelToken
    assert primagate.CancelledError is errors.CancelledError
    assert issubclass(primagate.CancelledError, primagate.PrimaGateError)
    assert primagate.CallCancelled is events.CallCancelled
    assert {"CancelToken", "CancelledError", "CallCancelled"} <= set(primagate.__all__)


def test_the_module_level_functions_take_the_settings_too(monkeypatch: pytest.MonkeyPatch) -> None:
    from primagate import client as client_module

    sent: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return _answer("openai", False)

    gate = PrimaGate(api_keys=KEYS, transport=httpx2.MockTransport(handle))
    monkeypatch.setattr(client_module, "_default", gate)
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        primagate.completion(model=GPT, messages=MSGS, cancel=token)
    with pytest.raises(CancelledError):
        asyncio.run(primagate.acompletion(model=GPT, messages=MSGS, cancel=token))
    assert sent == []
