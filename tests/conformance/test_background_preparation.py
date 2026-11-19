import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx2
import pytest
from test_events import Recorder
from test_preparation import DECLARED, GOOD, KEYS, POLICY, SECRET, TWO, Reply, _reply, _status
from test_retry_routing import _answer as worker_answer

from primagate import FileProfileStore, PrimaGate
from primagate.core.profile import PREP_PROMPT_VERSION
from primagate.drivers.preparation import PREP_RESTART_COOLDOWN, PreparationRunner
from primagate.schema import events
from primagate.schema.policy import TIER_ROUTING_ALIAS
from primagate.schema.profile import (
    PROFILE_SCHEMA_VERSION,
    PrepState,
    Profile,
    ProfileContent,
)
from primagate.stores.memory import InMemoryProfileStore

# Taken before any test can patch it, so a held dispatch can always really wait.
_real_sleep = asyncio.sleep
WAIT = 10.0
ROUTED: dict[str, Any] = {
    "model": TIER_ROUTING_ALIAS,
    "messages": [{"role": "user", "content": "where is it"}],
}


PRICE = {"gpt-4o": {"input": 9.0, "output": 9.0}}


def _asking(n: int) -> dict[str, Any]:
    # A different question each time, so no call is answered from the Cache before routing.
    return {**ROUTED, "messages": [{"role": "user", "content": f"question {n}"}]}


class Clock:
    def __init__(self) -> None:
        self.now = 5000.0

    def __call__(self) -> float:
        return self.now


class Background:
    """A gate with declared sources whose mock answers preparation from a script, and can hold
    the preparation dispatch until the test lets it go."""

    def __init__(
        self,
        mode: str,
        replies: list[Reply],
        *,
        hold: bool = False,
        clock: Clock | None = None,
        **kwargs: Any,
    ) -> None:
        self.mode = mode
        self.replies = list(replies)
        self.hold = hold
        self.entered = threading.Event()
        self.release = threading.Event()
        self.prepared: list[dict[str, Any]] = []
        self.prep_threads: list[threading.Thread] = []
        self.analyzer_prompts: list[str] = []
        self.handler = Recorder()
        kwargs.setdefault("profile_sources", DECLARED)
        kwargs.setdefault("tier_bindings", TWO)
        self.gate = PrimaGate(
            api_keys=KEYS,
            transport=httpx2.MockTransport(self._handle),
            policy=POLICY,
            analyzer="gpt-4o-mini",
            event_handler=self.handler,
            **kwargs,
        )
        if clock is not None:
            # Swapped in before any driver exists, so both drivers are handed this one.
            self.gate._runner = PreparationRunner(clock=clock)

    def _handle(self, request: httpx2.Request) -> Any:
        body = json.loads(request.content)
        system = str(body.get("messages", [{}])[0].get("content", ""))
        if f"Prompt version: {PREP_PROMPT_VERSION}" in system:
            self.prepared.append(body)
            self.prep_threads.append(threading.current_thread())
            reply = self.replies.pop(0)
            response = reply() if callable(reply) else _reply(_text(reply))
            if not self.hold:
                return response
            self.entered.set()
            if self.mode == "sync":
                assert self.release.wait(WAIT)
                return response

            async def later() -> httpx2.Response:
                while not self.release.is_set():
                    await _real_sleep(0.005)
                return response

            return later()
        if "Prompt version: analyzer" in system:
            self.analyzer_prompts.append(system)
            return _reply("SMALL 0.9")
        return worker_answer("openai", bool(body.get("stream")))

    @property
    def runner(self) -> PreparationRunner:
        runner = self.gate._runner
        assert runner is not None
        return runner

    @property
    def fingerprint(self) -> str:
        tiers = self.gate._tier_routing
        assert tiers is not None and tiers.fingerprint is not None
        return tiers.fingerprint

    @property
    def state(self) -> PrepState | None:
        return self.runner.state(self.fingerprint)

    @property
    def stored(self) -> Any:
        return self.gate._profile_store.read()

    @property
    def types(self) -> list[str]:
        return self.handler.types

    def settle_threads(self) -> None:
        for thread in list(self.runner._threads):
            thread.join(WAIT)
            assert not thread.is_alive()

    async def settle_tasks(self) -> None:
        await asyncio.wait_for(
            asyncio.gather(*list(self.runner._tasks), return_exceptions=True), WAIT
        )

    def run(self, **call: Any) -> Any:
        """Make one call and let any preparation it started finish, in this test's mode."""
        params = {**ROUTED, **call}
        if self.mode == "sync":
            result = self.gate.completion(**params)
            self.settle_threads()
            return result

        async def go() -> Any:
            result = await self.gate.acompletion(**params)
            await self.settle_tasks()
            return result

        return asyncio.run(go())


def _text(reply: str | dict[str, Any]) -> str:
    return reply if isinstance(reply, str) else json.dumps(reply)


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


async def _until(flag: threading.Event) -> None:
    for _ in range(int(WAIT / 0.005)):
        if flag.is_set():
            return
        await _real_sleep(0.005)
    raise AssertionError("the preparation dispatch was never reached")


def _prep_events(run: Background) -> list[events.Event]:
    return [e for e in run.handler.seen if e.request_id.startswith("prep-")]


# --- the first request ---


@pytest.mark.fr("FR22", "FR23")
def test_the_first_routed_call_starts_one_preparation_and_does_not_wait(mode: str) -> None:
    run = Background(mode, [GOOD], hold=True)

    def check_returned(answer: Any) -> None:
        assert answer.choices[0].message.content
        # The call is over while the Analyzer still holds the preparation.
        assert run.state is PrepState.PREPARING
        assert "profile_ready" not in run.types and run.stored is None

    if mode == "sync":
        check_returned(run.gate.completion(**ROUTED))
        assert run.entered.wait(WAIT)
        run.release.set()
        run.settle_threads()
    else:

        async def go() -> None:
            check_returned(await run.gate.acompletion(**ROUTED))
            await _until(run.entered)
            run.release.set()
            await run.settle_tasks()

        asyncio.run(go())

    assert run.state is PrepState.READY
    assert len(run.prepared) == 1 and SECRET in run.prepared[0]["messages"][1]["content"]
    assert run.stored is not None and run.stored.input_fingerprint == run.fingerprint
    assert [e.event_type for e in _prep_events(run)] == ["preparation_started", "profile_ready"]


@pytest.fixture
def no_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preparation retries sleep; make them instant without touching the held-dispatch wait."""

    async def instant(seconds: float) -> None:
        await _real_sleep(0)

    monkeypatch.setattr(time, "sleep", lambda seconds: None)
    monkeypatch.setattr(asyncio, "sleep", instant)


@pytest.mark.fr("FR23")
def test_every_preparation_event_has_a_prep_id_and_the_call_keeps_req(
    mode: str, no_waits: None
) -> None:
    run = Background(mode, ["prose", GOOD])
    run.run()
    assert run.state is PrepState.READY and len(run.prepared) == 2
    prep = {"preparation_started", "preparation_failed", "profile_ready"}
    for event in run.handler.seen:
        expected = "prep-" if event.event_type in prep else "req-"
        assert event.request_id.startswith(expected), event
    assert "preparation_failed" in run.types and "profile_ready" in run.types


@pytest.mark.fr("FR23")
def test_twenty_concurrent_first_calls_start_exactly_one(mode: str) -> None:
    run = Background(mode, [GOOD])
    if mode == "sync":
        barrier = threading.Barrier(20)
        failures: list[BaseException] = []

        def call(n: int) -> None:
            barrier.wait()
            try:
                run.gate.completion(**_asking(n))
            except BaseException as err:  # pragma: no cover - reported below
                failures.append(err)

        threads = [threading.Thread(target=call, args=(n,)) for n in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(WAIT)
            assert not t.is_alive()
        assert failures == []
        run.settle_threads()
    else:

        async def go() -> None:
            await asyncio.gather(*(run.gate.acompletion(**_asking(n)) for n in range(20)))
            await run.settle_tasks()

        asyncio.run(go())
    assert len(run.prepared) == 1
    assert run.types.count("preparation_started") == 1
    assert run.types.count("routing_decision") == 20
    assert run.state is PrepState.READY


@pytest.mark.fr("FR23")
def test_a_ready_fingerprint_is_not_prepared_again(mode: str) -> None:
    run = Background(mode, [GOOD])
    for n in range(3):
        run.run(**_asking(n))
    assert len(run.prepared) == 1 and run.types.count("preparation_started") == 1


# --- what the store already holds ---


def _profile(fingerprint: str) -> Profile:
    return Profile(
        "prep-earlier",
        fingerprint,
        PROFILE_SCHEMA_VERSION,
        "2026-11-17T10:00:00+00:00",
        ProfileContent(purpose="Answers order questions.", unknowns=()),
        source_ref="earlier",
    )


@pytest.mark.fr("FR29")
def test_a_matching_stored_profile_is_reused_after_a_restart(mode: str, tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    first = Background(mode, [GOOD], profile_store=FileProfileStore(path))
    first.run()
    assert first.state is PrepState.READY and len(first.prepared) == 1
    written = first.stored

    # A new instance is a restart: its runner starts empty, and only the file remembers.
    second = Background(mode, [], profile_store=FileProfileStore(path))
    assert second.state is None
    second.run()
    assert second.state is PrepState.READY
    assert second.prepared == [] and second.stored == written
    assert not any(t.startswith("prep") or t == "profile_ready" for t in second.types)


@pytest.mark.fr("FR24", "FR29")
def test_a_stale_stored_profile_is_prepared_again_and_replaced(mode: str) -> None:
    store = InMemoryProfileStore()
    store.write(_profile("fp-from-other-sources"))
    run = Background(mode, [GOOD], profile_store=store)
    run.run()
    assert run.state is PrepState.READY and len(run.prepared) == 1
    assert run.stored.input_fingerprint == run.fingerprint
    assert run.stored.profile_id.startswith("prep-") and run.stored.profile_id != "prep-earlier"


@pytest.mark.fr("FR17", "FR24")
def test_only_a_changed_source_gives_a_new_fingerprint_and_a_new_preparation(
    mode: str, tmp_path: Path
) -> None:
    path = tmp_path / "profile.json"
    first = Background(mode, [GOOD], profile_store=FileProfileStore(path))
    first.run()
    assert first.state is PrepState.READY and len(first.prepared) == 1

    # Prices and bindings are not sources, so instances differing only there reuse the Profile.
    priced = {**TWO, "SMALL": {"model": "gpt-4o-mini", "pricing": {"input": 1.0, "output": 2.0}}}
    for unchanged in (
        Background(mode, [], profile_store=FileProfileStore(path), price_overrides=PRICE),
        Background(mode, [], profile_store=FileProfileStore(path), tier_bindings=priced),
    ):
        assert unchanged.fingerprint == first.fingerprint
        unchanged.run()
        assert unchanged.prepared == [] and unchanged.state is PrepState.READY

    sources = {**DECLARED, "instructions": SECRET + " Refunds are handled too."}
    changed = Background(
        mode, [GOOD], profile_store=FileProfileStore(path), profile_sources=sources
    )
    assert changed.fingerprint != first.fingerprint
    changed.run()
    assert len(changed.prepared) == 1 and changed.state is PrepState.READY
    assert changed.stored.input_fingerprint == changed.fingerprint


# --- calls that never prepare ---


@pytest.mark.fr("FR23")
def test_no_sources_means_no_runner_and_no_preparation(mode: str) -> None:
    run = Background(mode, [], profile_sources=None)
    assert run.gate._runner is None
    if mode == "sync":
        run.gate.completion(**ROUTED)
    else:
        asyncio.run(run.gate.acompletion(**ROUTED))
    assert run.prepared == [] and run.stored is None
    assert not any(t.startswith("prep") for t in run.types)


@pytest.mark.fr("FR15", "FR23")
def test_a_plain_model_call_never_prepares(mode: str) -> None:
    run = Background(mode, [])
    run.run(model="gpt-4o")
    assert run.prepared == [] and run.state is None
    assert run.runner._threads == set() and run.runner._tasks == set()


# --- failures ---


class Broken(Exception):
    pass


class FailingStore(InMemoryProfileStore):
    def __init__(self, non_blocking: bool, *, on: str) -> None:
        super().__init__()
        self.non_blocking = non_blocking
        self.on = on

    def read(self) -> Profile | None:
        if self.on == "read":
            raise Broken("store is down")
        return super().read()

    def write(self, profile: Profile) -> None:
        if self.on == "write":
            raise Broken("store is down")
        super().write(profile)


@pytest.mark.parametrize("non_blocking", [True, False])
@pytest.mark.parametrize(("on", "attempt"), [("write", 1), ("read", 0)])
@pytest.mark.fr("FR23")
def test_a_store_error_is_a_final_store_error_failure_and_escapes_nowhere(
    mode: str, non_blocking: bool, on: str, attempt: int
) -> None:
    run = Background(mode, [GOOD], profile_store=FailingStore(non_blocking, on=on))
    answer = run.run()
    assert answer.choices[0].message.content
    assert run.state is PrepState.FAILED
    (failed,) = [e for e in run.handler.seen if isinstance(e, events.PreparationFailed)]
    assert (failed.reason, failed.final, failed.attempt) == ("store_error", True, attempt)
    assert failed.request_id.startswith("prep-")
    assert failed.input_fingerprint == run.fingerprint
    assert "profile_ready" not in run.types
    if on == "write":
        (started,) = [e for e in run.handler.seen if isinstance(e, events.PreparationStarted)]
        assert failed.request_id == started.request_id


@pytest.mark.fr("FR23")
def test_an_unexpected_error_is_a_final_provider_error(
    mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    def explode() -> httpx2.Response:
        raise RuntimeError("transport bug")

    caplog.set_level(logging.DEBUG, logger="primagate")
    run = Background(mode, [explode])
    run.run()
    assert run.state is PrepState.FAILED
    (failed,) = [e for e in run.handler.seen if isinstance(e, events.PreparationFailed)]
    assert (failed.reason, failed.final) == ("provider_error", True)
    (logged,) = [r for r in caplog.records if r.getMessage() == "preparation failed unexpectedly"]
    assert logged.levelno == logging.DEBUG and logged.exc_info is not None


@pytest.mark.fr("FR23")
@pytest.mark.mode_independent(
    "drives the sync runner's job directly; async preparation runs no such job"
)
def test_a_sync_job_hit_by_a_base_exception_records_a_failure_and_lets_it_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = Clock()
    run = Background("sync", [], clock=clock)
    driver = run.gate._get_driver()

    def exit_now(*args: Any) -> Any:
        raise SystemExit

    monkeypatch.setattr(driver, "_drive", exit_now)
    assert run.runner.try_start(run.fingerprint)
    tiers = run.gate._tier_routing
    assert tiers is not None and tiers.sources is not None
    with pytest.raises(SystemExit):
        driver._job(tiers.sources, run.fingerprint)
    assert run.state is PrepState.FAILED
    clock.now += PREP_RESTART_COOLDOWN
    assert run.runner.try_start(run.fingerprint)


@pytest.mark.fr("FR23")
def test_a_failure_waits_out_the_cooldown_before_one_new_start(mode: str) -> None:
    clock = Clock()
    run = Background(mode, [_status(401), GOOD], clock=clock)
    run.run()
    assert run.state is PrepState.FAILED and len(run.prepared) == 1

    clock.now += 5 * 60
    run.run(messages=[{"role": "user", "content": "five minutes on"}])
    assert len(run.prepared) == 1 and run.state is PrepState.FAILED

    clock.now += PREP_RESTART_COOLDOWN
    run.run(messages=[{"role": "user", "content": "over ten minutes on"}])
    run.run(messages=[{"role": "user", "content": "and again"}])
    assert len(run.prepared) == 2 and run.state is PrepState.READY
    assert run.types.count("preparation_started") == 2


@pytest.mark.fr("FR23", "FR24")
def test_an_overflow_failure_is_never_restarted(mode: str) -> None:
    clock = Clock()
    run = Background(mode, [], clock=clock, profile_sources={"instructions": "word " * 20_000})
    run.run()
    assert run.state is PrepState.FAILED
    clock.now += 100 * PREP_RESTART_COOLDOWN
    run.run(messages=[{"role": "user", "content": "much later"}])
    assert run.prepared == [] and run.types.count("preparation_started") == 1


# --- where it runs ---


@pytest.mark.fr("FR23")
@pytest.mark.mode_independent("asserts where the sync runner works")
def test_a_sync_call_prepares_on_a_daemon_thread() -> None:
    run = Background("sync", [GOOD])
    run.run()
    (thread,) = run.prep_threads
    assert thread.daemon and thread is not threading.main_thread()


@pytest.mark.fr("FR23")
@pytest.mark.mode_independent("asserts where the async runner works")
def test_an_async_call_prepares_on_a_task_of_its_own_loop() -> None:
    run = Background("async", [GOOD], hold=True)

    async def go() -> None:
        await run.gate.acompletion(**ROUTED)
        (task,) = run.runner._tasks
        assert task.get_loop() is asyncio.get_running_loop()
        await _until(run.entered)
        # Still on the loop's own thread: no worker thread was started for it.
        assert run.prep_threads == [threading.current_thread()]
        run.release.set()
        await run.settle_tasks()

    asyncio.run(go())
    assert run.state is PrepState.READY


@pytest.mark.fr("FR23", "FR27")
@pytest.mark.mode_independent("only an event loop can end under a running preparation")
def test_an_async_preparation_cut_off_by_its_loop_ending_waits_out_the_cooldown() -> None:
    clock = Clock()
    run = Background("async", [GOOD, GOOD], hold=True, clock=clock)

    async def call_only(n: int) -> None:
        await run.gate.acompletion(**_asking(n))
        await _until(run.entered)

    # asyncio.run cancels whatever is left when the call returns, the preparation included.
    asyncio.run(call_only(0))
    assert run.state is PrepState.FAILED and run.stored is None
    # Silent: the cut-off preparation sends no final event.
    assert [e.event_type for e in _prep_events(run)] == ["preparation_started"]
    run.release.set()

    clock.now += PREP_RESTART_COOLDOWN - 1
    run.run(**_asking(1))
    assert len(run.prepared) == 1 and run.state is PrepState.FAILED

    clock.now += 1
    run.run(**_asking(2))
    assert len(run.prepared) == 2 and run.state is PrepState.READY


# --- request in flight keeps what it started with ---


@pytest.mark.fr("FR23")
def test_only_a_call_that_starts_once_ready_reads_the_profile(mode: str) -> None:
    run = Background(mode, [GOOD])
    pinned = run.gate._tier_routing
    run.run()
    assert run.state is PrepState.READY
    run.run(messages=[{"role": "user", "content": "after ready"}])
    assert run.gate._tier_routing is pinned
    # The first call started PREPARING, so only the second asked the Analyzer.
    (prompt,) = run.analyzer_prompts
    assert "Purpose: Answers questions about orders." in prompt
    decisions = [e for e in run.handler.seen if isinstance(e, events.RoutingDecision)]
    assert [d.input_fingerprint for d in decisions] == ["unknown", run.fingerprint]


class CountingStore(InMemoryProfileStore):
    def __init__(self) -> None:
        super().__init__()
        self.writers: list[str] = []

    def write(self, profile: Profile) -> None:
        self.writers.append(threading.current_thread().name)
        super().write(profile)


@pytest.mark.fr("FR23")
def test_only_the_preparation_writes_a_profile(mode: str) -> None:
    store = CountingStore()
    run = Background(mode, [GOOD], profile_store=store)
    for n in range(3):
        run.run(messages=[{"role": "user", "content": f"call {n}"}])
    assert len(store.writers) == 1
    if mode == "sync":
        assert store.writers == ["primagate-preparation"]


# --- stopping ---


@pytest.mark.fr("FR23", "FR27")
def test_close_mid_preparation_stops_it_with_the_store_unchanged(mode: str) -> None:
    store = CountingStore()
    run = Background(mode, [GOOD], hold=True, profile_store=store)
    if mode == "sync":
        run.gate.completion(**ROUTED)
        assert run.entered.wait(WAIT)
        run.gate.close()
        run.release.set()
        run.settle_threads()
    else:

        async def go() -> None:
            await run.gate.acompletion(**ROUTED)
            await _until(run.entered)
            run.gate.close()
            run.release.set()
            await run.settle_tasks()

        asyncio.run(go())
    assert store.writers == [] and run.stored is None
    assert [e.event_type for e in _prep_events(run)] == ["preparation_started"]
    assert run.runner.try_start(run.fingerprint) is False


@pytest.mark.fr("FR23", "FR27")
def test_aclose_mid_preparation_cancels_and_awaits_it(mode: str) -> None:
    store = CountingStore()
    run = Background(mode, [GOOD], hold=True, profile_store=store)
    if mode == "sync":
        run.gate.completion(**ROUTED)
        assert run.entered.wait(WAIT)
        asyncio.run(run.gate.aclose())
        run.release.set()
        run.settle_threads()
    else:

        async def go() -> None:
            await run.gate.acompletion(**ROUTED)
            await _until(run.entered)
            (task,) = run.runner._tasks
            await run.gate.aclose()
            # Awaited by aclose itself, so it is over before aclose returns.
            assert task.cancelled()

        asyncio.run(go())
    assert store.writers == [] and run.stored is None
    assert [e.event_type for e in _prep_events(run)] == ["preparation_started"]


@pytest.mark.fr("FR23")
def test_after_close_nothing_starts(mode: str) -> None:
    run = Background(mode, [])
    if mode == "sync":
        run.gate.close()
    else:
        asyncio.run(run.gate.aclose())
    assert run.runner.is_closed
    assert run.runner.try_start(run.fingerprint) is False
