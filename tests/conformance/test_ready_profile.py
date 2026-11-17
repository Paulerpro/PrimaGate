import asyncio
import json
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from test_events import Recorder
from test_preparation import DECLARED, GOOD, KEYS, POLICY, TWO, Reply, _reply, _status
from test_retry_routing import _answer as worker_answer

from primagate import FileProfileStore, PrimaGate
from primagate.core.profile import PREP_PROMPT_VERSION
from primagate.drivers.preparation import PREP_RESTART_COOLDOWN, PreparationRunner
from primagate.schema import events
from primagate.schema.policy import TIER_ROUTING_ALIAS
from primagate.schema.profile import PrepState

# Taken before any test can patch it, so a held dispatch can always really wait.
_real_sleep = asyncio.sleep
WAIT = 10.0
INTERIM = "MEDIUM"
CHOSEN = "SMALL"
# A Tier the Analyzer cannot be asked about: only MEDIUM takes images with these bindings.
IMAGE_BINDINGS = {"SMALL": {"model": "claude-sonnet-5-5"}, "MEDIUM": {"model": "gpt-4o"}}


def _asking(n: int) -> dict[str, Any]:
    # A different question each time, so no call is answered from the Cache before routing.
    return {"messages": [{"role": "user", "content": f"question {n}"}]}


def _image(n: int) -> dict[str, Any]:
    content = [
        {"type": "text", "text": f"what is in photo {n}"},
        {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}},
    ]
    return {"messages": [{"role": "user", "content": content}], "drop_params": False}


class Clock:
    def __init__(self) -> None:
        self.now = 5000.0

    def __call__(self) -> float:
        return self.now


def _wait_until(done: Callable[[], bool]) -> None:
    deadline = time.monotonic() + WAIT
    while not done():
        assert time.monotonic() < deadline, "waited too long"
        threading.Event().wait(0.005)


class Served:
    """A gate with declared sources, whose mock can hold the preparation dispatch, and, in async
    mode, one long-lived loop, so a preparation started by one call can finish between calls."""

    def __init__(
        self,
        mode: str,
        replies: list[Reply],
        *,
        hold: bool = False,
        clock: Clock | None = None,
        bindings: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.mode = mode
        self.replies = list(replies)
        self.hold = hold
        self.release = threading.Event()
        self.prepared: list[dict[str, Any]] = []
        self.analyzer_prompts: list[str] = []
        self.workers: list[dict[str, Any]] = []
        # Called with the worker's response for the next worker dispatch only.
        self.on_worker: Callable[[httpx2.Response], Any] | None = None
        self.handler = Recorder()
        self.loop = asyncio.new_event_loop() if mode == "async" else None
        kwargs.setdefault("profile_sources", DECLARED)
        self.gate = PrimaGate(
            api_keys=KEYS,
            transport=httpx2.MockTransport(self._handle),
            tier_bindings=bindings or TWO,
            policy=POLICY,
            analyzer="gpt-4o-mini",
            event_handler=self.handler,
            **kwargs,
        )
        if clock is not None:
            self.gate._runner = PreparationRunner(clock=clock)

    def _handle(self, request: httpx2.Request) -> Any:
        body = json.loads(request.content)
        system = str(body.get("messages", [{}])[0].get("content", ""))
        if f"Prompt version: {PREP_PROMPT_VERSION}" in system:
            self.prepared.append(body)
            reply = self.replies.pop(0)
            response = reply() if callable(reply) else _reply(_text(reply))
            if not self.hold:
                return response
            return self._held(lambda: self.release.is_set(), response)
        if "Prompt version: analyzer" in system:
            self.analyzer_prompts.append(system)
            return _reply(f"{CHOSEN} 0.9")
        self.workers.append(body)
        response = worker_answer("openai", bool(body.get("stream")))
        hook, self.on_worker = self.on_worker, None
        return response if hook is None else hook(response)

    def _held(self, until: Callable[[], bool], response: httpx2.Response) -> Any:
        if self.mode == "sync":
            _wait_until(until)
            return response

        async def later() -> httpx2.Response:
            for _ in range(int(WAIT / 0.005)):
                if until():
                    return response
                await _real_sleep(0.005)
            raise AssertionError("waited too long")

        return later()

    def finish_preparation_mid_call(self, response: httpx2.Response) -> Any:
        """Let the held preparation go, and answer only once it is READY."""
        self.release.set()
        return self._held(lambda: self.state is PrepState.READY, response)

    @property
    def runner(self) -> PreparationRunner | None:
        return self.gate._runner

    @property
    def fingerprint(self) -> str:
        tiers = self.gate._tier_routing
        assert tiers is not None and tiers.fingerprint is not None
        return tiers.fingerprint

    @property
    def state(self) -> PrepState | None:
        assert self.runner is not None
        return self.runner.state(self.fingerprint)

    @property
    def stored(self) -> Any:
        return self.gate._profile_store.read()

    @property
    def types(self) -> list[str]:
        return self.handler.types

    @property
    def decisions(self) -> list[events.RoutingDecision]:
        return [e for e in self.handler.seen if isinstance(e, events.RoutingDecision)]

    @property
    def reasons(self) -> list[str]:
        return [e.reason for e in self.handler.seen if isinstance(e, events.RoutingDegraded)]

    def _in_mode(self, sync: Callable[[], Any], coro: Callable[[], Any]) -> Any:
        if self.loop is None:
            return sync()
        return self.loop.run_until_complete(coro())

    def call(self, **params: Any) -> Any:
        """One routed call, leaving any preparation it started running."""
        merged = {"model": TIER_ROUTING_ALIAS, **_asking(0), **params}

        async def go() -> Any:
            return await self.gate.acompletion(**merged)

        return self._in_mode(lambda: self.gate.completion(**merged), go)

    def stream(self, **params: Any) -> list[Any]:
        merged = {"model": TIER_ROUTING_ALIAS, **_asking(0), **params, "stream": True}

        async def go() -> list[Any]:
            opened = await self.gate.acompletion(**merged)
            return [chunk async for chunk in opened]

        return self._in_mode(lambda: list(self.gate.completion(**merged)), go)

    def settle(self) -> None:
        """Wait for every preparation started so far to end."""
        runner = self.runner
        if runner is None:
            return

        def threads() -> None:
            for thread in list(runner._threads):
                thread.join(WAIT)
                assert not thread.is_alive()

        async def tasks() -> None:
            await asyncio.wait_for(
                asyncio.gather(*list(runner._tasks), return_exceptions=True), WAIT
            )

        self._in_mode(threads, tasks)

    def run(self, **params: Any) -> Any:
        result = self.call(**params)
        self.settle()
        return result

    def finish(self) -> None:
        self.release.set()
        self.settle()

    def close(self) -> None:
        self.finish()
        if self.loop is not None:
            self.loop.close()


def _text(reply: str | dict[str, Any]) -> str:
    return reply if isinstance(reply, str) else json.dumps(reply)


def _tier(answer: Any) -> str | None:
    return answer.metadata.tier if answer.metadata is not None else None


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def make(mode: str) -> Iterator[Callable[..., Served]]:
    made: list[Served] = []

    def build(*args: Any, **kwargs: Any) -> Served:
        served = Served(mode, *args, **kwargs)
        made.append(served)
        return served

    yield build
    for served in made:
        served.close()


# --- PREPARING ---


def test_the_first_call_starts_preparation_and_is_served_by_the_interim_binding(
    make: Callable[..., Served],
) -> None:
    run = make([GOOD], hold=True)
    answer = run.call()
    assert _tier(answer) == INTERIM and run.workers[-1]["model"] == "gpt-4o"
    assert run.analyzer_prompts == []
    assert run.reasons == ["preparing"]
    assert run.state is PrepState.PREPARING
    run.finish()
    assert run.state is PrepState.READY and len(run.prepared) == 1


def test_every_call_while_preparing_goes_to_the_interim_binding_and_starts_nothing(
    make: Callable[..., Served],
) -> None:
    run = make([GOOD], hold=True)
    answers = [run.call(**_asking(n)) for n in range(3)]
    assert [_tier(a) for a in answers] == [INTERIM] * 3
    assert run.analyzer_prompts == []
    assert run.reasons == ["preparing"] * 3
    run.finish()
    assert len(run.prepared) == 1 and run.types.count("preparation_started") == 1


def test_a_stream_while_preparing_opens_on_the_interim_tier(make: Callable[..., Served]) -> None:
    run = make([GOOD], hold=True)
    chunks = run.stream()
    assert chunks
    assert run.workers[-1]["model"] == "gpt-4o" and run.workers[-1]["stream"] is True
    assert run.analyzer_prompts == [] and run.reasons == ["preparing"]


# --- READY ---


def test_a_stream_once_ready_asks_the_analyzer_with_the_profile_and_opens_on_its_choice(
    make: Callable[..., Served],
) -> None:
    run = make([GOOD])
    run.run(**_asking(0))
    assert run.state is PrepState.READY
    chunks = run.stream(**_asking(1))
    assert chunks
    (prompt,) = run.analyzer_prompts
    assert "Purpose: Answers questions about orders." in prompt
    assert "Not known about the app:\n- refund policy" in prompt
    assert run.workers[-1]["model"] == "gpt-4o-mini" and run.workers[-1]["stream"] is True
    assert run.reasons == ["preparing"]
    assert run.decisions[-1].input_fingerprint == run.fingerprint


def test_a_ready_profile_is_what_the_analyzer_reads(make: Callable[..., Served]) -> None:
    run = make([GOOD])
    run.run(**_asking(0))
    assert run.state is PrepState.READY
    answer = run.run(**_asking(1))
    assert _tier(answer) == CHOSEN
    (prompt,) = run.analyzer_prompts
    assert "Purpose: Answers questions about orders." in prompt
    assert "Interactions:\n- order status questions" in prompt
    assert "Capabilities:\n- look up an order" in prompt
    # The prepared gaps, not the placeholder's list of everything.
    assert "Not known about the app:\n- refund policy" in prompt
    assert "- purpose" not in prompt


def test_a_ready_decision_names_its_profile(make: Callable[..., Served]) -> None:
    run = make([GOOD])
    run.run(**_asking(0))
    run.run(**_asking(1))
    stored = run.stored
    assert stored is not None
    before, after = run.decisions
    assert (before.profile_id, before.input_fingerprint, before.prep_prompt_version) == (
        "unknown",
        "unknown",
        "unknown",
    )
    assert after.profile_id == stored.profile_id
    assert after.input_fingerprint == run.fingerprint
    assert after.prep_prompt_version == "prep-1"
    assert run.reasons == ["preparing"]


def test_a_call_in_flight_keeps_preparing_and_the_next_uses_ready(
    make: Callable[..., Served],
) -> None:
    run = make([GOOD], hold=True)
    run.on_worker = run.finish_preparation_mid_call
    answer = run.call(**_asking(0))
    # The preparation was READY before this call returned, and it still ended on Interim.
    assert run.state is PrepState.READY
    assert _tier(answer) == INTERIM and run.analyzer_prompts == []
    assert run.reasons == ["preparing"]
    later = run.run(**_asking(1))
    assert _tier(later) == CHOSEN and len(run.analyzer_prompts) == 1
    assert run.reasons == ["preparing"]


# --- the Cache ---


def test_an_interim_answer_is_not_replayed_once_ready(make: Callable[..., Served]) -> None:
    run = make([GOOD], hold=True)
    assert _tier(run.call(**_asking(0))) == INTERIM
    run.finish()
    again = run.run(**_asking(0))
    assert again.metadata is not None and not again.metadata.cache_hit
    assert _tier(again) == CHOSEN and len(run.analyzer_prompts) == 1
    assert "cache_hit" not in run.types


def test_a_ready_answer_is_cached_as_before(make: Callable[..., Served]) -> None:
    run = make([GOOD])
    run.run(**_asking(0))
    assert _tier(run.run(**_asking(1))) == CHOSEN
    again = run.run(**_asking(1))
    assert again.metadata is not None and again.metadata.cache_hit
    assert len(run.analyzer_prompts) == 1 and run.types.count("cache_hit") == 1


# --- FAILED ---


def test_after_a_final_failure_calls_stay_on_the_interim_binding(
    make: Callable[..., Served],
) -> None:
    run = make([_status(401)])
    run.run(**_asking(0))
    assert run.state is PrepState.FAILED
    (failed,) = [e for e in run.handler.seen if isinstance(e, events.PreparationFailed)]
    assert failed.final
    answers = [run.run(**_asking(n)) for n in (1, 2)]
    assert [_tier(a) for a in answers] == [INTERIM, INTERIM]
    assert run.analyzer_prompts == []
    assert run.reasons == ["preparing", "preparation_failed", "preparation_failed"]


def test_a_restart_after_the_cooldown_switches_calls_to_the_analyzer(
    make: Callable[..., Served],
) -> None:
    clock = Clock()
    run = make([_status(401), GOOD], clock=clock)
    run.run(**_asking(0))
    run.run(**_asking(1))
    clock.now += PREP_RESTART_COOLDOWN
    # This call starts the restart, so it is served while that preparation runs.
    assert _tier(run.run(**_asking(2))) == INTERIM
    assert run.state is PrepState.READY
    answer = run.run(**_asking(3))
    assert _tier(answer) == CHOSEN
    (prompt,) = run.analyzer_prompts
    assert "Purpose: Answers questions about orders." in prompt
    assert run.reasons == ["preparing", "preparation_failed", "preparing"]


# --- restart with a stored Profile ---


def test_a_stored_profile_is_served_from_the_second_call_after_a_restart(
    make: Callable[..., Served], tmp_path: Path
) -> None:
    path = tmp_path / "profile.json"
    first = make([GOOD], profile_store=FileProfileStore(path))
    first.run(**_asking(0))
    written = first.stored
    assert written is not None

    second = make([], profile_store=FileProfileStore(path))
    # No call reads the store, so the first one is served while the job loads it.
    assert _tier(second.run(**_asking(0))) == INTERIM
    assert second.reasons == ["preparing"] and second.state is PrepState.READY
    answer = second.run(**_asking(1))
    assert _tier(answer) == CHOSEN and second.prepared == []
    assert second.decisions[-1].profile_id == written.profile_id
    assert "preparation_started" not in second.types


# --- unchanged routing ---


def test_without_sources_the_analyzer_reads_the_empty_profile(
    make: Callable[..., Served],
) -> None:
    run = make([], profile_sources=None)
    answer = run.run()
    assert _tier(answer) == CHOSEN and run.reasons == []
    (prompt,) = run.analyzer_prompts
    assert "Not known about the app:\n- purpose" in prompt
    (decision,) = run.decisions
    assert (decision.profile_id, decision.input_fingerprint, decision.prep_prompt_version) == (
        "unknown",
        "unknown",
        "unknown",
    )


def test_a_call_only_one_tier_can_take_is_routed_to_it_whatever_the_state(
    make: Callable[..., Served],
) -> None:
    run = make([GOOD], hold=True, bindings=IMAGE_BINDINGS)
    assert _tier(run.call(**_asking(0))) == INTERIM
    assert run.state is PrepState.PREPARING
    assert _tier(run.call(**_image(0))) == "MEDIUM"
    run.finish()
    assert run.state is PrepState.READY
    assert _tier(run.call(**_image(1))) == "MEDIUM"
    # Only the first call, which two Tiers could take, was degraded or asked to prepare.
    assert run.reasons == ["preparing"] and run.analyzer_prompts == []
    assert len(run.decisions) == 3
