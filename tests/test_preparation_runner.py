import asyncio
import dataclasses
import threading
from typing import Any

import pytest

from primagate.core import steps
from primagate.core.call import advance, start
from primagate.core.policy import build_policy
from primagate.core.profile import PREP_PROMPT_VERSION
from primagate.drivers.preparation import (
    PREP_RESTART_COOLDOWN,
    PreparationRunner,
    Progress,
)
from primagate.drivers.providers import parse_tier_bindings
from primagate.schema import events
from primagate.schema.policy import TIER_ROUTING_ALIAS, TierRouting
from primagate.schema.profile import (
    PROFILE_SCHEMA_VERSION,
    PrepState,
    Profile,
    ProfileContent,
    ProfileSources,
)
from primagate.schema.types import ChatCompletion, ChatCompletionRequest, Choice, Message
from primagate.schema.wire import Capabilities

FP = "fp-1"


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def runner(clock: Clock) -> PreparationRunner:
    return PreparationRunner(clock=clock)


# --- the restart rule and try_start ---


def test_the_cooldown_is_ten_minutes() -> None:
    assert PREP_RESTART_COOLDOWN == 600.0


def test_the_first_start_wins_and_marks_preparing(runner: PreparationRunner) -> None:
    assert runner.state(FP) is None
    assert runner.try_start(FP) is True
    assert runner.state(FP) is PrepState.PREPARING
    assert runner.try_start(FP) is False


def test_fingerprints_are_kept_apart(runner: PreparationRunner) -> None:
    assert runner.try_start("a") and runner.try_start("b")


def test_ready_never_starts_again(runner: PreparationRunner, clock: Clock) -> None:
    runner.try_start(FP)
    runner.finish(FP, PrepState.READY, profile=READY)
    clock.now += 10 * PREP_RESTART_COOLDOWN
    assert runner.try_start(FP) is False
    assert runner.state(FP) is PrepState.READY


@pytest.mark.parametrize("reason", ["provider_error", "malformed_output", "store_error"])
def test_a_failure_restarts_only_after_the_cooldown(
    runner: PreparationRunner, clock: Clock, reason: Any
) -> None:
    runner.try_start(FP)
    runner.finish(FP, PrepState.FAILED, reason)
    clock.now += 300
    assert runner.try_start(FP) is False
    assert runner.state(FP) is PrepState.FAILED
    clock.now += 299.9
    assert runner.try_start(FP) is False
    clock.now += 0.2
    assert runner.try_start(FP) is True
    assert runner.state(FP) is PrepState.PREPARING
    assert runner.try_start(FP) is False


def test_an_overflow_failure_is_never_restarted(runner: PreparationRunner, clock: Clock) -> None:
    runner.try_start(FP)
    runner.finish(FP, PrepState.FAILED, "overflow")
    clock.now += 1_000_000
    assert runner.try_start(FP) is False


def test_the_cooldown_counts_from_the_latest_failure(
    runner: PreparationRunner, clock: Clock
) -> None:
    runner.try_start(FP)
    runner.finish(FP, PrepState.FAILED, "provider_error")
    clock.now += PREP_RESTART_COOLDOWN
    assert runner.try_start(FP)
    runner.finish(FP, PrepState.FAILED, "provider_error")
    clock.now += PREP_RESTART_COOLDOWN - 1
    assert runner.try_start(FP) is False


def test_nothing_starts_after_close(runner: PreparationRunner) -> None:
    runner.close()
    assert runner.is_closed
    assert runner.try_start(FP) is False
    assert runner.start_thread(FP, lambda: None) is False


def test_twenty_threads_start_exactly_one(runner: PreparationRunner) -> None:
    barrier = threading.Barrier(20)
    won: list[bool] = []
    lock = threading.Lock()

    def race() -> None:
        barrier.wait()
        result = runner.try_start(FP)
        with lock:
            won.append(result)

    threads = [threading.Thread(target=race) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert won.count(True) == 1 and len(won) == 20


def test_the_outcome_hands_a_call_the_state_and_the_ready_profile(
    runner: PreparationRunner,
) -> None:
    # Nothing known yet, as for a closed runner or an abandoned start: no Profile is coming.
    assert runner.outcome(FP, False) == steps.StartOutcome(False, PrepState.PREPARING)
    assert runner.try_start(FP)
    assert runner.outcome(FP, True) == steps.StartOutcome(True, PrepState.PREPARING)
    runner.finish(FP, PrepState.READY, profile=READY)
    # Kept in memory, so serving it reads no store.
    assert runner.outcome(FP, False) == steps.StartOutcome(False, PrepState.READY, READY)
    # Starting it counts even when the job has already finished, so the call is PREPARING.
    assert runner.outcome(FP, True) == steps.StartOutcome(True, PrepState.PREPARING)


def test_ready_without_its_profile_is_refused_and_changes_nothing(
    runner: PreparationRunner,
) -> None:
    runner.try_start(FP)
    with pytest.raises(ValueError, match="needs its Profile"):
        runner.finish(FP, PrepState.READY)
    assert runner.state(FP) is PrepState.PREPARING


def test_a_failed_fingerprint_hands_over_no_profile(
    runner: PreparationRunner, clock: Clock
) -> None:
    runner.try_start(FP)
    runner.finish(FP, PrepState.FAILED, "provider_error")
    assert runner.outcome(FP, False) == steps.StartOutcome(False, PrepState.FAILED)
    clock.now += PREP_RESTART_COOLDOWN
    assert runner.try_start(FP)
    runner.finish(FP, PrepState.READY, profile=READY)
    assert runner.outcome(FP, False).profile == READY


def test_abandon_frees_only_a_running_preparation(runner: PreparationRunner) -> None:
    runner.try_start(FP)
    runner.abandon(FP)
    assert runner.state(FP) is None and runner.try_start(FP)
    runner.finish(FP, PrepState.READY, profile=READY)
    runner.abandon(FP)
    assert runner.state(FP) is PrepState.READY


# --- where the job runs ---


def test_a_thread_job_runs_on_a_daemon_thread(runner: PreparationRunner) -> None:
    ran: list[threading.Thread] = []
    assert runner.start_thread(FP, lambda: ran.append(threading.current_thread()))
    for thread in list(runner._threads):
        thread.join(5)
    (thread,) = ran
    assert thread.daemon and thread is not threading.main_thread()
    assert runner._threads == set()


def test_a_thread_job_is_not_run_when_it_may_not_start(runner: PreparationRunner) -> None:
    ran: list[int] = []
    runner.try_start(FP)
    assert runner.start_thread(FP, lambda: ran.append(1)) is False
    assert ran == []


async def test_a_task_job_runs_on_the_callers_loop(runner: PreparationRunner) -> None:
    seen: list[asyncio.AbstractEventLoop] = []

    async def job() -> None:
        seen.append(asyncio.get_running_loop())

    loop = asyncio.get_running_loop()
    assert runner.start_task(FP, loop, job)
    (task,) = runner._tasks
    await task
    await asyncio.sleep(0)
    assert seen == [loop]
    assert runner._tasks == set()


async def test_a_task_is_held_until_it_is_done(runner: PreparationRunner) -> None:
    gate = asyncio.Event()

    async def job() -> None:
        await gate.wait()

    runner.start_task(FP, asyncio.get_running_loop(), job)
    await asyncio.sleep(0)
    assert len(runner._tasks) == 1
    gate.set()
    await asyncio.gather(*runner._tasks)
    await asyncio.sleep(0)
    assert runner._tasks == set()


async def test_a_task_not_allowed_to_start_is_never_created(runner: PreparationRunner) -> None:
    made: list[int] = []

    async def job() -> None:
        made.append(1)

    runner.try_start(FP)
    assert runner.start_task(FP, asyncio.get_running_loop(), job) is False
    assert runner._tasks == set() and made == []


async def test_aclose_cancels_and_awaits_its_tasks(runner: PreparationRunner) -> None:
    started = asyncio.Event()
    cancelled: list[bool] = []

    async def job() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    runner.start_task(FP, asyncio.get_running_loop(), job)
    await started.wait()
    (task,) = runner._tasks
    await runner.aclose()
    assert task.cancelled() and cancelled == [True]
    assert runner.is_closed and runner.try_start("other") is False


def test_a_task_cut_off_by_its_loop_ending_is_a_failure_with_the_cooldown(
    runner: PreparationRunner, clock: Clock
) -> None:
    async def job() -> None:
        await asyncio.Event().wait()

    async def call() -> None:
        assert runner.start_task(FP, asyncio.get_running_loop(), job)
        await asyncio.sleep(0)

    asyncio.run(call())
    assert runner.state(FP) is PrepState.FAILED and runner._tasks == set()
    clock.now += PREP_RESTART_COOLDOWN - 1
    assert runner.try_start(FP) is False
    clock.now += 1
    assert runner.try_start(FP)


def test_a_thread_that_cannot_start_leaves_nothing_behind(
    runner: PreparationRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(self: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    ran: list[int] = []
    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", refuse)
        assert runner.start_thread(FP, lambda: ran.append(1)) is False
    assert runner.state(FP) is None and runner._threads == set() and ran == []
    assert runner.try_start(FP)


def test_a_loop_that_cannot_take_a_task_leaves_nothing_behind(runner: PreparationRunner) -> None:
    made: list[Any] = []

    async def job() -> None:
        pass

    def make() -> Any:
        made.append(job())
        return made[-1]

    class Closing:
        def create_task(self, work: Any) -> Any:
            raise RuntimeError("loop is closed")

    assert runner.start_task(FP, Closing(), make) is False  # type: ignore[arg-type]
    (work,) = made
    # Closed, so it can never be reported as a coroutine that was never awaited.
    assert work.cr_frame is None
    assert runner.state(FP) is None and runner._tasks == set()
    assert runner.try_start(FP)


# --- what a job records as it goes ---


def test_progress_names_the_prep_id_attempts_and_final_reason() -> None:
    progress = Progress()
    started = events.PreparationStarted("prep-abc", FP, PREP_PROMPT_VERSION)
    progress.observe(steps.Emit((started,), steps.Return(None)))  # type: ignore[arg-type]
    dispatch = steps.Dispatch("openai", ChatCompletionRequest(model="m", messages=[]))
    progress.observe(steps.Analyze(dispatch, lambda _: steps.Return(None)))  # type: ignore[arg-type]
    final = events.PreparationFailed("prep-abc", FP, PREP_PROMPT_VERSION, "overflow", 1, True)
    progress.observe(steps.Emit((final,), steps.Return(None)))  # type: ignore[arg-type]
    assert (progress.prep_id, progress.attempts, progress.reason) == ("prep-abc", 1, "overflow")
    failed = progress.failed(FP, "store_error")
    assert (failed.request_id, failed.reason, failed.attempt, failed.final) == (
        "prep-abc",
        "store_error",
        1,
        True,
    )


def test_a_failure_before_any_id_still_gets_a_prep_id() -> None:
    failed = Progress().failed(FP, "store_error")
    assert failed.request_id.startswith("prep-") and len(failed.request_id) > 5
    assert failed.attempt == 0 and failed.input_fingerprint == FP


# --- core asks for it once, before analysing ---


def _tiers(fingerprint: str | None, small: str = "gpt-4o-mini") -> TierRouting:
    bindings = parse_tier_bindings({"SMALL": {"model": small}, "MEDIUM": {"model": "gpt-4o"}})
    policy, _ = build_policy(
        {
            "enabled_tiers": ["SMALL", "MEDIUM"],
            "confidence_threshold": 0.8,
            "context_token_budget": 1000,
            "interim_binding": "MEDIUM",
        },
        bindings,
    )
    sources = ProfileSources("x") if fingerprint else None
    return TierRouting(policy, bindings, "gpt-4o-mini", sources, fingerprint)


CAPS = {"openai": Capabilities(), "anthropic": Capabilities()}


READY = Profile(
    "prep-abc",
    FP,
    PROFILE_SCHEMA_VERSION,
    "2026-11-17T10:00:00+00:00",
    ProfileContent(purpose="Answers questions about orders.", unknowns=()),
    "prep-1;gpt-4o-mini",
)


def _drive(
    model: str,
    tiers: TierRouting,
    started: bool = True,
    state: PrepState = PrepState.READY,
    *,
    caps: dict[str, Capabilities] = CAPS,
    tools: list[dict[str, Any]] | None = None,
    hit: ChatCompletion | None = None,
) -> list[steps.Action]:
    """Run a call through core, answering each ask; the Analyzer declines so it degrades.

    A READY answer carries its Profile, as the runner's does.
    """
    seen: list[steps.Action] = []
    action: steps.Action = start(
        ChatCompletionRequest(
            model=model, messages=[Message(role="user", content="hi")], tools=tools
        ),
        caps,
        tier_routing=tiers,
        cache_salt=b"s",
    )
    while not isinstance(action, steps.Dispatch | steps.Raise | steps.Return):
        seen.append(action)
        outcome: steps.Outcome
        if isinstance(action, steps.NewId):
            outcome = steps.IdOutcome("req-1")
        elif isinstance(action, steps.CacheRead):
            outcome = steps.CacheReadOutcome(hit)
        elif isinstance(action, steps.Emit):
            outcome = steps.EmitOutcome()
        elif isinstance(action, steps.StartPreparation):
            outcome = steps.StartOutcome(
                started, state, READY if state is PrepState.READY else None
            )
        elif isinstance(action, steps.Analyze):
            outcome = steps.Declined("over_capacity")
        else:
            raise AssertionError(f"unexpected {type(action).__name__}")
        action = advance(action, outcome)
    seen.append(action)
    return seen


def test_start_preparation_is_in_the_closed_set() -> None:
    assert steps.StartPreparation in steps.ACTION_TYPES


def test_a_routed_call_with_sources_asks_once_before_analysing() -> None:
    seen = _drive(TIER_ROUTING_ALIAS, _tiers(FP))
    kinds = [type(a) for a in seen]
    assert kinds.count(steps.StartPreparation) == 1
    assert kinds.index(steps.StartPreparation) < kinds.index(steps.Analyze)
    (asked,) = [a for a in seen if isinstance(a, steps.StartPreparation)]
    assert asked.fingerprint == FP


@pytest.mark.parametrize("state", list(PrepState))
def test_the_call_goes_on_the_same_way_whether_or_not_it_started_one(state: PrepState) -> None:
    def shape(actions: list[steps.Action]) -> list[Any]:
        out: list[Any] = []
        for a in actions:
            if isinstance(a, steps.Analyze):
                out.append(("analyze", a.dispatch))
            elif isinstance(a, steps.Emit):
                out.append(("emit", a.events))
            elif isinstance(a, steps.Dispatch):
                out.append(("dispatch", dataclasses.replace(a, plan=None)))
            else:
                out.append(type(a))
        return out

    tiers = _tiers(FP)
    assert shape(_drive(TIER_ROUTING_ALIAS, tiers, True, state)) == shape(
        _drive(TIER_ROUTING_ALIAS, tiers, False, state)
    )


def test_no_sources_means_no_ask() -> None:
    seen = _drive(TIER_ROUTING_ALIAS, _tiers(None))
    assert steps.Analyze in [type(a) for a in seen]
    assert not any(isinstance(a, steps.StartPreparation) for a in seen)


def test_a_plain_model_call_never_asks() -> None:
    seen = _drive("gpt-4o", _tiers(FP))
    assert not any(isinstance(a, steps.StartPreparation) for a in seen)


def test_a_call_only_one_tier_can_take_never_asks() -> None:
    tool = {"type": "function", "function": {"name": "lookup"}}
    caps = {"openai": Capabilities(tools=True), "anthropic": Capabilities()}
    seen = _drive(
        TIER_ROUTING_ALIAS,
        _tiers(FP, small="anthropic/claude-3-5-haiku-latest"),
        caps=caps,
        tools=[tool],
    )
    assert isinstance(seen[-1], steps.Dispatch) and seen[-1].request.model == "gpt-4o"
    assert not any(isinstance(a, steps.StartPreparation | steps.Analyze) for a in seen)


def test_a_call_answered_from_the_cache_never_asks() -> None:
    answered = ChatCompletion(
        id="c",
        created=1,
        model="gpt-4o",
        choices=[Choice(index=0, message=Message(role="assistant", content="hello"))],
    )
    seen = _drive(TIER_ROUTING_ALIAS, _tiers(FP), hit=answered)
    assert isinstance(seen[-1], steps.Return) and seen[-1].response.metadata is not None
    assert seen[-1].response.metadata.cache_hit
    assert not any(isinstance(a, steps.StartPreparation) for a in seen)
