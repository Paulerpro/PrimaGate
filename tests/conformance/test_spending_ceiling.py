import logging
from typing import Any

import pytest
from conftest import Caller, StreamCaller
from test_tier_routing import POLICY, Run

from primagate.schema import events

pytestmark = pytest.mark.fr("FR35")

ALIAS = "primagate/auto"
# One routed call costs about 4e-5 dollars here: gpt-4o's answer plus gpt-4o-mini's analysis.
ONE_CALL = 4.0e-5


def _msg(n: int) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"question {n}"}]


def _run(ceiling: float | None, verdicts: list[Any] | None = None, **kwargs: Any) -> Run:
    policy = dict(POLICY) if ceiling is None else {**POLICY, "spending_ceiling": ceiling}
    return Run(verdicts or ["MEDIUM 0.9"], policy=policy, **kwargs)


def _crossings(run: Run) -> list[events.SpendingCeilingExceeded]:
    return run.of(events.SpendingCeilingExceeded)


def test_staying_under_the_ceiling_emits_nothing(call: Caller) -> None:
    run = _run(1.0)
    for n in range(3):
        call(run.gate, model=ALIAS, messages=_msg(n))
    assert _crossings(run) == []


def test_the_call_that_passes_the_ceiling_emits_one_event_and_one_warning(
    call: Caller, caplog: pytest.LogCaptureFixture
) -> None:
    run = _run(ONE_CALL * 1.5)
    with caplog.at_level(logging.WARNING, logger="primagate"):
        call(run.gate, model=ALIAS, messages=_msg(1))
        assert _crossings(run) == []
        result = call(run.gate, model=ALIAS, messages=_msg(2))
    (event,) = _crossings(run)
    assert event.ceiling == ONE_CALL * 1.5 and event.total > event.ceiling
    assert result.choices and result.metadata is not None
    warned = [r for r in caplog.records if "spending_ceiling_exceeded" in r.getMessage()]
    assert len(warned) == 1 and warned[0].levelno == logging.WARNING


def test_the_event_comes_after_the_call_succeeded_and_holds_numbers_only(call: Caller) -> None:
    run = _run(ONE_CALL * 0.5)
    call(run.gate, model=ALIAS, messages=_msg(1))
    assert run.handler.types[-2:] == ["call_succeeded", "spending_ceiling_exceeded"]
    shown = _crossings(run)[0].to_dict()
    assert set(shown) == {"event_type", "request_id", "timestamp", "ceiling", "total"}


def test_it_is_sent_once_however_far_over_the_total_goes(call: Caller) -> None:
    run = _run(ONE_CALL * 0.5)
    for n in range(4):
        call(run.gate, model=ALIAS, messages=_msg(n))
    assert len(_crossings(run)) == 1


def test_the_call_after_crossing_is_still_routed_to_its_resolved_tier(call: Caller) -> None:
    run = _run(ONE_CALL * 0.5, ["MEDIUM 0.9", "SMALL 0.9"])
    call(run.gate, model=ALIAS, messages=_msg(1))
    after = call(run.gate, model=ALIAS, messages=_msg(2))
    assert [w["model"] for w in run.workers] == ["gpt-4o", "gpt-4o-mini"]
    assert after.metadata is not None and after.metadata.tier == "SMALL"
    assert [e.reason for e in run.of(events.RoutingDegraded)] == []


def test_the_analyzers_cost_counts_toward_the_total(call: Caller) -> None:
    # The worker alone costs about 2.25e-5, so crossing 3e-5 on one call needs the Analyzer too.
    run = _run(3.0e-5)
    call(run.gate, model=ALIAS, messages=_msg(1))
    assert len(_crossings(run)) == 1


def test_a_degraded_call_counts_like_any_routed_call(call: Caller) -> None:
    run = _run(3.0e-5, ["nonsense"])
    call(run.gate, model=ALIAS, messages=_msg(1))
    assert len(run.of(events.RoutingDegraded)) == 1 and len(_crossings(run)) == 1


def test_with_no_ceiling_nothing_is_counted_or_emitted(call: Caller) -> None:
    run = _run(None)
    call(run.gate, model=ALIAS, messages=_msg(1))
    assert _crossings(run) == [] and run.gate._spend.total == 0.0


def test_a_repeat_answered_from_memory_adds_nothing(call: Caller) -> None:
    run = _run(1.0)
    call(run.gate, model=ALIAS, messages=_msg(1))
    counted = run.gate._spend.total
    call(run.gate, model=ALIAS, messages=_msg(1))
    assert counted > 0 and run.gate._spend.total == counted


def test_a_plain_model_call_adds_nothing(call: Caller) -> None:
    run = _run(1.0)
    call(run.gate, model="gpt-4o", messages=_msg(1))
    assert run.gate._spend.total == 0.0


def test_a_model_with_no_price_adds_nothing(call: Caller) -> None:
    one = {**POLICY, "enabled_tiers": ["MEDIUM"], "ambiguous_context_default": "MEDIUM"}
    one["spending_ceiling"] = 1e-12
    run = Run(["MEDIUM 0.9"], policy=one, bindings={"MEDIUM": {"model": "mystery-model"}})
    call(run.gate, model=ALIAS, messages=_msg(1))
    assert run.gate._spend.total == 0.0 and _crossings(run) == []


def test_a_stream_is_not_counted(stream_call: StreamCaller) -> None:
    run = _run(1e-12)
    result = stream_call(run.gate, model=ALIAS, messages=_msg(1))
    assert result.call_error is None and _crossings(run) == []


def test_two_instances_keep_separate_totals(call: Caller) -> None:
    a, b = _run(1.0), _run(1.0)
    call(a.gate, model=ALIAS, messages=_msg(1))
    assert a.gate._spend.total > 0 and b.gate._spend.total == 0.0


@pytest.mark.mode_independent("drives the sync and the async driver in one body")
def test_one_total_serves_both_drivers_of_an_instance() -> None:
    run = _run(1.0)
    assert run.gate._get_driver()._spend is run.gate._get_async_driver()._spend is run.gate._spend


def test_a_call_with_only_a_priced_worker_is_counted(call: Caller) -> None:
    one = {**POLICY, "enabled_tiers": ["MEDIUM"], "ambiguous_context_default": "MEDIUM"}
    run = Run(
        ["MEDIUM 0.9"],
        policy={**one, "spending_ceiling": 1.0},
        bindings={"MEDIUM": {"model": "gpt-4o"}},
    )
    call(run.gate, model=ALIAS, messages=_msg(1))
    # No Analyzer was asked, so this is the worker's answer alone (about 2.25e-5 dollars).
    assert run.analyzed == [] and 1e-5 < run.gate._spend.total < 4e-5


def test_a_call_with_only_a_priced_analyzer_is_counted(call: Caller) -> None:
    unpriced = {"SMALL": {"model": "mystery-a"}, "MEDIUM": {"model": "mystery-b"}}
    run = Run(["MEDIUM 0.9"], policy={**POLICY, "spending_ceiling": 1e-5}, bindings=unpriced)
    call(run.gate, model=ALIAS, messages=_msg(1))
    # Only the Analyzer (gpt-4o-mini) has a price, and that alone passes this small ceiling.
    assert len(_crossings(run)) == 1


@pytest.mark.mode_independent("checks the spend step directly; makes no call")
def test_a_non_finite_or_negative_amount_never_reaches_the_total() -> None:
    import math

    from primagate.core.call import _spent
    from primagate.core.steps import Dispatch, Return
    from primagate.schema.tiers import Tier
    from primagate.schema.types import ChatCompletion, ChatCompletionRequest, Cost, Metadata

    def finished(total: float) -> Return:
        meta = Metadata(cost=Cost(0.0, 0.0, total))
        return Return(ChatCompletion("c", 1, "m", [], None, metadata=meta))

    request = ChatCompletionRequest(model="m", messages=[])
    action = Dispatch("openai", request, tier=Tier.SMALL, spending_ceiling=1.0)
    for bad in (math.nan, math.inf, -1.0):
        done = finished(bad)
        assert _spent(action, done) is done
