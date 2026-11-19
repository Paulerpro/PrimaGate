import threading
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller
from test_tier_routing import MSGS, POLICY, TWO, Run, _analyzer_reply

from primagate.drivers.capacity import DEFAULT_MAX_CONCURRENT_ANALYZER_CALLS, AnalyzerCap
from primagate.schema import events

ALIAS = "primagate/auto"


def _cap(run: Run) -> AnalyzerCap:
    cap = run.gate._analyzer_cap
    assert cap is not None
    return cap


def _hold(cap: AnalyzerCap) -> None:
    while cap.try_acquire():
        pass


def _reasons(run: Run) -> list[str]:
    return [e.reason for e in run.of(events.RoutingDegraded)]


@pytest.mark.fr("FR22", "FR28")
def test_a_call_over_the_cap_goes_to_the_interim_binding_without_asking_the_analyzer(
    call: Caller,
) -> None:
    run = Run(["SMALL 0.9"], policy={**POLICY, "max_concurrent_analyzer_calls": 2})
    _hold(_cap(run))
    result = call(run.gate, model=ALIAS, messages=MSGS)
    assert run.analyzed == [] and [w["model"] for w in run.workers] == ["gpt-4o"]
    assert _reasons(run) == ["over_capacity"]
    assert result.metadata is not None and result.metadata.tier == "MEDIUM"
    assert result.metadata.analyzer is None


@pytest.mark.fr("FR28")
def test_an_over_capacity_event_says_the_analyzer_was_not_asked(call: Caller) -> None:
    run = Run(["SMALL 0.9"], policy={**POLICY, "max_concurrent_analyzer_calls": 1})
    _hold(_cap(run))
    call(run.gate, model=ALIAS, messages=MSGS)
    (event,) = run.of(events.RoutingDegraded)
    assert event.reason == "over_capacity" and event.binding_id == "MEDIUM:openai/gpt-4o"
    assert event.analyzer_prompt_version == "unknown"


@pytest.mark.fr("FR28")
def test_under_the_cap_the_analyzer_is_asked_and_the_permit_comes_back(call: Caller) -> None:
    run = Run(["SMALL 0.9"], policy={**POLICY, "max_concurrent_analyzer_calls": 2})
    cap = _cap(run)
    assert cap.try_acquire()
    call(run.gate, model=ALIAS, messages=MSGS)
    assert len(run.analyzed) == 1 and _reasons(run) == []
    assert cap.in_flight == 1


@pytest.mark.fr("FR28")
def test_a_permit_freed_lets_the_next_call_ask_again(call: Caller) -> None:
    run = Run(["SMALL 0.9"], policy={**POLICY, "max_concurrent_analyzer_calls": 1})
    cap = _cap(run)
    assert cap.try_acquire()
    call(run.gate, model=ALIAS, messages=MSGS)
    cap.release()
    call(run.gate, model=ALIAS, messages=[{"role": "user", "content": "another"}])
    assert len(run.analyzed) == 1 and _reasons(run) == ["over_capacity"]


def _down() -> httpx2.Response:
    return httpx2.Response(503, json={"error": {"message": "x", "type": "server_error"}})


def _timeout() -> httpx2.Response:
    raise httpx2.ReadTimeout("slow")


@pytest.mark.parametrize("verdict", [_down, _timeout, "nonsense", "SMALL 0.9"])
@pytest.mark.fr("FR28")
def test_the_permit_comes_back_whatever_the_analyzer_does(call: Caller, verdict: Any) -> None:
    run = Run([verdict], policy={**POLICY, "max_concurrent_analyzer_calls": 1})
    call(run.gate, model=ALIAS, messages=MSGS)
    assert _cap(run).in_flight == 0


class Cancelled(BaseException):
    """Stands in for a cancellation, which is not an `Exception` and skips every `except`."""


@pytest.mark.fr("FR27", "FR28")
def test_the_permit_comes_back_when_the_request_is_cancelled(call: Caller) -> None:
    def cancel() -> httpx2.Response:
        raise Cancelled

    run = Run([cancel], policy={**POLICY, "max_concurrent_analyzer_calls": 1})
    with pytest.raises(Cancelled):
        call(run.gate, model=ALIAS, messages=MSGS)
    assert _cap(run).in_flight == 0


@pytest.mark.fr("FR28")
def test_calls_that_never_reach_the_analyzer_take_no_permit(call: Caller) -> None:
    run = Run(["SMALL 0.9"], policy={**POLICY, "max_concurrent_analyzer_calls": 1})
    call(run.gate, model=ALIAS, messages=MSGS)
    _hold(_cap(run))
    # A repeat is answered from memory, so the full cap does not touch it.
    again = call(run.gate, model=ALIAS, messages=MSGS)
    assert again.metadata is not None and again.metadata.cache_hit and _reasons(run) == []
    # Nor does a request too large for the Analyzer to read: that is its own reason.
    tight = Run(["SMALL 0.9"], policy={**POLICY, "context_token_budget": 2})
    _hold(_cap(tight))
    call(tight.gate, model=ALIAS, messages=MSGS)
    assert _reasons(tight) == ["over_budget"]


@pytest.mark.fr("FR25", "FR28")
def test_one_able_tier_needs_no_permit(call: Caller) -> None:
    one = {**POLICY, "enabled_tiers": ["MEDIUM"], "ambiguous_context_default": "MEDIUM"}
    run = Run(["SMALL 0.9"], policy=one, bindings={"MEDIUM": {"model": "gpt-4o"}})
    _hold(_cap(run))
    call(run.gate, model=ALIAS, messages=MSGS)
    assert _reasons(run) == [] and len(run.workers) == 1


@pytest.mark.fr("FR28", "FR31")
def test_an_over_capacity_answer_is_never_cached(call: Caller) -> None:
    run = Run(["SMALL 0.9"], policy={**POLICY, "max_concurrent_analyzer_calls": 1})
    cap = _cap(run)
    assert cap.try_acquire()
    call(run.gate, model=ALIAS, messages=MSGS)
    cap.release()
    second = call(run.gate, model=ALIAS, messages=MSGS)
    assert second.metadata is not None and not second.metadata.cache_hit
    assert len(run.analyzed) == 1 and len(run.workers) == 2


@pytest.mark.fr("FR22", "FR28")
def test_a_stream_over_the_cap_opens_on_the_interim_tier(stream_call: StreamCaller) -> None:
    run = Run(["SMALL 0.9"], policy={**POLICY, "max_concurrent_analyzer_calls": 1})
    _hold(_cap(run))
    result = stream_call(run.gate, model=ALIAS, messages=MSGS)
    assert result.call_error is None and result.chunks
    assert run.analyzed == [] and _reasons(run) == ["over_capacity"]


@pytest.mark.fr("FR28")
@pytest.mark.mode_independent("inspects the instance; makes no call")
def test_the_limit_is_the_policys_or_the_named_default() -> None:
    assert _cap(Run(["SMALL 0.9"])).limit == DEFAULT_MAX_CONCURRENT_ANALYZER_CALLS
    explicit = Run(["SMALL 0.9"], policy={**POLICY, "max_concurrent_analyzer_calls": 3})
    assert _cap(explicit).limit == 3


@pytest.mark.fr("FR28")
@pytest.mark.mode_independent("drives the sync and the async driver in one body")
def test_one_cap_serves_the_sync_and_the_async_driver_of_an_instance() -> None:
    run = Run(["SMALL 0.9"])
    sync, async_ = run.gate._get_driver(), run.gate._get_async_driver()
    assert sync._analyzer_cap is async_._analyzer_cap is _cap(run)


@pytest.mark.fr("FR28")
@pytest.mark.mode_independent("inspects the instance; makes no call")
def test_instances_do_not_share_a_cap() -> None:
    assert _cap(Run(["SMALL 0.9"])) is not _cap(Run(["SMALL 0.9"]))


@pytest.mark.fr("FR28")
@pytest.mark.mode_independent("inspects the instance; makes no call")
def test_a_gate_without_a_policy_has_no_cap() -> None:
    from primagate import PrimaGate

    assert PrimaGate()._analyzer_cap is None


@pytest.mark.fr("FR22", "FR28")
@pytest.mark.mode_independent("races two real threads, which needs the blocking sync driver")
def test_a_real_burst_sends_one_analyzer_request_and_the_other_degrades_without_waiting() -> None:
    inside = threading.Event()
    let_go = threading.Event()

    def slow() -> httpx2.Response:
        inside.set()
        assert let_go.wait(5)
        return _analyzer_reply("SMALL 0.9")

    run = Run([slow], policy={**POLICY, "max_concurrent_analyzer_calls": 1}, bindings=TWO)
    first: list[Any] = []
    worker = threading.Thread(
        target=lambda: first.append(run.gate.completion(model=ALIAS, messages=MSGS))
    )
    worker.daemon = True
    worker.start()
    assert inside.wait(5)
    # The first call is still inside the Analyzer, so this one finds the cap full and moves on.
    try:
        second = run.gate.completion(
            model=ALIAS, messages=[{"role": "user", "content": "a different question"}]
        )
        assert second.metadata is not None and second.metadata.tier == "MEDIUM"
        assert len(run.analyzed) == 1 and _reasons(run) == ["over_capacity"]
    finally:
        let_go.set()
    worker.join(5)
    assert first and first[0].metadata.tier == "SMALL" and _cap(run).in_flight == 0
