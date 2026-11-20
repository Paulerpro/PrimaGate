import ast
import importlib
import json
import re
import sys
import tomllib
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx2
import pytest

from primagate.schema.types import (
    Attempt,
    ChatCompletionChunk,
    ChunkChoice,
    Cost,
    Delta,
    DispatchRecord,
    FunctionDelta,
    Metadata,
    ToolCallDelta,
)

ROOT = Path(__file__).parents[1]
EVALS = ROOT / "evals"


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    # `evals/` is a folder of scripts, not an installed package, so it is put on the path by hand.
    sys.path.insert(0, str(EVALS))
    try:
        return importlib.import_module("harness")
    finally:
        sys.path.remove(str(EVALS))


@pytest.fixture(scope="module")
def workload(harness: ModuleType) -> ModuleType:
    return importlib.import_module("workload")


def _good(task: Any) -> dict[str, Any]:
    """The reply fields that pass `task`."""
    if task.type == "tool":
        return {"reply": None, "tool_call": (task.tool, json.dumps(task.args))}
    return {"reply": task.answer, "tool_call": None}


def _arm(
    harness: ModuleType,
    workload: ModuleType,
    cost: Callable[[Any], float | None] = lambda t: 0.01,
    passes: Callable[[Any], bool] = lambda t: True,
    arm: str = "routed",
    tasks: Any = None,
    **each: Any,
) -> list[Any]:
    rows = []
    for task in tasks if tasks is not None else workload.TASKS:
        fields = _good(task) if passes(task) else {"reply": "I am not sure.", "tool_call": None}
        rows.append(
            harness.CallResult(
                task=task,
                arm=arm,
                worker_cost=cost(task),
                latency=0.2,
                first_latency=0.1,
                **{**fields, **each},
            )
        )
    return rows


def _verdict(harness: ModuleType, routed: list[Any], baseline: list[Any], offline: bool) -> str:
    return str(harness.verdict(routed, baseline, offline))


# --- the workload ---


def test_the_workload_is_a_hundred_tasks_in_the_promised_mix(workload: ModuleType) -> None:
    tasks = workload.TASKS
    assert len(tasks) == 100 and len({t.id for t in tasks}) == 100
    assert Counter(t.difficulty for t in tasks) == {"easy": 40, "medium": 35, "hard": 25}
    assert Counter(t.type for t in tasks) == {"oneshot": 60, "tool": 20, "multiturn": 20}


def test_every_difficulty_is_spread_over_every_type(workload: ModuleType) -> None:
    pairs = {(t.type, t.difficulty) for t in workload.TASKS}
    assert len(pairs) == 9


def test_every_text_task_passes_its_own_answer_and_fails_a_wrong_one(workload: ModuleType) -> None:
    for task in (t for t in workload.TASKS if t.type != "tool"):
        assert task.check(task.answer) and task.check(f"The answer is {task.answer}.")
        assert not task.check("I am not sure.") and not task.check(None) and not task.check("")


def test_a_number_check_reads_the_last_number_and_a_word_check_needs_the_whole_word(
    workload: ModuleType,
) -> None:
    number = next(t for t in workload.TASKS if t.kind == "number" and t.answer == "43")
    assert number.check("17 + 26 = 43") and not number.check("43 is wrong, it is 44")
    assert number.check("1,043") is False
    word = next(t for t in workload.TASKS if t.answer == "paris")
    assert word.check("Paris") and not word.check("Parisian")


def test_a_word_answer_must_end_a_short_unnegated_reply(workload: ModuleType) -> None:
    cold = next(t for t in workload.TASKS if t.answer == "cold")
    assert cold.check("Cold.") and cold.check("The opposite is cold") and not cold.check("not cold")
    letter = next(t for t in workload.TASKS if t.answer == "b")
    assert letter.check("b") and not letter.check("a, b, c")
    assert not cold.check("cold is what the word hot is not, " + "very " * 6 + "cold")


def test_a_task_with_an_unknown_kind_type_or_difficulty_is_refused(workload: ModuleType) -> None:
    with pytest.raises(ValueError):
        workload.Task("x", "easy", "oneshot", "p", "a", "words")
    with pytest.raises(ValueError):
        workload.Task("x", "easy", "essay", "p", "a", "word")
    with pytest.raises(ValueError):
        workload.Task("x", "trivial", "oneshot", "p", "a", "word")


def test_a_task_shape_must_match_its_type(workload: ModuleType) -> None:
    with pytest.raises(ValueError):
        workload.Task("x", "easy", "multiturn", "p", "a", "word")
    with pytest.raises(ValueError):
        workload.Task("x", "easy", "oneshot", "p", "a", "word", history=(("q", "a"),))
    with pytest.raises(ValueError):
        workload.Task("x", "easy", "tool", "p", kind="tool")
    with pytest.raises(ValueError):
        workload.Task("x", "easy", "tool", "p", "a", "word", tools=("get_weather",), tool="t")


def _tool_task(workload: ModuleType) -> Any:
    return next(t for t in workload.TASKS if t.type == "tool" and t.tool == "set_timer")


def test_a_tool_task_passes_only_on_the_right_tool_with_the_expected_arguments(
    workload: ModuleType,
) -> None:
    task = _tool_task(workload)
    name, args = task.tool, dict(task.args)
    key, value = next(iter(args.items()))
    assert task.check(None, (name, json.dumps(args)))
    assert task.check(None, (name, json.dumps({**args, "extra": 1})))
    assert not task.check(None, ("send_email", json.dumps(args)))
    assert not task.check(None, (name, "{not json"))
    assert not task.check(None, (name, "[]"))
    assert not task.check(None, (name, json.dumps({})))
    assert not task.check(None, (name, json.dumps({**args, key: value + 1})))
    assert not task.check(None, None) and not task.check("set_timer", None)
    assert task.check(None, (name, json.dumps({**args, key: float(value)})))
    assert not task.check(None, (name, json.dumps({**args, key: str(value)})))


def test_a_tool_string_argument_is_compared_without_case_or_padding(workload: ModuleType) -> None:
    task = next(t for t in workload.TASKS if t.type == "tool" and t.tool == "get_weather")
    city = task.args["city"]
    assert task.check(None, ("get_weather", json.dumps({"city": f" {city.upper()} "})))
    assert not task.check(None, ("get_weather", json.dumps({"city": city + "x"})))


def test_every_tool_task_offers_its_own_tool_and_asks_for_known_arguments(
    workload: ModuleType,
) -> None:
    for task in (t for t in workload.TASKS if t.type == "tool"):
        specs = task.tool_specs()
        names = [s["function"]["name"] for s in specs]
        assert task.tool in names and len(names) == len(set(names))
        spec = next(s for s in specs if s["function"]["name"] == task.tool)
        assert set(task.args) <= set(spec["function"]["parameters"]["properties"])
        assert set(spec["function"]["parameters"]["required"]) <= set(task.args)


def test_a_multiturn_task_has_earlier_exchanges_and_an_answer_that_needs_them(
    workload: ModuleType,
) -> None:
    multi = [t for t in workload.TASKS if t.type == "multiturn"]
    assert all(t.history for t in multi)
    for task in multi:
        words = re.findall(r"[a-z0-9.]+", task.prompt.lower())
        assert task.answer not in words, task.id
        messages = task.messages()
        assert len(messages) == 2 * len(task.history) + 1
        assert [m["role"] for m in messages] == ["user", "assistant"] * len(task.history) + ["user"]
        assert messages[-1]["content"] == task.prompt
    assert any(len(t.history) > 3 for t in multi)
    assert any(1 <= len(t.history) <= 3 for t in multi)


def test_a_one_shot_task_is_a_single_user_message_and_offers_no_tools(workload: ModuleType) -> None:
    task = next(t for t in workload.TASKS if t.type == "oneshot")
    assert task.messages() == [{"role": "user", "content": task.prompt}]
    assert task.tool_specs() is None


# --- results, metrics and the verdict ---


@pytest.mark.parametrize(
    ("routed_cost", "routed_passes", "expected"),
    [
        (0.6, 99, "SM-4 MET"),
        (0.8, 100, "SM-4 NOT MET"),
        (0.6, 95, "SM-4 NOT MET"),
        (0.7, 98, "SM-4 MET"),
        (0.71, 98, "SM-4 NOT MET"),
        (0.7, 97, "SM-4 NOT MET"),
    ],
)
def test_the_verdict_follows_the_target_exactly(
    harness: ModuleType,
    workload: ModuleType,
    routed_cost: float,
    routed_passes: int,
    expected: str,
) -> None:
    order = {t.id: i for i, t in enumerate(workload.TASKS)}
    routed = _arm(
        harness, workload, lambda t: routed_cost / 100, lambda t: order[t.id] < routed_passes
    )
    baseline = _arm(harness, workload, arm="baseline")
    line = _verdict(harness, routed, baseline, False)
    assert line.startswith(expected + ":") and "lower" in line


def test_a_met_verdict_shows_both_numbers(harness: ModuleType, workload: ModuleType) -> None:
    routed = _arm(harness, workload, lambda t: 0.005)
    line = _verdict(harness, routed, _arm(harness, workload, arm="baseline"), False)
    assert line.startswith("SM-4 MET") and "50.0%" in line and "0.0 points" in line


def test_a_saving_of_twelve_percent_is_not_met_and_says_so(
    harness: ModuleType, workload: ModuleType
) -> None:
    routed = _arm(harness, workload, lambda t: 0.0088)
    line = _verdict(harness, routed, _arm(harness, workload, arm="baseline"), False)
    assert line.startswith("SM-4 NOT MET") and "12.0%" in line


def test_a_big_saving_with_a_five_point_gap_names_the_gap(
    harness: ModuleType, workload: ModuleType
) -> None:
    order = {t.id: i for i, t in enumerate(workload.TASKS)}
    routed = _arm(harness, workload, lambda t: 0.006, lambda t: order[t.id] < 95)
    line = _verdict(harness, routed, _arm(harness, workload, arm="baseline"), False)
    assert line.startswith("SM-4 NOT MET") and "40.0%" in line and "5.0 points" in line


def _erring(harness: ModuleType, rows: list[Any], first: int) -> list[Any]:
    """`rows` with the first `first` calls turned into raised calls."""
    return [
        harness.CallResult(task=r.task, arm=r.arm, error="OSError", latency=0.1) if i < first else r
        for i, r in enumerate(rows)
    ]


def test_an_arm_answering_under_ninety_percent_is_inconclusive_and_says_so(
    harness: ModuleType, workload: ModuleType
) -> None:
    baseline = _arm(harness, workload, arm="baseline")
    line = _verdict(harness, _erring(harness, _arm(harness, workload), 11), baseline, False)
    assert line.startswith("INCONCLUSIVE") and "routed arm answered 89 of 100" in line
    line = _verdict(harness, _arm(harness, workload), _erring(harness, baseline, 11), False)
    assert line.startswith("INCONCLUSIVE") and "fixed arm answered 89 of 100" in line
    ninety = _verdict(harness, _erring(harness, _arm(harness, workload), 10), baseline, False)
    assert ninety.startswith("SM-4")
    assert _verdict(harness, [], baseline, False).startswith("INCONCLUSIVE")


def test_an_unknown_or_zero_cost_is_inconclusive(harness: ModuleType, workload: ModuleType) -> None:
    baseline = _arm(harness, workload, arm="baseline")
    unknown = _arm(harness, workload, lambda t: None if t.id == workload.TASKS[3].id else 0.01)
    line = _verdict(harness, unknown, baseline, False)
    assert line.startswith("INCONCLUSIVE") and "cost" in line
    free = _arm(harness, workload, lambda t: 0.0, arm="baseline")
    assert _verdict(harness, _arm(harness, workload), free, False).startswith("INCONCLUSIVE")


def test_the_verdict_compares_only_the_tasks_both_arms_answered(
    harness: ModuleType, workload: ModuleType
) -> None:
    # The routed arm lost 5 tasks and the fixed arm another 5. Costs are 0.01 a task in the
    # fixed arm and 0.006 in the routed arm, so the saving is 40% on any common set, but it
    # would not be on the raw sums (90 against 90 tasks of different mix).
    routed = _erring(harness, _arm(harness, workload, lambda t: 0.006), 5)
    baseline = _arm(harness, workload, lambda t: 0.01, arm="baseline")
    baseline = [
        harness.CallResult(task=r.task, arm="baseline", error="OSError") if 95 <= i else r
        for i, r in enumerate(baseline)
    ]
    shared_routed, shared_baseline = harness.common(routed, baseline)
    assert len(shared_routed) == len(shared_baseline) == 90
    assert {r.task.id for r in shared_routed} == {r.task.id for r in shared_baseline}
    line = _verdict(harness, routed, baseline, False)
    assert line.startswith("SM-4 MET") and "40.0%" in line
    text = harness.render_report(harness.Config(), routed, baseline, False, datetime.now(UTC))
    assert "Compared set: 90 task(s) answered by both arms" in text
    assert "routed 5, fixed 5" in text and "lower bound" in text


def test_the_verdict_words_name_the_direction_plainly(
    harness: ModuleType, workload: ModuleType
) -> None:
    order = {t.id: i for i, t in enumerate(workload.TASKS)}
    baseline = _arm(harness, workload, arm="baseline", passes=lambda t: order[t.id] < 99)
    dearer = _arm(harness, workload, lambda t: 0.0105)
    line = _verdict(harness, dearer, baseline, False)
    assert "cost rose 5.0%" in line and "Usefulness 1.0 points higher" in line
    assert "-5.0" not in line and "+" not in line.split("(")[0]
    cheaper = _arm(harness, workload, lambda t: 0.0095, passes=lambda t: order[t.id] < 98)
    line = _verdict(harness, cheaper, baseline, False)
    assert "cost fell 5.0%" in line and "Usefulness 1.0 points lower" in line


def test_an_offline_run_can_never_say_met(harness: ModuleType, workload: ModuleType) -> None:
    routed = _arm(harness, workload, lambda t: 0.001)
    line = _verdict(harness, routed, _arm(harness, workload, arm="baseline"), True)
    assert line.startswith("INCONCLUSIVE") and "offline" in line and "MET" not in line


def test_a_raised_call_is_a_failed_check_and_the_run_goes_on(
    harness: ModuleType, workload: ModuleType
) -> None:
    task = workload.TASKS[0]
    good = harness.CallResult(task=task, arm="routed", reply=task.answer, latency=0.1)
    bad = harness.CallResult(task=task, arm="routed", error="OSError", latency=0.1)
    summary = harness.summarize([good, bad])
    assert (summary.calls, summary.completed, summary.passes, summary.errors) == (2, 1, 1, 1)
    assert summary.usefulness == 50.0 and not bad.passed


def test_a_cost_that_is_unknown_is_listed_and_left_out_of_the_sums(
    harness: ModuleType, workload: ModuleType
) -> None:
    rows = _arm(harness, workload, lambda t: None if t.id == workload.TASKS[0].id else 0.01)
    summary = harness.summarize(rows)
    assert summary.unknown_cost == (workload.TASKS[0].id,)
    assert summary.worker_cost == pytest.approx(0.99)


def test_the_analyzer_is_reported_apart_from_the_worker(
    harness: ModuleType, workload: ModuleType
) -> None:
    rows = _arm(
        harness,
        workload,
        tasks=workload.TASKS[:1],
        analyzer_cost=0.002,
        analyzer_latency=0.05,
        tier="SMALL",
    )
    summary = harness.summarize(rows)
    assert summary.analyzer_cost == pytest.approx(0.002)
    assert summary.worker_cost == pytest.approx(0.01)
    assert summary.total_cost == pytest.approx(0.012)
    assert summary.analyzer_latency_avg == pytest.approx(0.05) and summary.tiers == {"SMALL": 1}


def test_an_analyzer_with_no_price_cannot_be_read_as_free(
    harness: ModuleType, workload: ModuleType
) -> None:
    rows = _arm(harness, workload, tasks=workload.TASKS[:1], tier="SMALL", analyzer_cost=None)
    assert harness.summarize(rows).unknown_cost == (workload.TASKS[0].id,)
    degraded = _arm(
        harness,
        workload,
        tasks=workload.TASKS[:1],
        tier="MEDIUM",
        analyzer_cost=None,
        degraded_reason="preparing",
    )
    assert harness.summarize(degraded).unknown_cost == ()


def test_cost_per_successful_interaction_is_the_total_over_the_passes(
    harness: ModuleType, workload: ModuleType
) -> None:
    order = {t.id: i for i, t in enumerate(workload.TASKS)}
    rows = _arm(harness, workload, passes=lambda t: order[t.id] < 50, analyzer_cost=0.01)
    summary = harness.summarize(rows)
    assert summary.total_cost == pytest.approx(2.0) and summary.passes == 50
    assert summary.cost_per_pass == pytest.approx(0.04)
    none = harness.summarize(_arm(harness, workload, passes=lambda t: False))
    assert none.cost_per_pass is None


def test_first_response_and_completion_latency_are_kept_apart(
    harness: ModuleType, workload: ModuleType
) -> None:
    rows = [
        harness.CallResult(
            task=task, arm="routed", reply=task.answer, latency=n / 10, first_latency=n / 100
        )
        for n, task in enumerate(workload.TASKS[:20], start=1)
    ]
    summary = harness.summarize(rows)
    assert summary.first_p95 == pytest.approx(0.19) and summary.latency_p95 == pytest.approx(1.9)
    assert summary.first_avg == pytest.approx(0.105) and summary.latency_avg == pytest.approx(1.05)


def test_the_fallback_rate_counts_calls_with_more_than_one_attempt(
    harness: ModuleType, workload: ModuleType
) -> None:
    rows = [
        harness.CallResult(task=t, arm="routed", reply=t.answer, attempts=2 if i < 5 else 1)
        for i, t in enumerate(workload.TASKS[:20])
    ]
    summary = harness.summarize(rows)
    assert summary.fallbacks == 5 and summary.fallback_rate == pytest.approx(0.25)


def _multi(workload: ModuleType, exchanges: int) -> Any:
    history = tuple((f"q{n}", f"a{n}") for n in range(exchanges))
    return workload.Task("m", "easy", "multiturn", "p", "a", "word", history=history)


def test_a_task_with_more_than_three_earlier_exchanges_counts_as_omitted(
    harness: ModuleType, workload: ModuleType
) -> None:
    def result(exchanges: int, **extra: Any) -> Any:
        return harness.CallResult(task=_multi(workload, exchanges), arm="routed", **extra)

    assert result(5).omitted and result(4).omitted
    assert not result(3).omitted and not result(1).omitted
    plain = workload.TASKS[0]
    assert harness.CallResult(task=plain, arm="routed", degraded_reason="over_budget").omitted
    assert not harness.CallResult(task=plain, arm="routed", degraded_reason="preparing").omitted
    summary = harness.summarize([result(5), result(2), result(2, degraded_reason="over_budget")])
    assert summary.omissions == 2 and summary.degraded == 1
    assert (summary.long_history, summary.over_budget) == (1, 1)


def test_the_fixed_arm_has_no_context_omissions_and_the_report_shows_a_dash(
    harness: ModuleType, workload: ModuleType
) -> None:
    long = _multi(workload, 5)
    row = harness.CallResult(task=long, arm="baseline", degraded_reason="over_budget")
    assert not row.omitted and not row.long_history and not row.over_budget
    assert harness.summarize([row]).omissions == 0
    routed = _arm(harness, workload)
    text = harness.render_report(
        harness.Config(), routed, _arm(harness, workload, arm="baseline"), False, datetime.now(UTC)
    )
    line = next(x for x in text.splitlines() if x.startswith("| Context omissions (proxy)"))
    assert line.endswith("| - |") and "7 with more than 3 earlier exchanges, 0 over budget" in line


def test_a_degraded_call_adds_no_analyzer_latency(
    harness: ModuleType, workload: ModuleType
) -> None:
    task = workload.TASKS[0]
    real = harness.CallResult(task=task, arm="routed", analyzer_latency=0.1, tier="SMALL")
    gone = harness.CallResult(
        task=task, arm="routed", analyzer_latency=9.0, degraded_reason="preparing"
    )
    assert harness.summarize([real, gone]).analyzer_latency_avg == pytest.approx(0.1)


def test_the_interim_binding_fallbacks_are_counted_and_flagged_in_the_verdict(
    harness: ModuleType, workload: ModuleType
) -> None:
    routed = _arm(
        harness,
        workload,
        lambda t: 0.005,
        tier="MEDIUM",
        analyzer_cost=0.0,
        degraded_reason="preparing",
    )
    summary = harness.summarize(routed)
    assert summary.degraded == 100
    line = _verdict(harness, routed, _arm(harness, workload, arm="baseline"), False)
    assert "100 routed call(s) fell back to the Interim Binding" in line


def test_usefulness_is_split_by_difficulty_and_by_type(
    harness: ModuleType, workload: ModuleType
) -> None:
    rows = _arm(harness, workload, passes=lambda t: t.type != "tool")
    summary = harness.summarize(rows)
    assert summary.by_type["tool"].passed == 0 and summary.by_type["tool"].asked == 20
    assert summary.by_type["oneshot"].passed == 60 and summary.by_type["multiturn"].asked == 20
    assert {k: v.asked for k, v in summary.by_difficulty.items()} == {
        "easy": 40,
        "medium": 35,
        "hard": 25,
    }


def test_the_95th_percentile_is_the_nearest_rank(harness: ModuleType) -> None:
    assert harness.percentile([n / 100 for n in range(1, 21)], 0.95) == pytest.approx(0.19)
    assert harness.percentile([], 0.95) == 0.0


# --- the per-type table, calibration ---


def _by_type(
    harness: ModuleType, workload: ModuleType, routed_cost: dict[str, float], gap: dict[str, int]
) -> tuple[list[Any], list[Any]]:
    seen: Counter[str] = Counter()

    def passes(task: Any) -> bool:
        seen[task.type] += 1
        return seen[task.type] > gap.get(task.type, 0)

    routed = _arm(harness, workload, lambda t: routed_cost[t.type] / 100, passes)
    return routed, _arm(harness, workload, arm="baseline")


def test_each_type_shows_its_saving_and_gap_and_where_the_target_holds(
    harness: ModuleType, workload: ModuleType
) -> None:
    routed, baseline = _by_type(
        harness, workload, {"oneshot": 0.8, "tool": 1.0, "multiturn": 0.4}, {"tool": 4}
    )
    rows = {r.type: r for r in harness.type_rows(routed, baseline)}
    assert rows["multiturn"].saving == pytest.approx(60.0) and rows["multiturn"].holds
    assert rows["oneshot"].saving == pytest.approx(20.0) and not rows["oneshot"].holds
    assert rows["tool"].saving == pytest.approx(0.0) and rows["tool"].gap == pytest.approx(20.0)
    assert not rows["tool"].holds


def test_a_type_with_an_unknown_cost_has_no_saving_and_does_not_hold(
    harness: ModuleType, workload: ModuleType
) -> None:
    routed = _arm(harness, workload, lambda t: None if t.type == "tool" else 0.001)
    rows = {r.type: r for r in harness.type_rows(routed, _arm(harness, workload, arm="baseline"))}
    assert rows["tool"].saving is None and not rows["tool"].holds and rows["oneshot"].holds


def test_a_not_met_report_names_the_saving_and_only_the_types_where_it_holds(
    harness: ModuleType, workload: ModuleType
) -> None:
    routed, baseline = _by_type(
        harness, workload, {"oneshot": 0.8, "tool": 1.0, "multiturn": 0.4}, {}
    )
    text = harness.render_report(harness.Config(), routed, baseline, False, datetime.now(UTC))
    assert "SM-4 NOT MET" in text and "24.0%" in text
    assert "SM-4 by workload type" in text
    assert "holds for: multiturn." in text and "No claim is made beyond" in text


def test_a_report_with_no_type_where_it_holds_says_so(
    harness: ModuleType, workload: ModuleType
) -> None:
    routed, baseline = _by_type(
        harness, workload, {"oneshot": 0.95, "tool": 0.95, "multiturn": 0.95}, {}
    )
    text = harness.render_report(harness.Config(), routed, baseline, False, datetime.now(UTC))
    assert "SM-4 NOT MET" in text and "holds for no workload type" in text


def _confident(
    harness: ModuleType, workload: ModuleType, plan: list[tuple[float, bool]], **extra: Any
) -> list[Any]:
    rows = []
    for task, (confidence, ok) in zip(workload.TASKS, plan, strict=False):
        fields = _good(task) if ok else {"reply": "I am not sure.", "tool_call": None}
        rows.append(
            harness.CallResult(
                task=task, arm="routed", confidence=confidence, tier="SMALL", **{**fields, **extra}
            )
        )
    return rows


def _plan(rates: dict[float, int], each: int = 10) -> list[tuple[float, bool]]:
    """`each` calls at every confidence, of which `rates[confidence]` pass."""
    return [(c, n < rates[c]) for c in rates for n in range(each)]


def test_calls_are_bucketed_by_confidence_with_the_edges_in_the_upper_bucket(
    harness: ModuleType, workload: ModuleType
) -> None:
    plan = [(c, True) for c in (0.1, 0.5, 0.69, 0.7, 0.89, 0.9, 1.0)]
    buckets = harness.calibration(_confident(harness, workload, plan))
    assert [b.calls for b in buckets] == [1, 2, 2, 2]
    assert [b.label for b in buckets] == ["below 0.5", "0.5 to 0.7", "0.7 to 0.9", "0.9 and above"]


def test_higher_confidence_with_a_higher_pass_rate_is_stated_plainly(
    harness: ModuleType, workload: ModuleType
) -> None:
    rows = _confident(harness, workload, _plan({0.3: 3, 0.6: 5, 0.8: 8, 0.95: 10}))
    threshold = harness.Config().threshold
    text = harness.calibration_statement(harness.calibration(rows, threshold))
    assert text.startswith("Higher confidence went with a higher pass rate")
    report = harness.render_report(
        harness.Config(), rows, _arm(harness, workload, arm="baseline"), False, datetime.now(UTC)
    )
    assert "## Calibration" in report and text in report


def test_a_confidence_that_does_not_track_the_pass_rate_is_not_called_calibrated(
    harness: ModuleType, workload: ModuleType
) -> None:
    rows = _confident(harness, workload, _plan({0.3: 9, 0.6: 5, 0.8: 8, 0.95: 4}))
    text = harness.calibration_statement(harness.calibration(rows))
    assert text.startswith("Higher confidence did not go with a higher pass rate")
    flat = _confident(harness, workload, _plan({0.3: 5, 0.95: 5}))
    assert "did not" in harness.calibration_statement(harness.calibration(flat))


def test_too_little_confidence_data_makes_no_statement(
    harness: ModuleType, workload: ModuleType
) -> None:
    few = harness.calibration_statement(harness.calibration(_confident(harness, workload, [])))
    assert few.startswith("Too few")
    one_bucket = _confident(harness, workload, _plan({0.95: 10}, each=30))
    assert harness.calibration_statement(harness.calibration(one_bucket)).startswith("Too few")


def test_calibration_leaves_out_calls_with_no_confidence_or_no_answer(
    harness: ModuleType, workload: ModuleType
) -> None:
    rows = _confident(harness, workload, [(0.95, True)] * 3)
    task = workload.TASKS[5]
    rows.append(harness.CallResult(task=task, arm="routed", reply=task.answer))
    rows.append(harness.CallResult(task=task, arm="routed", error="OSError", confidence=0.95))
    assert sum(b.calls for b in harness.calibration(rows)) == 3


def test_calibration_leaves_out_degraded_and_low_confidence_calls(
    harness: ModuleType, workload: ModuleType
) -> None:
    rows = _confident(harness, workload, [(0.95, True), (0.6, True), (0.8, True)])
    rows.append(
        harness.CallResult(
            task=workload.TASKS[9], arm="routed", confidence=0.95, degraded_reason="preparing"
        )
    )
    assert sum(b.calls for b in harness.calibration(rows)) == 3
    assert sum(b.calls for b in harness.calibration(rows, 0.7)) == 2


def test_a_bucket_of_a_few_calls_is_shown_but_not_used_for_the_trend(
    harness: ModuleType, workload: ModuleType
) -> None:
    # The thin bucket (3 calls, all failing) would break the rise if it counted.
    plan = [(0.6, False)] * 3 + _plan({0.8: 5, 0.95: 9})
    buckets = harness.calibration(_confident(harness, workload, plan))
    assert buckets[1].thin and buckets[1].calls == 3
    assert harness.calibration_statement(buckets).startswith("Higher confidence went with")
    text = harness.render_report(
        harness.Config(),
        _confident(harness, workload, [(0.75, False)] * 3 + _plan({0.95: 9})),
        _arm(harness, workload, arm="baseline"),
        False,
        datetime.now(UTC),
    )
    assert "(too few)" in text and "not stratified by difficulty" in text


def test_a_tie_between_buckets_is_not_a_rise(harness: ModuleType, workload: ModuleType) -> None:
    rows = _confident(harness, workload, _plan({0.6: 8, 0.8: 8, 0.95: 10}))
    text = harness.calibration_statement(harness.calibration(rows))
    assert text.startswith("Higher confidence did not go with")


# --- one call ---


def _chunk(
    delta: Delta, finish: str | None = None, meta: Metadata | None = None
) -> ChatCompletionChunk:
    return ChatCompletionChunk("id", 0, "m", [ChunkChoice(0, delta, finish)], None, metadata=meta)


def _meta(attempts: int = 1, **extra: Any) -> Metadata:
    chain = [Attempt(f"m{n}", [DispatchRecord("openai", 0)]) for n in range(attempts)]
    chain.append(Attempt("skipped", [], skipped="no tools"))
    return Metadata(attempt_chain=chain, cost=Cost(0.001, 0.002, 0.003), **extra)


class FakeGate:
    def __init__(self, chunks: list[Any], fail_at_call: bool = False) -> None:
        self.chunks, self.params, self.fail_at_call = chunks, {}, fail_at_call

    def completion(self, **params: Any) -> Any:
        self.params = params
        if self.fail_at_call:
            raise ConnectionError("sk-secret and the prompt text")
        return self._iterate()

    def _iterate(self) -> Any:
        for chunk in self.chunks:
            if isinstance(chunk, Exception):
                raise chunk
            yield chunk


def test_a_text_reply_is_read_from_a_stream_with_the_final_chunks_metadata(
    harness: ModuleType, workload: ModuleType
) -> None:
    task = workload.TASKS[0]
    gate = FakeGate(
        [
            _chunk(Delta(role="assistant", content="4")),
            _chunk(Delta(content="3")),
            _chunk(Delta(), "stop", _meta(attempts=2, tier="SMALL")),
        ]
    )
    result = harness.ask(gate, harness.Probe(), "primagate/auto", task, "routed")
    assert result.reply == "43" and result.tool_call is None and result.passed
    assert result.error is None and result.attempts == 2 and result.fallback
    assert result.worker_cost == pytest.approx(0.003) and result.tier == "SMALL"
    assert result.first_latency is not None and 0 <= result.first_latency <= result.latency
    params = gate.params
    assert params["stream"] is True and params["cache"] is False and params["temperature"] == 0
    assert params["model"] == "primagate/auto" and params["max_tokens"] == 64
    assert params["messages"] == task.messages() and "tools" not in params


def test_a_tool_call_is_assembled_from_its_deltas_and_the_tools_are_offered(
    harness: ModuleType, workload: ModuleType
) -> None:
    task = next(t for t in workload.TASKS if t.type == "tool" and t.tool == "get_weather")
    arguments = json.dumps(task.args)
    half = len(arguments) // 2

    def delta(**function: Any) -> Delta:
        return Delta(tool_calls=[ToolCallDelta(0, function=FunctionDelta(**function))])

    gate = FakeGate(
        [
            _chunk(delta(name=task.tool, arguments="")),
            _chunk(delta(arguments=arguments[:half])),
            _chunk(delta(arguments=arguments[half:])),
            _chunk(Delta(), "tool_calls", _meta()),
        ]
    )
    result = harness.ask(gate, harness.Probe(), "gpt-4o", task, "baseline")
    assert result.tool_call == (task.tool, arguments) and result.passed
    assert gate.params["tools"] == task.tool_specs() and not result.fallback


def test_a_repeated_tool_name_is_not_joined_and_a_cut_reply_is_reported_apart(
    harness: ModuleType, workload: ModuleType
) -> None:
    task = next(t for t in workload.TASKS if t.type == "tool" and t.tool == "set_timer")
    arguments = json.dumps(task.args)

    def delta(**function: Any) -> Delta:
        return Delta(tool_calls=[ToolCallDelta(0, function=FunctionDelta(**function))])

    gate = FakeGate(
        [
            _chunk(delta(name=task.tool, arguments="")),
            _chunk(delta(name=task.tool, arguments=arguments)),
            _chunk(Delta(), "tool_calls", _meta()),
        ]
    )
    result = harness.ask(gate, harness.Probe(), "gpt-4o", task, "baseline")
    assert result.tool_call == (task.tool, arguments) and result.passed
    assert result.finish_reason == "tool_calls" and not result.cut_off
    cut = FakeGate(
        [_chunk(delta(name=task.tool, arguments=arguments[:5])), _chunk(Delta(), "length", _meta())]
    )
    result = harness.ask(cut, harness.Probe(), "gpt-4o", task, "baseline")
    assert not result.passed and result.cut_off and harness.summarize([result]).cut_off == 1


def test_a_huge_integer_argument_is_a_mismatch_and_never_raises(
    harness: ModuleType, workload: ModuleType
) -> None:
    task = _tool_task(workload)
    huge = '{"minutes": ' + "9" * 400 + "}"
    assert task.check(None, (task.tool, huge)) is False
    row = harness.CallResult(task=task, arm="routed", tool_call=(task.tool, huge))
    assert row.passed is False


def test_a_bad_timestamp_in_a_decision_event_never_raises_in_the_probe(
    harness: ModuleType,
) -> None:
    from primagate import RoutingDecision

    fields = dict.fromkeys(
        ("profile_id", "input_fingerprint", "prep_prompt_version", "analyzer_prompt_version"),
        "unknown",
    )
    probe = harness.Probe()
    probe(RoutingDecision("r", **fields, binding_id="b", timestamp="not a time", confidence=0.8))
    assert probe.decided_at is None and probe.confidence == 0.8


def test_a_call_that_raises_is_recorded_by_class_name_and_fails(
    harness: ModuleType, workload: ModuleType
) -> None:
    task = workload.TASKS[0]
    result = harness.ask(
        FakeGate([], fail_at_call=True), harness.Probe(), "gpt-4o", task, "baseline"
    )
    assert result.error == "ConnectionError" and not result.passed and result.reply is None
    midway = FakeGate([_chunk(Delta(content=task.answer)), TimeoutError("sk-secret")])
    result = harness.ask(midway, harness.Probe(), "gpt-4o", task, "baseline")
    assert result.error == "TimeoutError" and not result.passed and result.reply is None
    assert "secret" not in repr(result)


def test_the_probe_notes_the_decision_time_the_confidence_and_a_degrade(
    harness: ModuleType,
) -> None:
    from primagate import RoutingDecision, RoutingDegraded

    fields = {
        "profile_id": "unknown",
        "input_fingerprint": "unknown",
        "prep_prompt_version": "unknown",
        "analyzer_prompt_version": "unknown",
        "binding_id": "SMALL:openai/m",
    }
    probe = harness.Probe()
    stamp = "2026-11-17T10:00:00.250+00:00"
    probe(RoutingDecision("r", **fields, timestamp=stamp, confidence=0.82))
    probe(RoutingDegraded("r", **fields, reason="over_budget"))
    assert probe.degraded_reason == "over_budget" and probe.confidence == 0.82
    assert probe.decided_at == datetime(2026, 11, 17, 10, 0, 0, 250000, UTC)
    probe.reset()
    assert probe.degraded_reason is None and probe.decided_at is None and probe.confidence is None


def test_the_added_latency_is_measured_from_the_start_and_never_negative(
    harness: ModuleType,
) -> None:
    start = datetime(2026, 11, 17, 10, 0, 0, tzinfo=UTC)
    assert harness._added(datetime(2026, 11, 17, 10, 0, 2, tzinfo=UTC), start) == 2.0
    assert harness._added(datetime(2026, 11, 17, 9, 59, 59, tzinfo=UTC), start) == 0.0
    assert harness._added(None, start) is None
    assert harness._added(datetime(2026, 11, 17, 10, 0, 2), start) is None


# --- the whole run, offline ---


@pytest.fixture(scope="module")
def offline(harness: ModuleType) -> tuple[list[Any], list[Any]]:
    transport = httpx2.MockTransport(harness.stand_in)
    return harness.run(harness.Config(), "offline-key", transport)  # type: ignore[no-any-return]


def test_the_offline_run_asks_every_task_in_both_arms_and_passes_the_baseline(
    workload: ModuleType, offline: tuple[list[Any], list[Any]]
) -> None:
    routed, fixed = offline
    assert len(routed) == len(fixed) == 100
    assert [r.task.id for r in routed] == [t.id for t in workload.TASKS]
    assert {r.arm for r in routed} == {"routed"} and {r.arm for r in fixed} == {"baseline"}
    assert all(r.error is None for r in routed + fixed)
    # The premium stand-in knows everything, so a miss would be a harness fault, tools included.
    assert all(r.passed for r in fixed)
    assert all(r.first_latency is not None for r in routed + fixed)


def test_the_offline_run_shows_routing_cheaper_and_gives_calibration_data(
    harness: ModuleType, offline: tuple[list[Any], list[Any]]
) -> None:
    routed, fixed = (harness.summarize(rows) for rows in offline)
    assert routed.analyzer_cost > 0 and routed.analyzer_latency_avg is not None
    assert routed.total_cost < fixed.total_cost and not routed.unknown_cost
    assert sum(routed.tiers.values()) == 100
    assert set(routed.tiers) == {"SMALL", "MEDIUM", "POWERFUL"}
    buckets = harness.calibration(offline[0])
    assert sum(b.calls for b in buckets) == 100 and sum(1 for b in buckets if b.calls) >= 3


def test_the_two_arms_are_asked_in_turn(harness: ModuleType, monkeypatch: Any) -> None:
    order: list[str] = []
    real = harness.ask

    def spy(gate: Any, probe: Any, model: str, task: Any, arm: str) -> Any:
        order.append(model)
        return real(gate, probe, model, task, arm)

    monkeypatch.setattr(harness, "ask", spy)
    harness.run(harness.Config(), "offline-key", httpx2.MockTransport(harness.stand_in))
    assert len(order) == 200
    # Which arm goes first alternates task by task.
    assert order[:4] == ["primagate/auto", "gpt-4o", "gpt-4o", "primagate/auto"]
    assert [order[n] for n in range(0, 200, 4)] == ["primagate/auto"] * 50


def test_the_offline_run_works_with_no_key_and_writes_a_report_and_a_results_file(
    harness: ModuleType,
    workload: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert harness.main(["--offline", "--out-dir", str(tmp_path)]) == 0
    shown = capsys.readouterr().out
    assert harness.OFFLINE_BANNER in shown and "SM-4 MET" not in shown
    assert "INCONCLUSIVE" in shown
    for metric in (
        "Usefulness",
        "Total cost",
        "Cost per successful interaction",
        "Analyzer overhead",
        "First response, average",
        "First response, 95th percentile",
        "Completion, average",
        "Completion, 95th percentile",
        "Fallback rate",
        "Context omissions",
        "Interim Binding",
        "Calls per Tier",
        "Usefulness by difficulty",
        "Usefulness by type",
        "SM-4 by workload type",
        "## Calibration",
    ):
        assert metric in shown, metric
    (report,) = tmp_path.glob("offline-harness-*.md")
    assert harness.OFFLINE_BANNER in report.read_text(encoding="utf-8")
    (results,) = tmp_path.glob("offline-harness-*.json")
    data = json.loads(results.read_text(encoding="utf-8"))
    assert data["offline"] is True and len(data["calls"]) == 200
    wanted = {
        "task_id",
        "type",
        "difficulty",
        "arm",
        "passed",
        "worker_cost",
        "analyzer_cost",
        "first_latency",
        "latency",
        "analyzer_latency",
        "tier",
        "confidence",
        "error",
    }
    assert all(wanted <= set(call) for call in data["calls"])
    assert {c["arm"] for c in data["calls"]} == {"routed", "baseline"}
    assert {c["task_id"] for c in data["calls"]} == {t.id for t in workload.TASKS}
    routed = [c for c in data["calls"] if c["arm"] == "routed"]
    assert all(c["confidence"] is not None for c in routed)


def test_neither_the_report_nor_the_results_file_holds_a_key_a_prompt_or_a_reply(
    harness: ModuleType, workload: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert harness.main(["--offline", "--out-dir", str(tmp_path)]) == 0
    text = capsys.readouterr().out + "".join(
        p.read_text(encoding="utf-8") for p in tmp_path.iterdir()
    )
    assert "offline-key" not in text and "I am not sure" not in text
    for task in workload.TASKS:
        assert task.prompt not in text
        for question, reply in task.history:
            # Replies like "Noted." are too short to tell from report words, so only the long ones.
            assert all(part not in text for part in (question, reply) if len(part) > 12)
        if task.type == "tool":
            assert json.dumps(task.args) not in text


def test_a_live_run_without_a_key_stops_before_any_call(
    harness: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: Any
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a call was made")

    monkeypatch.setattr(harness, "run", never)
    assert harness.main(["--out-dir", str(tmp_path)]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err and list(tmp_path.iterdir()) == []


def test_the_key_never_appears_in_the_output_or_the_files(
    harness: ModuleType,
    workload: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: Any,
) -> None:
    seen: list[str] = []

    def fake_run(config: Any, key: str, transport: Any) -> Any:
        seen.append(key)
        return _arm(harness, workload, lambda t: 0.005), _arm(harness, workload, arm="baseline")

    monkeypatch.setattr(harness, "run", fake_run)
    secret = "sk-live-secret-123"
    assert harness.main(["--api-key", secret, "--out-dir", str(tmp_path)]) == 0
    assert seen == [secret]
    (report,) = tmp_path.glob("harness-*.md")
    assert "SM-4 MET" in report.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("offline-*"))
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err
    assert all(secret not in p.read_text(encoding="utf-8") for p in tmp_path.iterdir())


def test_a_live_report_carries_a_real_verdict_and_both_arms(
    harness: ModuleType, workload: ModuleType
) -> None:
    routed = _arm(harness, workload, lambda t: 0.005, analyzer_cost=0.0, tier="SMALL")
    baseline = _arm(harness, workload, arm="baseline")
    text = harness.render_report(harness.Config(), routed, baseline, False, datetime.now(UTC))
    assert "SM-4 MET" in text and harness.OFFLINE_BANNER not in text
    assert "Tier Routing" in text and "Fixed model" in text and "Analyzer overhead" in text


def test_a_report_shows_the_share_per_tier_and_the_failures(
    harness: ModuleType, workload: ModuleType
) -> None:
    routed = _arm(harness, workload, passes=lambda t: t.id != workload.TASKS[-1].id, tier="SMALL")
    baseline = _arm(harness, workload, arm="baseline")
    text = harness.render_report(harness.Config(), routed, baseline, False, datetime.now(UTC))
    assert (
        "SMALL 100 (100%)" in text and f"Failed in the routed arm: {workload.TASKS[-1].id}" in text
    )
    assert "interim binding MEDIUM" in text and "(indicative)" in text


def test_two_runs_on_one_day_do_not_overwrite_each_other_and_the_name_has_the_models(
    harness: ModuleType, tmp_path: Path, monkeypatch: Any
) -> None:
    class Moment(datetime):
        ticks = 0

        @classmethod
        def now(cls, tz: Any = None) -> Any:
            # Each look at the clock is a second later, so two runs can never share a name.
            cls.ticks += 1
            return datetime(2026, 11, 17, 10, 0, 0, tzinfo=UTC) + timedelta(seconds=cls.ticks)

    monkeypatch.setattr(harness, "datetime", Moment)
    for _ in range(2):
        assert harness.main(["--offline", "--out-dir", str(tmp_path)]) == 0
    names = sorted(p.name for p in tmp_path.glob("offline-harness-*.md"))
    assert len(names) == 2 and all("gpt-4o-vs-gpt-4o-mini-gpt-4.1-mini-gpt-4o" in n for n in names)


def test_a_report_that_cannot_be_written_is_still_printed(
    harness: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert harness.main(["--offline", "--out-dir", str(blocker / "sub")]) == 1
    captured = capsys.readouterr()
    assert harness.OFFLINE_BANNER in captured.out and "Could not write" in captured.err


def test_a_run_where_one_arm_answered_nothing_exits_non_zero(
    harness: ModuleType,
    workload: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: Any,
) -> None:
    fixed = [
        harness.CallResult(task=t, arm="baseline", error="OSError", latency=0.1)
        for t in workload.TASKS
    ]
    good = _arm(harness, workload, lambda t: 0.005)
    monkeypatch.setattr(harness, "run", lambda *args, **kwargs: (good, fixed))
    assert harness.main(["--api-key", "sk-x", "--out-dir", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "No call was answered in the fixed arm" in err
    assert "SM-4 MET" not in "".join(p.read_text(encoding="utf-8") for p in tmp_path.iterdir())


def test_a_run_where_nothing_was_answered_exits_non_zero(
    harness: ModuleType,
    workload: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: Any,
) -> None:
    def broken(arm: str) -> list[Any]:
        return [
            harness.CallResult(task=t, arm=arm, error="OSError", latency=0.1)
            for t in workload.TASKS
        ]

    monkeypatch.setattr(
        harness, "run", lambda *args, **kwargs: (broken("routed"), broken("baseline"))
    )
    assert harness.main(["--api-key", "sk-x", "--out-dir", str(tmp_path)]) == 1
    assert "No call was answered" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["0", "1.5", "-1", "nan"])
def test_a_threshold_outside_zero_to_one_is_refused(harness: ModuleType, bad: str) -> None:
    with pytest.raises(SystemExit):
        harness.main(["--offline", "--threshold", bad])


# --- the structure of the folder ---


def test_the_harness_uses_only_the_public_package_and_no_judge_model() -> None:
    tree = ast.parse((EVALS / "harness.py").read_text(encoding="utf-8"))
    imported = [
        n.module
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("primagate")
    ]
    assert imported and set(imported) == {"primagate"}
    assert not any(
        isinstance(n, ast.Import) and any(a.name.startswith("primagate") for a in n.names)
        for n in ast.walk(tree)
    )


def test_evals_stay_outside_the_installed_package_and_the_spike_is_gone() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["primagate"]
    assert not (ROOT / "primagate" / "evals").exists()
    assert (EVALS / "harness.py").exists() and (EVALS / "workload.py").exists()
    assert not (EVALS / "spike.py").exists()


def test_the_readme_states_the_size_of_a_full_run() -> None:
    text = (EVALS / "README.md").read_text(encoding="utf-8")
    assert "300 calls" in text and "US" in text and "harness.py" in text
    assert "spike.py" not in text


def test_the_evals_files_use_unix_line_endings() -> None:
    for path in (*EVALS.glob("*.py"), EVALS / "README.md", Path(__file__)):
        assert b"\r" not in path.read_bytes(), path.name
