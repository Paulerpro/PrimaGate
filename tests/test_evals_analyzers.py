"""Offline tests for the Analyzer comparison.

No test here makes a real provider call: runs use the harness's scripted stand-in, and the rest use
results built by hand.
"""

import importlib
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx2
import pytest

ROOT = Path(__file__).parents[1]
EVALS = ROOT / "evals"


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    sys.path.insert(0, str(EVALS))
    try:
        return importlib.import_module("harness")
    finally:
        sys.path.remove(str(EVALS))


@pytest.fixture(scope="module")
def analyzers(harness: ModuleType) -> ModuleType:
    return importlib.import_module("analyzers")


@pytest.fixture(scope="module")
def workload(harness: ModuleType) -> ModuleType:
    return importlib.import_module("workload")


def _call(
    harness: ModuleType,
    task: Any,
    *,
    tier: str | None = "SMALL",
    confidence: float | None = 0.9,
    passed: bool = True,
    reason: str | None = None,
    analyzer_cost: float | None = 0.001,
    latency: float | None = 0.1,
    error: str | None = None,
) -> Any:
    if error is not None:
        return harness.CallResult(task, "routed", error=error)
    good: dict[str, Any] = (
        {"reply": None, "tool_call": (task.tool, json.dumps(task.args))}
        if task.type == "tool"
        else {"reply": task.answer, "tool_call": None}
    )
    bad: dict[str, Any] = {"reply": "no", "tool_call": None}
    return harness.CallResult(
        task=task,
        arm="routed",
        **(good if passed else bad),
        latency=1.0,
        first_latency=0.5,
        worker_cost=0.01,
        analyzer_cost=analyzer_cost,
        analyzer_latency=latency,
        tier=tier,
        degraded_reason=reason,
        confidence=confidence,
    )


# --- candidates ---


def test_candidates_need_two_and_default_to_the_two_mini_models(analyzers: ModuleType) -> None:
    assert analyzers.parse_candidates("a, b ,c") == ["a", "b", "c"]
    assert analyzers.DEFAULT_CANDIDATES == ("gpt-4o-mini", "gpt-4.1-mini")
    for bad in ("a", "", "a,a", "a,,"):
        with pytest.raises(ValueError):
            analyzers.parse_candidates(bad)


def test_the_provider_of_a_candidate_follows_the_sdk_naming(analyzers: ModuleType) -> None:
    assert analyzers.provider_of("gpt-4o-mini") == "openai"
    assert analyzers.provider_of("openai/x") == "openai"
    assert analyzers.provider_of("claude-3-5-haiku") == "anthropic"
    assert analyzers.provider_of("anthropic/claude-x") == "anthropic"
    assert analyzers.provider_of("weird/x") is None


def test_a_missing_key_names_the_variable_and_no_key_is_needed_when_present(
    analyzers: ModuleType,
) -> None:
    workers = ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4o")
    assert analyzers.missing_keys(
        ["gpt-4o-mini", "claude-x"], workers, {"OPENAI_API_KEY": "k"}
    ) == ["ANTHROPIC_API_KEY"]
    assert analyzers.missing_keys(["gpt-4o-mini", "gpt-4.1-mini"], workers, {}) == [
        "OPENAI_API_KEY"
    ]
    both = {"OPENAI_API_KEY": "k", "ANTHROPIC_API_KEY": "k"}
    assert analyzers.missing_keys(["gpt-4o-mini", "claude-x"], workers, both) == []


def test_the_candidate_order_rotates_each_task(analyzers: ModuleType) -> None:
    names = ["a", "b", "c"]
    assert [analyzers.rotation(names, n) for n in range(4)] == [
        ["a", "b", "c"],
        ["b", "c", "a"],
        ["c", "a", "b"],
        ["a", "b", "c"],
    ]
    firsts = {analyzers.rotation(names, n)[0] for n in range(3)}
    assert firsts == set(names)


# --- the criteria ---


def test_the_rank_correlation_handles_ties_and_is_na_when_undefined(
    analyzers: ModuleType,
) -> None:
    assert analyzers.spearman([0, 1, 2], [0, 1, 2]) == pytest.approx(1.0)
    assert analyzers.spearman([0, 1, 2], [2, 1, 0]) == pytest.approx(-1.0)
    assert analyzers.spearman([0, 0, 1, 2], [0, 1, 1, 2]) == pytest.approx(5 / 6)
    assert analyzers.spearman([0, 1, 2], [1, 1, 1]) is None
    assert analyzers.spearman([1], [1]) is None


def _by_difficulty(
    harness: ModuleType, workload: ModuleType, tier_of: Any, **each: Any
) -> list[Any]:
    needs = {"easy": "SMALL", "medium": "MEDIUM", "hard": "POWERFUL"}
    return [_call(harness, t, tier=tier_of(needs[t.difficulty]), **each) for t in workload.TASKS]


def test_an_accurate_analyzer_scores_full_agreement_and_a_constant_one_scores_na(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    exact = analyzers.criteria(_by_difficulty(harness, workload, lambda n: n), 0.7)
    assert exact.exact == pytest.approx(1.0) and exact.at_least == pytest.approx(1.0)
    assert exact.rank == pytest.approx(1.0)
    always_medium = analyzers.criteria(_by_difficulty(harness, workload, lambda n: "MEDIUM"), 0.7)
    assert always_medium.rank is None
    assert always_medium.exact < 0.5
    assert always_medium.at_least < 1.0


def test_cost_per_routing_call_and_its_share_of_the_routed_cost(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    rows = [_call(harness, t, analyzer_cost=0.002) for t in workload.TASKS[:10]]
    c = analyzers.criteria(rows, 0.7)
    assert c.requests == 10
    assert c.cost_per_call == pytest.approx(0.002)
    assert c.cost_share == pytest.approx(0.02 / (0.02 + 0.1))
    unknown = analyzers.criteria(
        [_call(harness, t, analyzer_cost=None) for t in rows_tasks(rows)], 0.7
    )
    assert unknown.cost_per_call is None


def rows_tasks(rows: list[Any]) -> list[Any]:
    return [r.task for r in rows]


def test_added_latency_is_average_and_95th_percentile(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    rows = [_call(harness, t, latency=0.01 * (i + 1)) for i, t in enumerate(workload.TASKS[:20])]
    c = analyzers.criteria(rows, 0.7)
    assert c.latency_avg == pytest.approx(0.105)
    assert c.latency_p95 == pytest.approx(0.19)


def test_invalid_answers_lower_the_valid_rate_and_each_reason_is_counted(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    tasks = workload.TASKS
    rows = [
        _call(harness, tasks[0], tier="MEDIUM", confidence=None, reason="malformed_output"),
        _call(harness, tasks[1], tier="MEDIUM", confidence=None, reason="malformed_output"),
        _call(
            harness,
            tasks[2],
            tier="MEDIUM",
            confidence=None,
            reason="provider_error",
            analyzer_cost=None,
        ),
        _call(harness, tasks[3], tier="MEDIUM", confidence=None, reason="tier_not_enabled"),
        _call(harness, tasks[4]),
        # Never asked the Analyzer, so it is not in the rate.
        _call(harness, tasks[5], tier="MEDIUM", confidence=None, reason="over_budget"),
    ]
    c = analyzers.criteria(rows, 0.7)
    assert c.valid == 1 and c.asked == 5
    assert c.valid_rate == pytest.approx(0.2)
    assert c.invalid == {"malformed_output": 2, "provider_error": 1, "tier_not_enabled": 1}
    assert c.requests == 5
    assert c.other_degraded == 1


def test_usefulness_and_total_cost_of_the_routed_run_show_a_cheap_analyzer_that_routes_badly(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    good = analyzers.criteria([_call(harness, t) for t in workload.TASKS], 0.7)
    bad = analyzers.criteria([_call(harness, t, passed=False) for t in workload.TASKS], 0.7)
    assert good.usefulness == pytest.approx(100.0) and bad.usefulness == pytest.approx(0.0)
    assert good.total_cost == pytest.approx(0.011 * len(workload.TASKS))
    assert bad.cost_per_pass is None


# --- the threshold table ---


def _confident(
    harness: ModuleType, workload: ModuleType, plan: list[tuple[float, bool]]
) -> list[Any]:
    return [
        _call(harness, t, confidence=conf, passed=ok)
        for t, (conf, ok) in zip(workload.TASKS, plan, strict=False)
    ]


def test_the_table_gives_kept_share_and_both_pass_rates_for_five_thresholds(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    plan = [(0.95, True)] * 10 + [(0.75, True)] * 5 + [(0.75, False)] * 5 + [(0.55, False)] * 10
    rows = analyzers.threshold_table(_confident(harness, workload, plan))
    assert [r.threshold for r in rows] == [0.5, 0.6, 0.7, 0.8, 0.9]
    by = {r.threshold: r for r in rows}
    assert by[0.5].kept == 30 and by[0.5].kept_share == pytest.approx(1.0)
    assert by[0.5].kept_pass == pytest.approx(15 / 30) and by[0.5].rest_pass is None
    assert by[0.7].kept == 20 and by[0.7].kept_share == pytest.approx(20 / 30)
    assert by[0.7].kept_pass == pytest.approx(15 / 20)
    assert by[0.7].rest_pass == pytest.approx(0.0)
    assert by[0.9].kept == 10 and by[0.9].kept_pass == pytest.approx(1.0)
    assert by[0.9].rest_pass == pytest.approx(5 / 20)


def test_the_suggestion_is_the_lowest_threshold_that_keeps_half_and_passes_95_percent(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    plan = [(0.95, True)] * 30 + [(0.65, False)] * 10 + [(0.85, True)] * 20
    rows = analyzers.threshold_table(_confident(harness, workload, plan))
    # The 0.65 failures drag 0.5 and 0.6 under 95%; 0.7 is the first clean threshold.
    text = analyzers.suggestion(rows)
    assert "0.7" in text and "starting point" in text and "not a guarantee" in text
    assert "thin" not in text


def test_no_threshold_qualifying_says_so_and_claims_nothing(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    plan = [(0.95, True)] * 5 + [(0.95, False)] * 25
    text = analyzers.suggestion(analyzers.threshold_table(_confident(harness, workload, plan)))
    assert "No threshold" in text and "starting point" not in text


def test_a_thin_table_is_flagged(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    plan = [(0.95, True)] * 12
    text = analyzers.suggestion(analyzers.threshold_table(_confident(harness, workload, plan)))
    assert "thin" in text and "20" in text


def test_the_table_reads_recorded_outcomes_and_makes_no_call(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType, monkeypatch: Any
) -> None:
    def never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a call was made")

    monkeypatch.setattr(harness, "ask", never)
    rows = _confident(harness, workload, [(0.9, True)] * 5)
    assert analyzers.threshold_table(rows)[0].kept == 5


# --- a run, offline ---


@pytest.fixture(scope="module")
def offline_run(harness: ModuleType, analyzers: ModuleType) -> Any:
    return analyzers.run_comparison(
        harness.Config(),
        ["gpt-4o-mini", "gpt-4.1-mini", "stand-in-flaky"],
        {"openai": "offline-key"},
        httpx2.MockTransport(harness.stand_in),
    )


def test_every_candidate_answers_every_task_and_the_stand_in_makes_them_differ(
    analyzers: ModuleType, workload: ModuleType, offline_run: Any
) -> None:
    assert [c.name for c in offline_run] == ["gpt-4o-mini", "gpt-4.1-mini", "stand-in-flaky"]
    assert all(c.error is None and len(c.results) == len(workload.TASKS) for c in offline_run)
    crit = {c.name: analyzers.criteria(c.results, 0.7) for c in offline_run}
    assert crit["stand-in-flaky"].valid_rate < crit["gpt-4o-mini"].valid_rate
    assert crit["gpt-4o-mini"].cost_per_call != crit["gpt-4.1-mini"].cost_per_call
    assert crit["gpt-4o-mini"].exact != crit["gpt-4.1-mini"].exact
    assert crit["stand-in-flaky"].invalid.get("malformed_output", 0) > 0


def test_only_the_analyzer_differs_between_candidates(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    log: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        log.append(json.loads(request.content))
        return harness.stand_in(request)  # type: ignore[no-any-return]

    analyzers.run_comparison(
        harness.Config(),
        ["gpt-4o-mini", "gpt-4.1-mini"],
        {"openai": "offline-key"},
        httpx2.MockTransport(handler),
    )
    workers = [b for b in log if "Prompt version" not in str(b["messages"][0].get("content"))]
    assert all(b.get("temperature") == 0 for b in workers)
    # No baseline model is asked, and nothing is cached across candidates.
    assert {b["model"] for b in workers} <= {"gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"}
    assert len(workers) == 2 * len(workload.TASKS)


def test_the_order_of_asking_rotates_between_tasks(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    order: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if "Prompt version: analyzer" in str(body["messages"][0].get("content")):
            order.append(body["model"])
        return harness.stand_in(request)  # type: ignore[no-any-return]

    analyzers.run_comparison(
        harness.Config(),
        ["gpt-4o-mini", "gpt-4.1-mini"],
        {"openai": "offline-key"},
        httpx2.MockTransport(handler),
    )
    assert order[0] == "gpt-4o-mini" and order[1] == "gpt-4.1-mini"
    assert order[2] == "gpt-4.1-mini" and order[3] == "gpt-4o-mini"


def test_a_candidate_that_cannot_be_set_up_is_listed_as_failed_and_the_rest_still_run(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    run = analyzers.run_comparison(
        harness.Config(),
        ["gpt-4o-mini", "nope/model"],
        {"openai": "offline-key"},
        httpx2.MockTransport(harness.stand_in),
    )
    good, broken = run
    assert good.error is None and len(good.results) == len(workload.TASKS)
    assert broken.error is not None and broken.results == []
    text = analyzers.comparison_section(harness.Config(), run, True)
    assert "nope/model" in text and "failed" in text.lower()
    assert analyzers.all_answered(run) is False


# --- the report and the data ---


def test_the_section_reports_each_criterion_per_candidate_with_no_best_analyzer(
    harness: ModuleType, analyzers: ModuleType, offline_run: Any
) -> None:
    text = analyzers.comparison_section(harness.Config(), offline_run, True)
    assert text.startswith("## Analyzer comparison")
    assert harness.OFFLINE_BANNER in text
    for needle in (
        "Cost per routing call",
        "Added latency",
        "Valid-output rate",
        "Tier-to-difficulty",
        "Calibration",
        "confidence_threshold",
        "Leads",
        "malformed_output",
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "stand-in-flaky",
    ):
        assert needle in text, needle
    lowered = text.lower()
    assert "winner" not in lowered and "best analyzer" not in lowered
    assert "no analyzer is named as best" in lowered and "owner" not in lowered
    assert text.count("| 0.5 |") >= 3 and text.count("| 0.9 |") >= 3


def test_a_real_run_has_no_offline_banner(
    harness: ModuleType, analyzers: ModuleType, offline_run: Any
) -> None:
    text = analyzers.comparison_section(harness.Config(), offline_run, False)
    assert harness.OFFLINE_BANNER not in text


def test_leaders_are_per_criterion_and_may_be_shared(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    a = analyzers.Candidate("a", [_call(harness, t, analyzer_cost=0.001) for t in workload.TASKS])
    b = analyzers.Candidate("b", [_call(harness, t, analyzer_cost=0.002) for t in workload.TASKS])
    leads = analyzers.leaders([a, b], 0.7)
    assert leads["Cost per routing call"] == ["a"]
    assert leads["Valid-output rate"] == ["a", "b"]


def test_jev_is_labelled_like_any_candidate_and_its_calibration_comes_from_data(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    rows = [_call(harness, t) for t in workload.TASKS]
    run = [analyzers.Candidate("Jev-1", rows), analyzers.Candidate("gpt-4o-mini", rows)]
    text = analyzers.comparison_section(harness.Config(), run, False)
    assert text.count("early access, unvalidated") == 1
    assert "Jev-1 (early access, unvalidated)" in text
    # One confidence bucket only, so the real branch is the "too few" statement.
    assert "Too few routed calls with a confidence to say" in text
    assert "calibration is not assumed" in text


def test_the_data_holds_facts_only_and_no_key_or_text(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType, offline_run: Any
) -> None:
    data = analyzers.comparison_data(harness.Config(), offline_run, True)
    text = json.dumps(data)
    assert "offline-key" not in text and "I am not sure" not in text
    for task in workload.TASKS:
        assert task.prompt not in text
    cand = data["candidates"]["gpt-4o-mini"]
    for key in ("criteria", "thresholds", "suggestion", "calibration", "calls"):
        assert key in cand
    assert [t["threshold"] for t in cand["thresholds"]] == [0.5, 0.6, 0.7, 0.8, 0.9]
    assert "leaders" in data and data["offline"] is True


# --- through the command line ---


def test_one_candidate_is_refused_with_exit_2(
    harness: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as err:
        harness.main(
            ["--offline", "--compare-analyzers", "gpt-4o-mini", "--out-dir", str(tmp_path)]
        )
    assert err.value.code == 2
    assert "at least two" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_a_missing_key_stops_before_any_call_and_names_the_variable(
    harness: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: Any
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a call was made")

    monkeypatch.setattr(harness, "run", never)
    code = harness.main(
        ["--compare-analyzers", "gpt-4o-mini,claude-3-5-haiku-latest", "--out-dir", str(tmp_path)]
    )
    err = capsys.readouterr().err
    assert code == 2 and "ANTHROPIC_API_KEY" in err and "sk-test-secret" not in err
    assert list(tmp_path.iterdir()) == []


def test_the_default_candidates_are_used_when_none_are_named(
    harness: ModuleType, analyzers: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert harness.main(["--offline", "--compare-analyzers", "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "## Analyzer comparison" in out and "gpt-4.1-mini" in out


def _compare_args(tmp_path: Path, *more: str) -> list[str]:
    return ["--offline", "--out-dir", str(tmp_path), *more]


def test_the_comparison_is_its_own_mode_with_no_baseline_or_main_evaluation(
    harness: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: Any
) -> None:
    def never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the main evaluation ran")

    monkeypatch.setattr(harness, "run", never)
    code = harness.main(_compare_args(tmp_path, "--compare-analyzers", "gpt-4o-mini,gpt-4.1-mini"))
    assert code == 0
    out = capsys.readouterr().out
    assert "# Tier Routing evaluation" not in out
    assert "## Analyzer comparison" in out and harness.OFFLINE_BANNER in out
    (report,) = tmp_path.glob("offline-analyzers-*.md")
    (results,) = tmp_path.glob("offline-analyzers-*.json")
    data = json.loads(results.read_text(encoding="utf-8"))
    assert data["version"] == harness.RESULTS_VERSION == 3
    assert data["calls"] == []
    assert set(data["analyzer_comparison"]["candidates"]) == {"gpt-4o-mini", "gpt-4.1-mini"}
    assert all(b"\r" not in p.read_bytes() for p in tmp_path.iterdir())
    text = report.read_text(encoding="utf-8") + results.read_text(encoding="utf-8")
    assert "offline-key" not in text


@pytest.mark.parametrize(
    "extra", [["--experiment", "all"], ["--export-rating"], ["--score-rating", "x.csv"]]
)
def test_the_comparison_does_not_combine_with_the_other_modes(
    harness: ModuleType, tmp_path: Path, extra: list[str]
) -> None:
    with pytest.raises(SystemExit) as err:
        harness.main(_compare_args(tmp_path, "--compare-analyzers", "a,b", *extra))
    assert err.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_an_empty_candidate_list_and_seven_candidates_are_refused(
    harness: ModuleType, analyzers: ModuleType, tmp_path: Path
) -> None:
    for value in ("", ",,"):
        with pytest.raises(SystemExit) as err:
            harness.main(_compare_args(tmp_path, "--compare-analyzers", value))
        assert err.value.code == 2
    assert analyzers.MAX_CANDIDATES == 6
    assert len(analyzers.parse_candidates("a,b,c,d,e,f")) == 6
    with pytest.raises(SystemExit) as err:
        harness.main(_compare_args(tmp_path, "--compare-analyzers", "a,b,c,d,e,f,g"))
    assert err.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_base_url_needs_the_comparison_and_the_help_says_it_reaches_the_workers(
    harness: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as err:
        harness.main(_compare_args(tmp_path, "--base-url", "https://x.example/v1"))
    assert err.value.code == 2
    assert "including the worker" in " ".join(harness._parser().format_help().split())


def test_with_a_base_url_every_request_goes_there(
    harness: ModuleType, analyzers: ModuleType
) -> None:
    urls: list[str] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        urls.append(str(request.url))
        return harness.stand_in(request)  # type: ignore[no-any-return]

    analyzers.run_comparison(
        harness.Config(),
        ["gpt-4o-mini", "gpt-4.1-mini"],
        {"openai": "offline-key"},
        httpx2.MockTransport(handler),
        {"openai": "https://proxy.example/v1"},
    )
    assert urls and all(u.startswith("https://proxy.example/v1") for u in urls)


def _capture(analyzers: ModuleType, monkeypatch: Any, seen: dict[str, Any]) -> None:
    def fake(config: Any, candidates: Any, keys: Any, transport: Any, base_urls: Any = None) -> Any:
        seen.update(keys=dict(keys), base_urls=base_urls, transport=transport)
        return [analyzers.Candidate(n, [], "Stopped") for n in candidates]

    monkeypatch.setattr(analyzers, "run_comparison", fake)


def test_each_provider_gets_its_own_key_and_no_base_url_is_set_by_default(
    harness: ModuleType,
    analyzers: ModuleType,
    tmp_path: Path,
    monkeypatch: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    seen: dict[str, Any] = {}
    _capture(analyzers, monkeypatch, seen)
    args = ["--out-dir", str(tmp_path), "--compare-analyzers", "gpt-4o-mini,claude-3-5-haiku"]
    harness.main(args)
    assert seen["keys"] == {"openai": "sk-openai", "anthropic": "sk-anthropic"}
    assert seen["base_urls"] is None and seen["transport"] is None
    err = capsys.readouterr()
    assert "sk-openai" not in err.out + err.err and "sk-anthropic" not in err.out + err.err
    # The estimate is printed from the same constants the docs quote.
    assert analyzers.cost_note() in err.err
    harness.main([*args, "--base-url", "https://x.example/v1"])
    assert seen["base_urls"] == {"openai": "https://x.example/v1"}


def test_an_unknown_provider_prefix_stops_before_any_call(
    harness: ModuleType,
    analyzers: ModuleType,
    tmp_path: Path,
    monkeypatch: Any,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    seen: dict[str, Any] = {}
    _capture(analyzers, monkeypatch, seen)
    code = harness.main(["--out-dir", str(tmp_path), "--compare-analyzers", "gpt-4o-mini,weird/x"])
    assert code == 2 and seen == {} and "weird/x" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_a_failed_candidate_gives_exit_1_and_the_others_are_still_compared(
    harness: ModuleType,
    analyzers: ModuleType,
    offline_run: Any,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: Any,
) -> None:
    failed = analyzers.Candidate("broken", [], "BadRequestError")
    monkeypatch.setattr(analyzers, "run_comparison", lambda *a, **k: [offline_run[0], failed])
    code = harness.main(_compare_args(tmp_path, "--compare-analyzers", "gpt-4o-mini,broken"))
    assert code == 1
    out = capsys.readouterr().out
    assert "`broken` failed (BadRequestError)" in out and "gpt-4o-mini" in out
    (results,) = tmp_path.glob("offline-analyzers-*.json")
    data = json.loads(results.read_text(encoding="utf-8"))["analyzer_comparison"]
    assert data["candidates"]["broken"]["failed"] is True
    assert data["candidates"]["gpt-4o-mini"]["failed"] is False


def test_a_comparison_that_raises_or_cannot_be_written_exits_1_with_no_text(
    harness: ModuleType,
    analyzers: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: Any,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("comparison broke")

    monkeypatch.setattr(analyzers, "run_comparison", boom)
    args = _compare_args(tmp_path, "--compare-analyzers", "gpt-4o-mini,gpt-4.1-mini")
    assert harness.main(args) == 1
    captured = capsys.readouterr()
    assert "RuntimeError" in captured.err and "comparison broke" not in captured.err
    assert list(tmp_path.iterdir()) == []


def test_both_files_are_built_before_either_is_written(
    harness: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: Any
) -> None:
    real = Path.write_text

    def fail_json(self: Path, *args: Any, **kwargs: Any) -> int:
        if self.suffix == ".json":
            raise OSError("disk full")
        return real(self, *args, **kwargs)  # type: ignore[no-any-return]

    monkeypatch.setattr(Path, "write_text", fail_json)
    args = _compare_args(tmp_path, "--compare-analyzers", "gpt-4o-mini,gpt-4.1-mini")
    assert harness.main(args) == 1
    assert "Could not write" in capsys.readouterr().err


def test_the_comparison_flag_reaches_the_workers_in_the_docs_and_the_cost_matches(
    analyzers: ModuleType, workload: ModuleType
) -> None:
    calls, low, high = analyzers.cost_estimate()
    assert calls == 2 * len(workload.TASKS)
    note = analyzers.cost_note()
    assert f"about {calls} calls" in note and f"{low:.0f} to {high:.0f} US cents" in note
    for path in (ROOT / "docs" / "analyzer-guide.md", EVALS / "README.md"):
        text = " ".join(path.read_text(encoding="utf-8").split())
        assert f"about {calls} calls" in text and f"{low:.0f} to {high:.0f} US cents" in text
        assert "including the worker" in " ".join(text.split())


# --- the criteria, in the cases a review found ---


def test_calls_the_analyzer_failed_on_count_against_the_kept_share(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    tasks = workload.TASKS
    good = [_call(harness, t, confidence=0.95) for t in tasks[:8]]
    bad = [
        _call(harness, t, tier="MEDIUM", confidence=None, reason="malformed_output")
        for t in tasks[8:20]
    ]
    table = analyzers.threshold_table([*good, *bad])
    assert table[0].kept == 8 and table[0].kept_share == pytest.approx(8 / 20)
    assert table[0].rest == 12
    assert "No threshold" in analyzers.suggestion(table)


def test_only_thresholds_at_or_above_the_run_threshold_are_offered(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    rows = _confident(harness, workload, [(0.6, True)] * 30)
    table = analyzers.threshold_table(rows, 0.7)
    assert [t.approximate for t in table] == [True, True, False, False, False]
    assert "0.5" in analyzers.suggestion(analyzers.threshold_table(rows))
    assert "No threshold" in analyzers.suggestion(table, 0.7)


def test_the_report_marks_rows_under_the_run_threshold_approximate(
    harness: ModuleType, analyzers: ModuleType, offline_run: Any
) -> None:
    text = analyzers.comparison_section(harness.Config(), offline_run, True)
    assert "| 0.5 | " in text and "(approximate)" in text
    row = next(line for line in text.splitlines() if line.startswith("| 0.9 |"))
    assert "approximate" not in row


def test_a_candidate_with_no_usable_answer_is_failed_and_never_leads(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    run = analyzers.run_comparison(
        harness.Config(),
        ["gpt-4o-mini", "stand-in-broken"],
        {"openai": "offline-key"},
        httpx2.MockTransport(harness.stand_in),
    )
    assert run[0].error is None and run[1].error is not None
    assert analyzers.all_answered(run) is False
    assert all("stand-in-broken" not in who for who in analyzers.leaders(run, 0.7).values())
    unusable = analyzers.Candidate(
        "z",
        [
            _call(harness, t, tier="MEDIUM", confidence=None, reason="malformed_output")
            for t in workload.TASKS[:5]
        ],
    )
    ok = analyzers.Candidate("a", [_call(harness, t) for t in workload.TASKS[:5]])
    assert analyzers.leaders([ok, unusable], 0.7)["Valid-output rate"] == ["a"]


def test_cost_per_routing_call_is_na_when_no_cost_was_seen(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    rows = [
        _call(
            harness, t, tier="MEDIUM", confidence=None, reason="provider_error", analyzer_cost=None
        )
        for t in workload.TASKS[:5]
    ]
    assert analyzers.criteria(rows, 0.7).cost_per_call is None


def test_a_call_below_the_run_threshold_still_counts_as_an_analyzer_request_with_cost(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    rows = [
        _call(harness, workload.TASKS[0], tier="MEDIUM", confidence=0.4, analyzer_cost=0.003),
        _call(harness, workload.TASKS[1], analyzer_cost=0.001),
    ]
    c = analyzers.criteria(rows, 0.7)
    assert c.requests == 2 and c.cost_per_call == pytest.approx(0.002)


def test_added_latency_includes_unusable_answers(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    tasks = workload.TASKS
    rows = [
        _call(harness, tasks[0], latency=0.1),
        _call(
            harness,
            tasks[1],
            tier="MEDIUM",
            confidence=None,
            reason="malformed_output",
            latency=0.3,
        ),
        _call(
            harness, tasks[2], tier="MEDIUM", confidence=None, reason="provider_error", latency=0.5
        ),
        # Never asked, so its decision time is not the Analyzer's.
        _call(harness, tasks[3], tier="MEDIUM", confidence=None, reason="over_budget", latency=9.0),
    ]
    assert analyzers.criteria(rows, 0.7).latency_avg == pytest.approx(0.3)


def test_one_raising_call_fails_only_its_candidate(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType, monkeypatch: Any
) -> None:
    real = harness.ask
    seen: list[int] = []

    def flaky(gate: Any, probe: Any, model: str, task: Any, arm: str) -> Any:
        seen.append(1)
        if len(seen) == 1:
            raise RuntimeError("boom")
        return real(gate, probe, model, task, arm)

    monkeypatch.setattr(harness, "ask", flaky)
    first, second = analyzers.run_comparison(
        harness.Config(),
        ["gpt-4o-mini", "gpt-4.1-mini"],
        {"openai": "offline-key"},
        httpx2.MockTransport(harness.stand_in),
    )
    assert first.error == "RuntimeError"
    assert second.error is None and len(second.results) == len(workload.TASKS)


def test_candidate_names_cannot_break_a_table_and_jev_is_sniffed_in_one_place(
    harness: ModuleType, analyzers: ModuleType, workload: ModuleType
) -> None:
    rows = [_call(harness, t) for t in workload.TASKS]
    run = [analyzers.Candidate("a|b`c", rows), analyzers.Candidate("gpt-4o-mini", rows)]
    text = analyzers.comparison_section(harness.Config(), run, False)
    escaped = "a\\|b'c"
    assert f"`{escaped}`" in text and "a|b" not in text.replace(escaped, "")
    header = next(line for line in text.splitlines() if line.startswith("| Criterion"))
    assert header.replace("\\|", "").count("|") == 4
    assert analyzers.is_early_access("My-JEV-1") and not analyzers.is_early_access("gpt-4o-mini")


# --- docs and code rules ---


def test_the_module_uses_only_the_public_api_and_has_no_story_refs() -> None:
    source = (EVALS / "analyzers.py").read_text(encoding="utf-8")
    assert "from primagate." not in source and "import primagate." not in source
    assert not re.search(r"\b(Story|Epic)\b|(?<![\d.])8\.\d\b", source) and "\r" not in source


def test_the_guide_and_the_readmes_cover_the_comparison() -> None:
    guide = (ROOT / "docs" / "analyzer-guide.md").read_text(encoding="utf-8")
    for needle in (
        "Cost per routing call",
        "Added latency",
        "Valid-output rate",
        "Tier-to-difficulty",
        "Calibration",
        "confidence_threshold",
        "--compare-analyzers",
        "Anthropic",
        "privacy",
        "clone of the repository",
        "`n/a`",
        "failed (",
        "rerun",
        "early-access",
    ):
        assert needle in guide, needle
    assert "\r" not in guide
    assert "owner" not in guide and "special code" not in guide
    evals = (EVALS / "README.md").read_text(encoding="utf-8")
    assert "--compare-analyzers" in evals and "analyzer-guide.md" in evals
    assert "version 3" in evals and "analyzer_comparison" in evals
    top = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "analyzer-guide.md" in top
    assert all(len(line) <= 100 for line in top.splitlines() if "analyzer-guide.md" in line)


def test_the_guide_states_the_real_analyzer_request_settings() -> None:
    import inspect

    from primagate.core import analyzer as core

    guide = (ROOT / "docs" / "analyzer-guide.md").read_text(encoding="utf-8")
    assert "temperature=0" in inspect.getsource(core.begin_analysis)
    assert "`temperature` 0" in guide
    assert f"`max_tokens` ({core._MAX_ANSWER_TOKENS})" in guide


def test_the_guides_readme_anchor_exists() -> None:
    guide = (ROOT / "docs" / "analyzer-guide.md").read_text(encoding="utf-8")
    (anchor,) = re.findall(r"\(\.\./README\.md#([^)]+)\)", guide)
    slugs = set()
    for line in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            title = line.lstrip("#").strip().lower()
            slugs.add(re.sub(r"[^a-z0-9 -]", "", title).replace(" ", "-"))
    assert anchor in slugs
