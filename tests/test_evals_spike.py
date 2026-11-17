import ast
import importlib
import sys
import tomllib
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).parents[1]
EVALS = ROOT / "evals"


@pytest.fixture(scope="module")
def spike() -> ModuleType:
    # `evals/` is a folder of scripts, not an installed package, so it is put on the path by hand.
    sys.path.insert(0, str(EVALS))
    try:
        return importlib.import_module("spike")
    finally:
        sys.path.remove(str(EVALS))


@pytest.fixture(scope="module")
def workload(spike: ModuleType) -> ModuleType:
    return importlib.import_module("workload")


def _summary(spike: ModuleType, **changes: Any) -> Any:
    base: dict[str, Any] = {
        "calls": 30,
        "completed": 30,
        "passes": 30,
        "usefulness": 100.0,
        "worker_cost": 1.0,
        "analyzer_cost": 0.0,
        "unknown_cost": (),
        "latency_avg": 0.1,
        "latency_p95": 0.2,
        "analyzer_latency_avg": None,
        "tiers": {},
        "degraded": 0,
        "errors": 0,
        "by_level": {"easy": (14, 14), "medium": (10, 10), "hard": (6, 6)},
        "failed": (),
    }
    return spike.Summary(**{**base, **changes})


def test_the_workload_is_thirty_tasks_in_the_promised_mix(workload: ModuleType) -> None:
    tasks = workload.TASKS
    assert len(tasks) == 30 and len({t.id for t in tasks}) == 30
    assert Counter(t.level for t in tasks) == {"easy": 14, "medium": 10, "hard": 6}


def test_every_task_passes_its_own_answer_and_fails_a_wrong_one(workload: ModuleType) -> None:
    for task in workload.TASKS:
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


@pytest.mark.parametrize(
    ("routed_cost", "routed_use", "expected"),
    [
        (0.6, 99.0, "SM-4 MET"),
        (0.8, 100.0, "SM-4 NOT MET"),
        (0.6, 95.0, "SM-4 NOT MET"),
        (0.7, 98.0, "SM-4 MET"),
        (0.71, 98.0, "SM-4 NOT MET"),
        (0.7, 97.9, "SM-4 NOT MET"),
    ],
)
def test_the_verdict_follows_the_target_exactly(
    spike: ModuleType, routed_cost: float, routed_use: float, expected: str
) -> None:
    routed = _summary(spike, worker_cost=routed_cost, usefulness=routed_use)
    line = spike.verdict(routed, _summary(spike), offline=False)
    assert line.startswith(expected + ":") and "lower" in line


def test_too_few_answers_or_an_unknown_cost_is_inconclusive(spike: ModuleType) -> None:
    few = _summary(spike, completed=29)
    assert spike.verdict(few, _summary(spike), False).startswith("INCONCLUSIVE")
    assert spike.verdict(_summary(spike), few, False).startswith("INCONCLUSIVE")
    unknown = _summary(spike, unknown_cost=("easy-01",))
    assert spike.verdict(unknown, _summary(spike), False).startswith("INCONCLUSIVE")
    free = _summary(spike, worker_cost=0.0)
    assert spike.verdict(_summary(spike), free, False).startswith("INCONCLUSIVE")


def test_an_offline_run_can_never_say_met(spike: ModuleType) -> None:
    routed = _summary(spike, worker_cost=0.1)
    line = spike.verdict(routed, _summary(spike), offline=True)
    assert line == "SM-4 NOT EVALUATED: this run used the offline stand-in"


def _result(spike: ModuleType, workload: ModuleType, **changes: Any) -> Any:
    task = workload.TASKS[0]
    fields: dict[str, Any] = {
        "task": task,
        "reply": task.answer,
        "error": None,
        "latency": 0.1,
        "worker_cost": 0.01,
    }
    return spike.CallResult(**{**fields, **changes})


def test_a_raised_call_is_a_failed_check_and_the_run_goes_on(
    spike: ModuleType, workload: ModuleType
) -> None:
    good, bad = _result(spike, workload), _result(spike, workload, reply=None, error="OSError")
    summary = spike.summarize([good, bad])
    assert (summary.calls, summary.completed, summary.passes, summary.errors) == (2, 1, 1, 1)
    assert summary.usefulness == 50.0


def test_a_cost_that_is_unknown_is_listed_and_left_out_of_the_sums(
    spike: ModuleType, workload: ModuleType
) -> None:
    priced = _result(spike, workload)
    unpriced = _result(spike, workload, worker_cost=None)
    summary = spike.summarize([priced, unpriced])
    assert summary.unknown_cost == (workload.TASKS[0].id,)
    assert summary.worker_cost == pytest.approx(0.01)


def test_the_analyzer_is_reported_apart_from_the_worker(
    spike: ModuleType, workload: ModuleType
) -> None:
    rows = [_result(spike, workload, analyzer_cost=0.002, analyzer_latency=0.05, tier="SMALL")]
    summary = spike.summarize(rows)
    assert summary.analyzer_cost == pytest.approx(0.002)
    assert summary.worker_cost == pytest.approx(0.01)
    assert summary.total_cost == pytest.approx(0.012)
    assert summary.analyzer_latency_avg == pytest.approx(0.05) and summary.tiers == {"SMALL": 1}


def test_the_95th_percentile_is_the_nearest_rank(spike: ModuleType, workload: ModuleType) -> None:
    rows = [_result(spike, workload, latency=n / 100) for n in range(1, 21)]
    assert spike.summarize(rows).latency_p95 == pytest.approx(0.19)


def test_the_offline_run_works_with_no_key_and_is_marked_not_evidence(
    spike: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: Any
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert spike.main(["--offline", "--out-dir", str(tmp_path)]) == 0
    shown = capsys.readouterr().out
    assert spike.OFFLINE_BANNER in shown and "SM-4 NOT EVALUATED" in shown
    assert "Analyzer overhead" in shown and "Analyzer cost" in shown and "Calls per Tier" in shown
    (report,) = tmp_path.glob("offline-spike-*.md")
    assert spike.OFFLINE_BANNER in report.read_text(encoding="utf-8")


def test_the_offline_run_shows_routing_cheaper_than_the_premium_arm(
    spike: ModuleType, workload: ModuleType
) -> None:
    import httpx2

    routed, fixed = spike.run(spike.Config(), "offline-key", httpx2.MockTransport(spike.stand_in))
    assert routed.completed == fixed.completed == 30 and routed.errors == 0
    assert routed.tiers == {"SMALL": 14, "MEDIUM": 10, "POWERFUL": 6}
    assert routed.analyzer_cost > 0 and routed.analyzer_latency_avg is not None
    assert routed.total_cost < fixed.total_cost and routed.usefulness == fixed.usefulness


def test_a_live_run_without_a_key_stops_before_any_call(
    spike: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: Any
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a call was made")

    monkeypatch.setattr(spike, "run", never)
    assert spike.main(["--out-dir", str(tmp_path)]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().err and list(tmp_path.iterdir()) == []


def test_the_key_never_appears_in_the_output_or_the_report(
    spike: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: Any
) -> None:
    seen: list[str] = []

    def fake_run(config: Any, key: str, transport: Any) -> Any:
        seen.append(key)
        return _summary(spike), _summary(spike)

    monkeypatch.setattr(spike, "run", fake_run)
    secret = "sk-live-secret-123"
    assert spike.main(["--api-key", secret, "--out-dir", str(tmp_path)]) == 0
    assert seen == [secret]
    (report,) = tmp_path.glob("spike-*.md")
    assert secret not in report.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    assert secret not in captured.out + captured.err


def test_a_live_report_carries_a_real_verdict_and_both_arms(spike: ModuleType) -> None:
    routed = _summary(spike, worker_cost=0.5, analyzer_cost=0.1, usefulness=99.0)
    text = spike.render_report(spike.Config(), routed, _summary(spike), False, datetime.now(UTC))
    assert "SM-4 MET" in text and spike.OFFLINE_BANNER not in text
    assert "Tier Routing" in text and "Fixed model" in text and "Analyzer overhead" in text


def test_the_spike_uses_only_the_public_package_and_no_judge_model() -> None:
    tree = ast.parse((EVALS / "spike.py").read_text(encoding="utf-8"))
    imported = [
        n.module
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("primagate")
    ]
    assert imported and set(imported) == {"primagate"}


def test_evals_stay_outside_the_installed_package() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["primagate"]
    assert not (ROOT / "primagate" / "evals").exists() and (EVALS / "spike.py").exists()


def test_a_word_answer_must_end_a_short_unnegated_reply(workload: ModuleType) -> None:
    cold = next(t for t in workload.TASKS if t.answer == "cold")
    assert cold.check("Cold.") and cold.check("The opposite is cold") and not cold.check("not cold")
    letter = next(t for t in workload.TASKS if t.answer == "b")
    assert letter.check("b") and not letter.check("a, b, c")
    assert not cold.check("cold is what the word hot is not, " + "very " * 6 + "cold")


def test_a_task_with_an_unknown_kind_is_refused(workload: ModuleType) -> None:
    with pytest.raises(ValueError):
        workload.Task("x", "easy", "p", "a", "words")


def test_an_analyzer_with_no_price_cannot_be_read_as_free(
    spike: ModuleType, workload: ModuleType
) -> None:
    unpriced = _result(spike, workload, tier="SMALL", analyzer_cost=None)
    free_looking = spike.summarize([unpriced])
    assert free_looking.unknown_cost == (workload.TASKS[0].id,)
    routed = _summary(spike, unknown_cost=free_looking.unknown_cost)
    assert spike.verdict(routed, _summary(spike), False).startswith("INCONCLUSIVE")


def test_a_degraded_call_is_counted_and_flagged_in_the_verdict(
    spike: ModuleType, workload: ModuleType
) -> None:
    rows = [_result(spike, workload, tier="MEDIUM", degraded=True, analyzer_cost=None)]
    summary = spike.summarize(rows)
    assert summary.degraded == 1 and summary.unknown_cost == ()
    routed = _summary(spike, worker_cost=0.5, degraded=3)
    assert "3 routed call(s) fell back to the Interim Binding" in spike.verdict(
        routed, _summary(spike), False
    )


def test_the_probe_notes_a_degrade_and_the_time_of_the_decision(spike: ModuleType) -> None:
    from primagate import RoutingDecision, RoutingDegraded

    fields = {
        "profile_id": "unknown",
        "input_fingerprint": "unknown",
        "prep_prompt_version": "unknown",
        "analyzer_prompt_version": "unknown",
        "binding_id": "SMALL:openai/m",
    }
    probe = spike.Probe()
    probe(RoutingDecision("r", **fields, timestamp="2026-11-17T10:00:00.250+00:00"))
    probe(RoutingDegraded("r", **fields, reason="over_budget"))
    assert probe.degraded and probe.decided_at == datetime(2026, 11, 17, 10, 0, 0, 250000, UTC)
    probe.reset()
    assert not probe.degraded and probe.decided_at is None


def test_the_added_latency_is_measured_from_the_start_and_never_negative(
    spike: ModuleType,
) -> None:
    start = datetime(2026, 11, 17, 10, 0, 0, tzinfo=UTC)
    assert spike._added(datetime(2026, 11, 17, 10, 0, 2, tzinfo=UTC), start) == 2.0
    assert spike._added(datetime(2026, 11, 17, 9, 59, 59, tzinfo=UTC), start) == 0.0
    assert spike._added(None, start) is None
    assert spike._added(datetime(2026, 11, 17, 10, 0, 2), start) is None


def test_a_report_shows_the_share_per_tier_per_level_and_the_failures(
    spike: ModuleType,
) -> None:
    routed = _summary(
        spike,
        tiers={"SMALL": 20, "MEDIUM": 10},
        by_level={"easy": (14, 14), "medium": (9, 10), "hard": (2, 6)},
        failed=("medium-20", "hard-25"),
    )
    text = spike.render_report(spike.Config(), routed, _summary(spike), False, datetime.now(UTC))
    assert "SMALL 20 (67%)" in text and "| hard | 2/6 | 6/6 |" in text
    assert "Failed in the routed arm: medium-20, hard-25" in text
    assert "interim binding MEDIUM" in text and "(indicative)" in text


def test_the_two_arms_are_asked_in_turn(spike: ModuleType, monkeypatch: Any) -> None:
    import httpx2

    order: list[str] = []
    real = spike.ask

    def spy(gate: Any, probe: Any, model: str, task: Any) -> Any:
        order.append(model)
        return real(gate, probe, model, task)

    monkeypatch.setattr(spike, "ask", spy)
    spike.run(spike.Config(), "offline-key", httpx2.MockTransport(spike.stand_in))
    assert order[:4] == ["primagate/auto", "gpt-4o"] * 2 and len(order) == 60


def test_two_runs_on_one_day_do_not_overwrite_each_other_and_the_name_has_the_models(
    spike: ModuleType, tmp_path: Path, monkeypatch: Any
) -> None:
    class Moment(datetime):
        ticks = 0

        @classmethod
        def now(cls, tz: Any = None) -> Any:
            # Each look at the clock is a second later, so two runs can never share a name.
            cls.ticks += 1
            return datetime(2026, 11, 17, 10, 0, 0, tzinfo=UTC) + timedelta(seconds=cls.ticks)

    monkeypatch.setattr(spike, "datetime", Moment)
    for _ in range(2):
        assert spike.main(["--offline", "--out-dir", str(tmp_path)]) == 0
    names = sorted(p.name for p in tmp_path.glob("offline-spike-*.md"))
    assert len(names) == 2 and all("gpt-4o-vs-gpt-4o-mini-gpt-4.1-mini-gpt-4o" in n for n in names)


def test_a_report_that_cannot_be_written_is_still_printed(
    spike: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert spike.main(["--offline", "--out-dir", str(blocker / "sub")]) == 1
    captured = capsys.readouterr()
    assert spike.OFFLINE_BANNER in captured.out and "Could not write" in captured.err


def test_a_run_where_nothing_was_answered_exits_non_zero(
    spike: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: Any
) -> None:
    none = _summary(spike, calls=30, completed=0, passes=0, usefulness=0.0, errors=30)
    monkeypatch.setattr(spike, "run", lambda *args, **kwargs: (none, none))
    assert spike.main(["--api-key", "sk-x", "--out-dir", str(tmp_path)]) == 1
    assert "No call was answered" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["0", "1.5", "-1", "nan"])
def test_a_threshold_outside_zero_to_one_is_refused(spike: ModuleType, bad: str) -> None:
    with pytest.raises(SystemExit):
        spike.main(["--offline", "--threshold", bad])
