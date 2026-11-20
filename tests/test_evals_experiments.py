"""Offline tests for the isolation experiments and the human-rated sample.

No test here makes a real provider call: every one runs against the harness's scripted stand-in or
against results built by hand.
"""

import csv
import importlib
import json
import re
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx2
import pytest

ROOT = Path(__file__).parents[1]
EVALS = ROOT / "evals"
CHECK = httpx2.MockTransport


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    sys.path.insert(0, str(EVALS))
    try:
        return importlib.import_module("harness")
    finally:
        sys.path.remove(str(EVALS))


@pytest.fixture(scope="module")
def experiments(harness: ModuleType) -> ModuleType:
    return importlib.import_module("experiments")


@pytest.fixture(scope="module")
def rating(harness: ModuleType) -> ModuleType:
    return importlib.import_module("rating")


@pytest.fixture(scope="module")
def workload(harness: ModuleType) -> ModuleType:
    return importlib.import_module("workload")


def _recording(harness: ModuleType, log: list[dict[str, Any]]) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        log.append(json.loads(request.content))
        return harness.stand_in(request)  # type: ignore[no-any-return]

    return httpx2.MockTransport(handler)


@pytest.fixture(scope="module")
def control(harness: ModuleType) -> tuple[list[Any], list[Any]]:
    return harness.run(  # type: ignore[no-any-return]
        harness.Config(), "offline-key", httpx2.MockTransport(harness.stand_in)
    )


@pytest.fixture(scope="module")
def profile_run(
    harness: ModuleType, experiments: ModuleType, control: tuple[list[Any], list[Any]]
) -> Any:
    return experiments.run_profile(
        harness.Config(), "offline-key", httpx2.MockTransport(harness.stand_in), 10.0
    )


@pytest.fixture(scope="module")
def history_arm(
    harness: ModuleType, experiments: ModuleType, control: tuple[list[Any], list[Any]]
) -> list[Any]:
    return experiments.run_history(  # type: ignore[no-any-return]
        harness.Config(), "offline-key", httpx2.MockTransport(harness.stand_in)
    )


def _result(harness: ModuleType, task: Any, arm: str = "routed", **each: Any) -> Any:
    good = (
        {"reply": None, "tool_call": (task.tool, json.dumps(task.args))}
        if task.type == "tool"
        else {"reply": task.answer, "tool_call": None}
    )
    fields = {
        "worker_cost": 0.01,
        "analyzer_cost": 0.001,
        "latency": 0.2,
        "first_latency": 0.1,
        "tier": "SMALL",
        **good,
        **each,
    }
    return harness.CallResult(task=task, arm=arm, **fields)


# --- the Policy setting the history experiment turns off ---


def test_the_harness_builds_its_gate_with_the_exchange_count_it_is_given(
    harness: ModuleType,
) -> None:
    probe = harness.Probe()
    default = harness.build_gate(harness.Config(), "k", None, probe)
    assert default._policy.context_exchanges == 3
    none = harness.build_gate(harness.Config(), "k", None, probe, context_exchanges=0)
    assert none._policy.context_exchanges == 0


# --- the stand-in behaves differently with a Profile and with history ---


def test_the_stand_in_prepares_a_profile_when_asked_to(
    harness: ModuleType, experiments: ModuleType
) -> None:
    log: list[dict[str, Any]] = []
    run = experiments.run_profile(harness.Config(), "k", _recording(harness, log), 10.0)
    assert run.ready and run.reason is None
    prep = [b for b in log if "Prompt version: prep" in str(b["messages"][0]["content"])]
    assert len(prep) == 1


def test_the_profile_arm_declares_the_workloads_instructions_and_tool_descriptions(
    experiments: ModuleType, workload: ModuleType
) -> None:
    sources = experiments.profile_sources()
    assert sources["instructions"].strip()
    used = {name for task in workload.TASKS for name in task.tools}
    for name in used:
        spec = workload.TOOL_SPECS[name]["function"]
        assert any(
            name in text and spec["description"] in text for text in sources["tool_descriptions"]
        )
    assert len(sources["tool_descriptions"]) == len(used)


def test_the_analyzer_reads_the_profile_in_the_profile_arm_only(
    harness: ModuleType, experiments: ModuleType
) -> None:
    log: list[dict[str, Any]] = []
    experiments.run_profile(harness.Config(), "k", _recording(harness, log), 10.0)
    analyzer = [
        str(b["messages"][0]["content"])
        for b in log
        if "Prompt version: analyzer" in str(b["messages"][0]["content"])
    ]
    assert analyzer and all("Purpose:" in text for text in analyzer)


# --- the Profile experiment ---


def test_the_profile_arm_asks_every_task_and_none_was_served_while_preparing(
    workload: ModuleType, profile_run: Any
) -> None:
    arm = profile_run.results
    assert profile_run.ready
    assert [r.task.id for r in arm] == [t.id for t in workload.TASKS]
    assert {r.arm for r in arm} == {"profile"}
    assert not any(r.not_ready for r in arm) and all(r.error is None for r in arm)


def test_the_wait_for_the_profile_is_not_one_of_the_routed_calls(
    harness: ModuleType, experiments: ModuleType, monkeypatch: Any
) -> None:
    seen: list[str] = []
    real = harness.ask

    def spy(gate: Any, probe: Any, model: str, task: Any, arm: str) -> Any:
        seen.append(arm)
        return real(gate, probe, model, task, arm)

    monkeypatch.setattr(harness, "ask", spy)
    run = experiments.run_profile(
        harness.Config(), "k", httpx2.MockTransport(harness.stand_in), 10.0
    )
    # One call only starts the preparation; it is not in the results.
    assert seen.count("profile") == 100 and seen.count("warm-up") == 1
    assert len(run.results) == 100


def test_the_stand_in_gives_the_profile_arm_different_numbers(
    harness: ModuleType, control: tuple[list[Any], list[Any]], profile_run: Any
) -> None:
    routed = harness.summarize(control[0])
    real = harness.summarize(profile_run.results)
    assert real.passes > routed.passes
    assert real.total_cost != routed.total_cost and real.tiers != routed.tiers


def test_a_profile_that_fails_to_prepare_runs_nothing_and_says_why(
    harness: ModuleType, experiments: ModuleType
) -> None:
    def broken(request: httpx2.Request) -> httpx2.Response:
        if "Prompt version: prep" in str(json.loads(request.content)["messages"][0]["content"]):
            return httpx2.Response(500, json={"error": {"message": "x", "type": "server_error"}})
        return harness.stand_in(request)  # type: ignore[no-any-return]

    run = experiments.run_profile(harness.Config(), "k", httpx2.MockTransport(broken), 10.0)
    assert not run.ready and run.results == []
    assert run.reason == "provider_error"
    text = experiments.profile_section([], run, False)
    assert text.startswith(
        "## Profile experiment\n\n**NOT RUN: Profile not ready (provider_error)**"
    )
    assert "NOT RUN: Profile not ready" in text and "provider_error" in text
    assert "cost fell" not in text and "cost rose" not in text and "Usefulness" not in text


def test_a_profile_that_never_becomes_ready_times_out_and_says_so(
    harness: ModuleType, experiments: ModuleType
) -> None:
    release = threading.Event()

    def stuck(request: httpx2.Request) -> httpx2.Response:
        if "Prompt version: prep" in str(json.loads(request.content)["messages"][0]["content"]):
            release.wait(3)
        return harness.stand_in(request)  # type: ignore[no-any-return]

    try:
        run = experiments.run_profile(harness.Config(), "k", httpx2.MockTransport(stuck), 0.2)
    finally:
        release.set()
    assert not run.ready and run.results == [] and "timeout" in run.reason
    assert "NOT RUN: Profile not ready" in experiments.profile_section([], run, False)


def test_calls_served_while_the_profile_was_not_ready_are_flagged_and_left_out(
    harness: ModuleType, experiments: ModuleType, workload: ModuleType
) -> None:
    tasks = workload.TASKS[:10]
    control = [_result(harness, t) for t in tasks]
    arm = [
        _result(harness, t, arm="profile", degraded_reason="preparing" if n < 2 else None)
        for n, t in enumerate(tasks)
    ]
    assert arm[0].not_ready and not arm[2].not_ready
    run = experiments.ProfileRun(True, None, arm)
    text = experiments.profile_section(control, run, False)
    assert "2 call(s) served while the Profile was not ready" in text
    assert "8 task(s)" in text
    data = experiments.experiments_data(run, None)
    flagged = [c for c in data["profile"]["calls"] if c["profile_not_ready"]]
    assert len(flagged) == 2


def test_the_profile_section_gives_both_arms_directions_and_no_cause(
    harness: ModuleType,
    experiments: ModuleType,
    control: tuple[list[Any], list[Any]],
    profile_run: Any,
) -> None:
    text = experiments.profile_section(control[0], profile_run, False)
    for needle in (
        "Empty Profile",
        "Real Profile",
        "Total cost",
        "Usefulness",
        "Analyzer overhead",
        "Tier chosen by difficulty",
        "at least as capable",
    ):
        assert needle in text, needle
    assert "points higher" in text or "points lower" in text or "did not change" in text
    assert "cost fell" in text or "cost rose" in text or "cost did not change" in text
    for cause in ("because", "caused", "due to", "proves"):
        assert cause not in text
    assert "SM-4" not in text


def test_the_capable_enough_share_counts_a_tier_at_or_above_what_the_task_needed(
    harness: ModuleType, experiments: ModuleType, workload: ModuleType
) -> None:
    easy = next(t for t in workload.TASKS if t.difficulty == "easy")
    hard = next(t for t in workload.TASKS if t.difficulty == "hard")
    rows = [
        _result(harness, easy, tier="POWERFUL"),
        _result(harness, hard, tier="MEDIUM"),
        _result(harness, hard, tier="POWERFUL"),
    ]
    assert experiments.capable_enough(rows) == pytest.approx(2 / 3)
    assert experiments.capable_enough([]) is None


# --- the history experiment ---


def test_the_history_arm_runs_only_the_multi_turn_tasks_with_no_earlier_exchanges(
    harness: ModuleType, experiments: ModuleType, workload: ModuleType
) -> None:
    log: list[dict[str, Any]] = []
    arm = experiments.run_history(harness.Config(), "k", _recording(harness, log))
    multi = [t.id for t in workload.TASKS if t.type == "multiturn"]
    assert [r.task.id for r in arm] == multi and len(multi) == 20
    assert {r.arm for r in arm} == {"no_history"}
    analyzer = [b for b in log if "Prompt version: analyzer" in str(b["messages"][0]["content"])]
    assert len(analyzer) == 20
    by_prompt = {t.prompt: t for t in workload.TASKS if t.type == "multiturn"}
    for body in analyzer:
        text = str(body["messages"][1]["content"])
        task = next(t for p, t in by_prompt.items() if p in text)
        assert all(question not in text for question, _ in task.history)
    # The worker still gets the whole conversation, so only the Analyzer's view differs.
    workers = [b for b in log if "Prompt version" not in str(b["messages"][0]["content"])]
    assert all(len(b["messages"]) > 1 for b in workers)


def test_the_stand_in_gives_the_no_history_arm_different_numbers(
    harness: ModuleType, control: tuple[list[Any], list[Any]], history_arm: list[Any]
) -> None:
    kept = [r for r in control[0] if r.task.type == "multiturn"]
    a, b = harness.summarize(kept), harness.summarize(history_arm)
    assert b.passes < a.passes and b.total_cost != a.total_cost


def test_the_history_section_gives_usefulness_cost_agreement_and_the_small_sample_caveat(
    experiments: ModuleType, control: tuple[list[Any], list[Any]], history_arm: list[Any]
) -> None:
    text = experiments.history_section(control[0], history_arm, False)
    for needle in (
        "3 earlier exchanges",
        "No history",
        "Usefulness",
        "Total cost",
        "Tier agreement",
        "20 tasks",
        "5 points",
    ):
        assert needle in text, needle
    for cause in ("because", "caused", "proves"):
        assert cause not in text


def test_tier_agreement_is_the_share_of_tasks_sent_to_the_tier_they_needed(
    harness: ModuleType, experiments: ModuleType, workload: ModuleType
) -> None:
    easy = next(t for t in workload.TASKS if t.difficulty == "easy")
    hard = next(t for t in workload.TASKS if t.difficulty == "hard")
    rows = [_result(harness, easy, tier="SMALL"), _result(harness, hard, tier="MEDIUM")]
    assert experiments.tier_agreement(rows) == 0.5
    assert experiments.tier_agreement([]) is None


# --- the whole command ---


def test_all_runs_both_experiments_and_the_control_arm_once(
    harness: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: Any,
) -> None:
    runs: list[int] = []
    real = harness.run

    def counted(*args: Any, **kwargs: Any) -> Any:
        runs.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(harness, "run", counted)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = harness.main(["--offline", "--experiment", "all", "--out-dir", str(tmp_path)])
    assert code == 0 and len(runs) == 1
    shown = capsys.readouterr().out
    assert harness.OFFLINE_BANNER in shown
    assert "## Profile experiment" in shown and "## History experiment" in shown
    (report,) = tmp_path.glob("offline-harness-*.md")
    text = report.read_text(encoding="utf-8")
    assert "## Profile experiment" in text and "## History experiment" in text
    (results,) = tmp_path.glob("offline-harness-*.json")
    data = json.loads(results.read_text(encoding="utf-8"))
    assert data["version"] == harness.RESULTS_VERSION == 3
    assert set(data["experiments"]) == {"profile", "history"}
    assert data["experiments"]["profile"]["status"] == "ran"
    assert len(data["experiments"]["profile"]["calls"]) == 100
    assert len(data["experiments"]["history"]["calls"]) == 20
    assert "profile_not_ready" in data["calls"][0]


@pytest.mark.parametrize(
    ("choice", "present", "absent"),
    [
        ("profile", "Profile experiment", "History experiment"),
        ("history", "History experiment", "Profile experiment"),
    ],
)
def test_one_experiment_adds_only_its_own_section(
    harness: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    choice: str,
    present: str,
    absent: str,
) -> None:
    assert harness.main(["--offline", "--experiment", choice, "--out-dir", str(tmp_path)]) == 0
    shown = capsys.readouterr().out
    assert f"## {present}" in shown and f"## {absent}" not in shown
    (results,) = tmp_path.glob("offline-harness-*.json")
    assert set(json.loads(results.read_text(encoding="utf-8"))["experiments"]) == {choice}


def test_a_plain_run_has_no_experiment_section_and_the_results_say_so(
    harness: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert harness.main(["--offline", "--out-dir", str(tmp_path)]) == 0
    assert "experiment" not in capsys.readouterr().out.lower().replace("isolation", "")
    (results,) = tmp_path.glob("offline-harness-*.json")
    assert json.loads(results.read_text(encoding="utf-8"))["experiments"] == {}


def test_an_unknown_experiment_is_refused(harness: ModuleType) -> None:
    with pytest.raises(SystemExit) as stopped:
        harness.main(["--offline", "--experiment", "cache"])
    assert stopped.value.code == 2


def test_a_profile_that_is_not_ready_exits_one_and_the_report_says_not_run(
    harness: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: Any,
) -> None:
    real = harness.stand_in

    def broken(request: httpx2.Request) -> httpx2.Response:
        if "Prompt version: prep" in str(json.loads(request.content)["messages"][0]["content"]):
            return httpx2.Response(500, json={"error": {"message": "x", "type": "server_error"}})
        return real(request)  # type: ignore[no-any-return]

    monkeypatch.setattr(harness, "stand_in", broken)
    code = harness.main(["--offline", "--experiment", "profile", "--out-dir", str(tmp_path)])
    assert code == 1
    shown = capsys.readouterr().out
    assert "NOT RUN: Profile not ready" in shown and "provider_error" in shown
    (results,) = tmp_path.glob("offline-harness-*.json")
    profile = json.loads(results.read_text(encoding="utf-8"))["experiments"]["profile"]
    assert profile["status"] == "not_run" and profile["calls"] == []


def test_an_experiment_without_a_key_stops_before_any_call(
    harness: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: Any
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("a call was made")

    monkeypatch.setattr(harness, "run", never)
    for flags in (["--experiment", "all"], ["--export-rating"]):
        assert harness.main([*flags, "--out-dir", str(tmp_path)]) == 2
    assert list(tmp_path.iterdir()) == []


def test_the_experiment_files_hold_no_key_prompt_or_reply(
    harness: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    assert harness.main(["--offline", "--experiment", "all", "--out-dir", str(tmp_path)]) == 0
    text = "".join(p.read_text(encoding="utf-8") for p in tmp_path.iterdir())
    assert "offline-key" not in text and "I am not sure" not in text
    for task in workload.TASKS:
        assert task.prompt not in text


def test_no_experiment_file_has_a_windows_line_ending(harness: ModuleType, tmp_path: Path) -> None:
    assert (
        harness.main(
            ["--offline", "--experiment", "all", "--export-rating", "--out-dir", str(tmp_path)]
        )
        == 0
    )
    assert all(b"\r" not in p.read_bytes() for p in tmp_path.iterdir())


def test_the_experiment_modules_use_only_the_public_api_and_no_story_refs() -> None:
    for name in ("experiments.py", "rating.py"):
        source = (EVALS / name).read_text(encoding="utf-8")
        assert "from primagate." not in source and "import primagate." not in source
        assert "Story" not in source and "8.4" not in source
        assert "\r" not in source


# --- the human-rated sample ---


def _both_arms(harness: ModuleType, workload: ModuleType) -> tuple[list[Any], list[Any]]:
    return (
        [_result(harness, t, arm="routed") for t in workload.TASKS],
        [_result(harness, t, arm="baseline", tier=None) for t in workload.TASKS],
    )


def _export(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, folder: Path, **each: Any
) -> tuple[Path, Path]:
    routed, baseline = _both_arms(harness, workload)
    when = datetime(2026, 11, 20, 10, 0, 0, tzinfo=UTC)
    made = rating.export(routed, baseline, folder, False, when, "harness-x.json", **each)
    return made.csv, made.key


def test_the_export_writes_fifty_rows_with_a_neutral_id_the_prompt_and_an_empty_rating(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    path, _ = _export(harness, rating, workload, tmp_path)
    assert path.name == "rating-20261120-100000.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 50 and list(rows[0]) == ["item", "prompt", "response", "useful"]
    assert len({r["item"] for r in rows}) == 50 and all(r["useful"] == "" for r in rows)
    assert all(r["prompt"] and r["response"] for r in rows)


def test_the_rating_file_never_shows_which_arm_or_task_made_a_response(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    path, key = _export(harness, rating, workload, tmp_path)
    text = path.read_text(encoding="utf-8").lower()
    assert not re.search(r"(routed|baseline|arm|tier|task_id)", text)
    assert not any(t.id.lower() in text for t in workload.TASKS if len(t.id) > 3)
    mapping = json.loads(key.read_text(encoding="utf-8"))
    assert key.name == "rating-20261120-100000.key.json"
    assert len(mapping["items"]) == 50
    assert {v["arm"] for v in mapping["items"].values()} <= {"routed", "baseline"}
    assert all("task_id" in v for v in mapping["items"].values())


def test_the_sample_is_stratified_by_type_and_difficulty_and_covers_both_arms(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    _, key = _export(harness, rating, workload, tmp_path)
    items = json.loads(key.read_text(encoding="utf-8"))["items"]
    by_id = {t.id: t for t in workload.TASKS}
    strata: dict[tuple[str, str], int] = {}
    for item in items.values():
        task = by_id[item["task_id"]]
        strata[(task.type, task.difficulty)] = strata.get((task.type, task.difficulty), 0) + 1
    everything = {(t.type, t.difficulty) for t in workload.TASKS}
    assert set(strata) == everything
    sizes = {
        s: 2 * sum(1 for t in workload.TASKS if (t.type, t.difficulty) == s) for s in everything
    }
    for stratum, count in strata.items():
        # Proportional to the stratum's share of the 200 responses, within one.
        assert abs(count - 50 * sizes[stratum] / 200) <= 1
    assert {v["arm"] for v in items.values()} == {"routed", "baseline"}


def test_the_same_results_and_seed_give_the_same_sample(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    _, key_a = _export(harness, rating, workload, first)
    _, key_b = _export(harness, rating, workload, second)
    assert key_a.read_bytes() == key_b.read_bytes()
    assert (first / "rating-20261120-100000.csv").read_bytes() == (
        second / "rating-20261120-100000.csv"
    ).read_bytes()


def test_a_tool_call_is_shown_as_the_tool_and_its_arguments(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    path, key = _export(harness, rating, workload, tmp_path)
    by_id = {t.id: t for t in workload.TASKS}
    items = json.loads(key.read_text(encoding="utf-8"))["items"]
    with path.open(encoding="utf-8", newline="") as handle:
        rows = {r["item"]: r for r in csv.DictReader(handle)}
    tools = [i for i, v in items.items() if by_id[v["task_id"]].type == "tool"]
    assert tools and all(by_id[items[i]["task_id"]].tool in rows[i]["response"] for i in tools)


def test_fewer_than_fifty_responses_exports_what_exists_and_says_so(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    few = [_result(harness, t) for t in workload.TASKS[:12]]
    made = rating.export(few, [], tmp_path, False, datetime.now(UTC), "r.json")
    with made.csv.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 12
    assert made.count == 12 and "only 12" in made.note.lower()


def test_a_call_that_raised_has_no_response_to_rate(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    rows = [_result(harness, t) for t in workload.TASKS[:5]]
    rows.append(harness.CallResult(task=workload.TASKS[5], arm="routed", error="OSError"))
    made = rating.export(rows, [], tmp_path, False, datetime.now(UTC), "r.json")
    assert made.count == 5


def _rated(path: Path, ratings: dict[str, str]) -> Path:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["useful"] = ratings.get(row["item"], "yes")
    # Filled in place, as a person would, so the key beside it is still found by name.
    done = path
    with done.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return done


def _score_setup(
    harness: ModuleType,
    rating: ModuleType,
    workload: ModuleType,
    folder: Path,
    **each: Any,
) -> tuple[Path, dict[str, Any]]:
    routed, baseline = _both_arms(harness, workload)
    when = datetime(2026, 11, 20, 10, 0, 0, tzinfo=UTC)
    # The results file the key points at, so the import can read the automatic check.
    results = harness.results_data(harness.Config(), routed, baseline, False, when)
    (folder / "harness-x.json").write_text(json.dumps(results), encoding="utf-8")
    made = rating.export(routed, baseline, folder, False, when, "harness-x.json")
    return made.csv, json.loads(made.key.read_text(encoding="utf-8"))


def test_scoring_reports_human_usefulness_per_arm_and_agreement_with_the_checks(
    harness: ModuleType,
    rating: ModuleType,
    workload: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, key = _score_setup(harness, rating, workload, tmp_path)
    items = key["items"]
    # Every automatic check passes, so a "no" is a disagreement. Disagree on three items.
    chosen = sorted(items)[:3]
    done = _rated(path, {i: "no" for i in chosen})
    assert harness.main(["--score-rating", str(done)]) == 0
    shown = capsys.readouterr().out
    arms = {v["arm"] for v in items.values()}
    for arm in arms:
        assert arm in shown
    assert "Agreement" in shown and "47 of 50" in shown
    for i in chosen:
        assert items[i]["task_id"] in shown
    assert "judge" not in shown.lower().replace("no judge", "")


def test_scoring_counts_each_arm_separately(
    harness: ModuleType,
    rating: ModuleType,
    workload: ModuleType,
    tmp_path: Path,
) -> None:
    path, key = _score_setup(harness, rating, workload, tmp_path)
    routed = [i for i, v in key["items"].items() if v["arm"] == "routed"]
    done = _rated(path, {i: "no" for i in routed})
    report = rating.score(done)
    assert report.per_arm["routed"].useful == 0
    assert report.per_arm["routed"].rated == len(routed)
    assert report.per_arm["baseline"].useful == report.per_arm["baseline"].rated
    assert report.agreed == 50 - len(routed)
    assert sorted(report.disagreements) == sorted(key["items"][i]["task_id"] for i in routed)


@pytest.mark.parametrize("bad", ["maybe", "", "  ", "Y", "true"])
def test_a_rating_that_is_not_yes_or_no_is_an_error_naming_the_row_and_nothing_is_scored(
    harness: ModuleType,
    rating: ModuleType,
    workload: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    bad: str,
) -> None:
    path, key = _score_setup(harness, rating, workload, tmp_path)
    item = sorted(key["items"])[7]
    done = _rated(path, {item: bad})
    code = harness.main(["--score-rating", str(done)])
    captured = capsys.readouterr()
    assert (code, item in captured.err) == (1, True)
    assert "Agreement" not in captured.out and captured.out.strip() == ""


def test_yes_and_no_are_accepted_in_any_case_and_with_spaces(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    path, key = _score_setup(harness, rating, workload, tmp_path)
    one, two = sorted(key["items"])[:2]
    report = rating.score(_rated(path, {one: " YES ", two: "No"}))
    assert sum(a.rated for a in report.per_arm.values()) == 50


def test_every_bad_row_is_named_not_just_the_first(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    path, key = _score_setup(harness, rating, workload, tmp_path)
    one, two = sorted(key["items"])[:2]
    with pytest.raises(rating.RatingError) as err:
        rating.score(_rated(path, {one: "maybe", two: ""}))
    assert one in str(err.value) and two in str(err.value)


def test_a_rating_file_with_an_unknown_or_repeated_item_is_refused(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    path, _ = _score_setup(harness, rating, workload, tmp_path)
    done = _rated(path, {})
    lines = done.read_text(encoding="utf-8").splitlines()
    done.write_text("\n".join([*lines, lines[1]]) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(rating.RatingError, match="more than once"):
        rating.score(done)
    done.write_text("\n".join([lines[0], "zzz,p,r,yes"]) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(rating.RatingError, match="zzz"):
        rating.score(done)


def test_a_missing_key_file_is_an_error_not_a_guess(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    path, _ = _score_setup(harness, rating, workload, tmp_path)
    done = _rated(path, {})
    path.with_suffix(".key.json").unlink()
    with pytest.raises(rating.RatingError, match="key"):
        rating.score(done)


def test_the_export_command_makes_the_run_then_the_files_and_scoring_needs_no_key(
    harness: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert harness.main(["--offline", "--export-rating", "--out-dir", str(tmp_path)]) == 0
    (csv_path,) = tmp_path.glob("rating-*.csv")
    (key_path,) = tmp_path.glob("rating-*.key.json")
    assert key_path.name == csv_path.name.replace(".csv", ".key.json")
    assert "50" in capsys.readouterr().out
    done = _rated(csv_path, {})
    # The key names the results file it came from, which sits in the same folder.
    assert harness.main(["--score-rating", str(done)]) == 0
    shown = capsys.readouterr().out
    assert harness.OFFLINE_BANNER in shown and "Agreement" in shown


def test_the_rating_files_hold_no_key_and_stay_in_the_ignored_folder() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "evals/results/*" in ignore
    readme = (EVALS / "README.md").read_text(encoding="utf-8")
    assert "git ignores" in readme or "ignored" in readme


def test_the_readme_documents_the_experiments_the_rating_flow_and_the_format_bump() -> None:
    readme = (EVALS / "README.md").read_text(encoding="utf-8")
    for needle in (
        "--experiment",
        "--export-rating",
        "--score-rating",
        "context_exchanges",
        "about 200",
        "about 40",
        "version 2",
        "profile_not_ready",
        "NOT RUN: Profile not ready",
    ):
        assert needle in readme, needle
    top = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "`context_exchanges`" in top
    versioning = (ROOT / "docs" / "versioning.md").read_text(encoding="utf-8")
    assert "optional Policy field" in versioning


# --- review round ---


def test_the_control_arms_analyzer_never_reads_a_profile(harness: ModuleType) -> None:
    log: list[dict[str, Any]] = []
    harness.run(harness.Config(), "k", _recording(harness, log))
    analyzer = [
        str(b["messages"][0]["content"])
        for b in log
        if "Prompt version: analyzer" in str(b["messages"][0]["content"])
    ]
    assert len(analyzer) == 100 and not any("Purpose:" in text for text in analyzer)


def test_a_retried_preparation_failure_is_not_the_end(experiments: ModuleType) -> None:
    from primagate import PreparationFailed, ProfileReady

    watcher = experiments.Watcher()
    watcher(PreparationFailed("prep-1", "fp", "prep-1", "provider_error", 1, False))
    assert not watcher.done.is_set()
    watcher(ProfileReady("prep-1", "fp", "prep-1", "profile-1"))
    assert watcher.done.is_set() and watcher.ready
    final = experiments.Watcher()
    final(PreparationFailed("prep-1", "fp", "prep-1", "malformed_output", 2, True))
    assert final.done.is_set() and not final.ready and final.reason == "malformed_output"


def test_a_warm_up_call_that_fails_ends_the_wait_at_once(
    harness: ModuleType, experiments: ModuleType
) -> None:
    def down(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down")

    run = experiments.run_profile(harness.Config(), "k", httpx2.MockTransport(down), 30.0)
    assert not run.ready and run.results == []
    assert "warm-up" in run.reason and "timeout" not in run.reason


def test_the_profile_section_states_the_warm_up_cost_and_that_preparation_is_excluded(
    experiments: ModuleType, control: tuple[list[Any], list[Any]], profile_run: Any
) -> None:
    text = experiments.profile_section(control[0], profile_run, False)
    assert "Warm-up call cost" in text and "preparation" in text.lower()
    assert "not included in the totals" in text


def test_the_point_size_of_a_task_comes_from_the_compared_set(
    harness: ModuleType, experiments: ModuleType, workload: ModuleType
) -> None:
    tasks = workload.TASKS[:4]
    control = [_result(harness, t) for t in tasks]
    arm = [_result(harness, t, arm="profile") for t in tasks]
    text = experiments.profile_section(control, experiments.ProfileRun(True, None, arm), False)
    assert "one task is 25 points" in text
    multi = [t for t in workload.TASKS if t.type == "multiturn"][:4]
    text = experiments.history_section(
        [_result(harness, t) for t in multi],
        [_result(harness, t, arm="no_history") for t in multi],
        False,
    )
    assert "4 tasks" in text and "one task is 25 points" in text


def test_an_empty_compared_set_says_it_could_not_be_compared(
    harness: ModuleType, experiments: ModuleType, workload: ModuleType
) -> None:
    run = experiments.ProfileRun(True, None, [])
    text = experiments.profile_section([_result(harness, workload.TASKS[0])], run, False)
    assert "could not be compared" in text and "0.0%" not in text
    text = experiments.history_section([], [], False)
    assert "could not be compared" in text and "0.0%" not in text


def test_a_tier_name_outside_the_known_ones_is_skipped_and_counted(
    harness: ModuleType, experiments: ModuleType, workload: ModuleType
) -> None:
    easy = next(t for t in workload.TASKS if t.difficulty == "easy")
    rows = [_result(harness, easy, tier="SMALL"), _result(harness, easy, tier="HUGE")]
    assert experiments.capable_enough(rows) == 1.0
    assert experiments.tier_agreement(rows) == 1.0
    assert experiments.unknown_tiers(rows) == 1
    multi = [t for t in workload.TASKS if t.type == "multiturn"]
    text = experiments.history_section(
        [_result(harness, t, tier="HUGE") for t in multi],
        [_result(harness, t, arm="no_history") for t in multi],
        False,
    )
    assert "unrecognised Tier" in text


def test_the_extra_call_counts_come_from_the_workload(
    experiments: ModuleType, workload: ModuleType
) -> None:
    assert experiments.extra_calls("profile") == 2 * len(workload.TASKS) == 200
    assert experiments.extra_calls("history") == 40
    assert experiments.extra_calls("all") == 240
    assert experiments.extra_calls(None) == 0


@pytest.mark.parametrize("bad", ["nan", "inf", "0", "-1"])
def test_a_profile_timeout_that_is_not_finite_and_positive_is_refused(
    harness: ModuleType, bad: str
) -> None:
    with pytest.raises(SystemExit) as stopped:
        harness.main(["--offline", "--profile-timeout", bad])
    assert stopped.value.code == 2


def test_a_short_profile_timeout_reaches_the_experiment_and_exits_one(
    harness: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: Any
) -> None:
    release = threading.Event()
    real = harness.stand_in

    def stuck(request: httpx2.Request) -> httpx2.Response:
        if "Prompt version: prep" in str(json.loads(request.content)["messages"][0]["content"]):
            release.wait(3)
        return real(request)  # type: ignore[no-any-return]

    monkeypatch.setattr(harness, "stand_in", stuck)
    flags = ["--experiment", "profile", "--profile-timeout", "0.2", "--out-dir", str(tmp_path)]
    try:
        code = harness.main(["--offline", *flags])
    finally:
        release.set()
    assert code == 1 and "timeout after 0.2s" in capsys.readouterr().out


@pytest.mark.parametrize("flag", [["--offline"], ["--experiment", "all"], ["--export-rating"]])
def test_scoring_with_run_flags_is_an_error(harness: ModuleType, flag: list[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        harness.main(["--score-rating", "x.csv", *flag])
    assert stopped.value.code == 2


def test_the_main_report_is_saved_before_the_experiments_and_survives_their_failure(
    harness: ModuleType,
    experiments: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: Any,
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("experiment broke")

    monkeypatch.setattr(experiments, "run_requested", boom)
    code = harness.main(["--offline", "--experiment", "all", "--out-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1 and "RuntimeError" in captured.err and "experiment broke" not in captured.err
    assert "# Tier Routing evaluation" in captured.out
    (results,) = tmp_path.glob("offline-harness-*.json")
    assert json.loads(results.read_text(encoding="utf-8"))["version"] == 3
    assert len(list(tmp_path.glob("offline-harness-*.md"))) == 1


def test_a_silent_arm_stops_before_any_experiment(
    harness: ModuleType,
    experiments: ModuleType,
    workload: ModuleType,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    fixed = [
        harness.CallResult(task=t, arm="baseline", error="OSError", latency=0.1)
        for t in workload.TASKS
    ]
    good = [_result(harness, t) for t in workload.TASKS]
    monkeypatch.setattr(harness, "run", lambda *a, **k: (good, fixed))

    def never(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("an experiment was run")

    monkeypatch.setattr(experiments, "run_requested", never)
    code = harness.main(["--offline", "--experiment", "all", "--out-dir", str(tmp_path)])
    assert code == 1


def test_an_experiment_arm_that_answered_nothing_exits_one(
    harness: ModuleType,
    experiments: ModuleType,
    workload: ModuleType,
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    dead = [
        harness.CallResult(task=t, arm="no_history", error="OSError")
        for t in workload.TASKS
        if t.type == "multiturn"
    ]
    monkeypatch.setattr(experiments, "run_history", lambda *a, **k: dead)
    code = harness.main(["--offline", "--experiment", "history", "--out-dir", str(tmp_path)])
    assert code == 1


def test_a_real_key_experiment_prints_the_extra_calls_and_cost_first(
    harness: ModuleType,
    experiments: ModuleType,
    workload: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: Any,
) -> None:
    def fake_run(config: Any, key: str, transport: Any) -> Any:
        return [_result(harness, t) for t in workload.TASKS], [
            _result(harness, t, arm="baseline") for t in workload.TASKS
        ]

    monkeypatch.setattr(harness, "run", fake_run)
    monkeypatch.setattr(experiments, "run_requested", lambda *a, **k: ([], {}, True))
    flags = ["--api-key", "sk-x", "--experiment", "all", "--out-dir", str(tmp_path)]
    assert harness.main(flags) == 0
    err = capsys.readouterr().err
    assert "240 extra calls" in err and "US cents" in err and "sk-x" not in err


@pytest.mark.parametrize("flags", [["--offline", "--experiment", "all"], ["--help"]])
def test_the_script_runs_as_a_script(flags: list[str], tmp_path: Path) -> None:
    command = [sys.executable, "-I", str(EVALS / "harness.py"), *flags]
    if "--help" not in flags:
        command += ["--out-dir", str(tmp_path)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=300, cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    if "--help" in flags:
        assert "--experiment" in done.stdout
    else:
        assert "## Profile experiment" in done.stdout and "## History experiment" in done.stdout


# --- rating, hardening ---


def test_cells_a_spreadsheet_would_run_as_formulas_are_neutralised(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    task = workload.TASKS[0]
    texts = ("=1+1", "+1", "-5", "@x", "\tx", "ok")
    rows = [_result(harness, task, reply=text) for text in texts]
    made = rating.export(rows, [], tmp_path, False, datetime.now(UTC), "r.json")
    with made.csv.open(encoding="utf-8", newline="") as handle:
        cells = [r["response"] for r in csv.DictReader(handle)]
    assert sorted(c for c in cells if c != "ok") == sorted("'" + t for t in texts[:-1])
    assert "ok" in cells
    assert rating.neutral("=1+1") == "'=1+1" and rating.neutral("ok") == "ok"


def test_an_apostrophe_a_spreadsheet_added_to_a_rating_is_ignored(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    path, key = _score_setup(harness, rating, workload, tmp_path)
    item = sorted(key["items"])[0]
    done = _rated(path, {item: "'yes"})
    assert sum(a.rated for a in rating.score(done).per_arm.values()) == 50


def test_a_file_that_is_not_utf8_or_is_semicolon_separated_is_a_rating_error(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    path, _ = _score_setup(harness, rating, workload, tmp_path)
    path.write_bytes(b"item,prompt,response,useful\n\xff\xfe,x,y,yes\n")
    with pytest.raises(rating.RatingError, match=path.name):
        rating.score(path)
    path.write_bytes(b"item;prompt;response;useful\nitem-001;x;y;yes\n")
    with pytest.raises(rating.RatingError, match="semicolon"):
        rating.score(path)


def test_rows_missing_from_the_file_are_listed_and_never_scored_as_complete(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    path, _ = _score_setup(harness, rating, workload, tmp_path)
    done = _rated(path, {})
    with done.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    gone = rows[1][0]
    with done.open("w", encoding="utf-8", newline="") as out:
        csv.writer(out, lineterminator="\n").writerows([rows[0], *rows[2:]])
    report = rating.score(done)
    assert report.rated == 49 and report.missing == [gone]
    text = report.render()
    assert "49 of 50" in text and gone in text


@pytest.mark.parametrize(
    "items",
    [{"item-001": "x"}, {"item-001": {"arm": 1, "task_id": "a"}}, {"item-001": {"arm": "r"}}, []],
)
def test_a_malformed_key_is_a_rating_error(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path, items: Any
) -> None:
    path, key = _score_setup(harness, rating, workload, tmp_path)
    done = _rated(path, {})
    key["items"] = items
    path.with_suffix(".key.json").write_text(json.dumps(key), encoding="utf-8")
    with pytest.raises(rating.RatingError, match="key"):
        rating.score(done)


def test_a_missing_results_file_or_an_absent_task_is_a_rating_error(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    path, _ = _score_setup(harness, rating, workload, tmp_path)
    done = _rated(path, {})
    results = tmp_path / "harness-x.json"
    data = json.loads(results.read_text(encoding="utf-8"))
    data["calls"] = data["calls"][:1]
    results.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(rating.RatingError, match="no routed call|no baseline call"):
        rating.score(done)
    results.unlink()
    with pytest.raises(rating.RatingError, match="results file"):
        rating.score(done)


def test_the_results_entry_in_the_key_is_taken_as_a_file_name_only(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    path, key = _score_setup(harness, rating, workload, tmp_path)
    done = _rated(path, {})
    key["results"] = "../elsewhere/harness-x.json"
    path.with_suffix(".key.json").write_text(json.dumps(key), encoding="utf-8")
    assert rating.score(done).rated == 50


def test_nothing_answered_is_an_error_and_leaves_no_files(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path
) -> None:
    dead = [harness.CallResult(task=t, arm="routed", error="OSError") for t in workload.TASKS[:3]]
    with pytest.raises(rating.RatingError, match="no answered"):
        rating.export(dead, [], tmp_path, False, datetime.now(UTC), "r.json")
    assert list(tmp_path.iterdir()) == []


def test_a_failed_csv_write_leaves_no_key_behind(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, tmp_path: Path, monkeypatch: Any
) -> None:
    rows = [_result(harness, t) for t in workload.TASKS[:3]]
    real = Path.open

    def refuse(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.suffix == ".csv":
            raise OSError("disk full")
        return real(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse)
    with pytest.raises(OSError):
        rating.export(rows, [], tmp_path, False, datetime.now(UTC), "r.json")
    monkeypatch.undo()
    assert list(tmp_path.iterdir()) == []


def test_a_prompt_with_an_unexpected_role_does_not_break_the_export(
    harness: ModuleType, rating: ModuleType, workload: ModuleType, monkeypatch: Any
) -> None:
    task = workload.TASKS[0]
    row = _result(harness, task)
    monkeypatch.setattr(type(task), "messages", lambda self: [{"role": "tool", "content": "x"}])
    assert "Tool: x" in rating._prompt(row)
