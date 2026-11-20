"""Isolation experiments: how much of the routing result comes from the Profile, and how much from
the conversation history the Analyzer reads.

Each experiment changes exactly one thing and compares the result with the harness's routed arm,
which already runs with an empty Profile and three earlier exchanges, so that arm is the control and
is never run twice. Nothing here is judged by a model, and nothing claims a cause: it reports which
way a number moved and by how much.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import harness
import httpx2
from workload import TASKS, TOOL_SPECS

from primagate import TIER_ROUTING_ALIAS, Event, PreparationFailed, ProfileReady

PROFILE_TIMEOUT = 60.0
# Cheapest first, so "at least as capable" is a comparison of positions.
_RANK = {"SMALL": 0, "MEDIUM": 1, "POWERFUL": 2}
_NEEDED = {"easy": "SMALL", "medium": "MEDIUM", "hard": "POWERFUL"}
_INSTRUCTIONS = (
    "You are a small assistant that answers short factual and arithmetic questions, follows "
    "multi-turn conversations, and calls the tools it is given when a request needs one. Replies "
    "are brief: only the answer, with no explanation."
)


@dataclass(frozen=True)
class ProfileRun:
    """What the real-Profile arm did: whether its Profile became READY, why not, and its calls."""

    ready: bool
    reason: str | None
    results: list[harness.CallResult]
    # The call that started the preparation. It is not in `results`, but its cost is reported.
    warmup: harness.CallResult | None = None


class Watcher(harness.Probe):
    """A Probe that also notices when preparation ends, one way or the other."""

    def __init__(self) -> None:
        super().__init__()
        self.done = threading.Event()
        self.ready = False
        self.reason: str | None = None

    def __call__(self, event: Event) -> None:
        super().__call__(event)
        if isinstance(event, ProfileReady):
            self.ready = True
            self.done.set()
        elif isinstance(event, PreparationFailed) and event.final:
            # An earlier attempt that will be retried is not the end, so only the last one counts.
            self.reason = event.reason
            self.done.set()


def profile_sources() -> dict[str, Any]:
    """What a real app would declare: its instructions and the description of each tool it uses."""
    used = sorted({name for task in TASKS for name in task.tools})
    return {
        "instructions": _INSTRUCTIONS,
        "tool_descriptions": [
            f"{name}: {TOOL_SPECS[name]['function']['description']}" for name in used
        ],
    }


def run_profile(
    config: harness.Config, key: str, transport: httpx2.BaseTransport | None, timeout: float
) -> ProfileRun:
    """Ask every task through a gate whose Profile is READY, or report that it never became so."""
    watcher = Watcher()
    gate = harness.build_gate(config, key, transport, watcher, profile_sources=profile_sources())
    try:
        # The first routed call is what starts preparing the Profile. It is served by the Interim
        # Binding, and is neither timed nor counted with the routed calls.
        warmup = harness.ask(gate, watcher, TIER_ROUTING_ALIAS, TASKS[0], "warm-up")
        if warmup.error is not None and not watcher.done.is_set():
            # The call failed before preparation could finish, so waiting out the timeout would
            # only hide the cause.
            return ProfileRun(False, f"warm-up call failed ({warmup.error})", [], warmup)
        if not watcher.done.wait(timeout):
            return ProfileRun(False, f"timeout after {timeout:g}s", [], warmup)
        if not watcher.ready:
            return ProfileRun(False, watcher.reason or "unknown", [], warmup)
        results = [harness.ask(gate, watcher, TIER_ROUTING_ALIAS, t, "profile") for t in TASKS]
    finally:
        gate.close()
    return ProfileRun(True, None, results, warmup)


def run_history(
    config: harness.Config, key: str, transport: httpx2.BaseTransport | None
) -> list[harness.CallResult]:
    """Ask the multi-turn tasks with the Analyzer shown no earlier exchange."""
    probe = harness.Probe()
    gate = harness.build_gate(config, key, transport, probe, context_exchanges=0)
    try:
        return [
            harness.ask(gate, probe, TIER_ROUTING_ALIAS, t, "no_history")
            for t in TASKS
            if t.type == "multiturn"
        ]
    finally:
        gate.close()


def extra_calls(choice: str | None) -> int:
    """About how many calls the experiments add: each task is one worker and one Analyzer call."""
    multi = sum(1 for t in TASKS if t.type == "multiturn")
    wanted = {"profile": 2 * len(TASKS), "history": 2 * multi}
    if choice == "all":
        return sum(wanted.values())
    return wanted.get(choice or "", 0)


def _graded(rows: list[harness.CallResult]) -> list[harness.CallResult]:
    return [r for r in rows if r.error is None and r.tier in _RANK]


def unknown_tiers(rows: list[harness.CallResult]) -> int:
    """Calls whose Tier name is not one this harness ranks; they are left out of the shares."""
    return sum(1 for r in rows if r.error is None and r.tier is not None and r.tier not in _RANK)


def capable_enough(rows: list[harness.CallResult]) -> float | None:
    """The share of calls sent to a Tier at least as capable as the task needed."""
    graded = _graded(rows)
    if not graded:
        return None
    enough = sum(1 for r in graded if _RANK[r.tier or ""] >= _RANK[_NEEDED[r.task.difficulty]])
    return enough / len(graded)


def tier_agreement(rows: list[harness.CallResult]) -> float | None:
    """The share of calls sent to exactly the Tier the task's difficulty calls for."""
    graded = _graded(rows)
    if not graded:
        return None
    return sum(1 for r in graded if r.tier == _NEEDED[r.task.difficulty]) / len(graded)


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _tiers(rows: list[harness.CallResult], difficulty: str) -> str:
    counts: dict[str, int] = {}
    for r in _graded(rows):
        if r.task.difficulty == difficulty and r.tier:
            counts[r.tier] = counts.get(r.tier, 0) + 1
    return ", ".join(f"{k} {counts[k]}" for k in sorted(counts, key=_RANK.__getitem__)) or "none"


def _cost_move(before: harness.Summary, after: harness.Summary) -> str:
    if before.unknown_cost or after.unknown_cost or before.total_cost <= 0:
        return "cost could not be compared: a cost is unknown or zero"
    change = 100.0 * (before.total_cost - after.total_cost) / before.total_cost
    if abs(change) < 0.05:
        return "cost did not change"
    return f"cost fell {change:.1f}%" if change > 0 else f"cost rose {-change:.1f}%"


def _points_move(before: float, after: float) -> str:
    gap = after - before
    if abs(gap) < 0.05:
        return "did not change"
    return f"{gap:.1f} points higher" if gap > 0 else f"{-gap:.1f} points lower"


def _share_move(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "could not be compared"
    return _points_move(100 * before, 100 * after)


def _head(title: str, offline: bool) -> list[str]:
    lines = [f"## {title}", ""]
    if offline:
        lines += [f"**{harness.OFFLINE_BANNER}**", ""]
    return lines


def profile_section(control: list[harness.CallResult], run: ProfileRun, offline: bool) -> str:
    """The Profile experiment as a report section, or the plain statement that it did not run."""
    lines = _head("Profile experiment", offline)
    if not run.ready:
        lines += [
            f"**NOT RUN: Profile not ready ({run.reason})**",
            "",
            "The real Profile did not become READY, so the arm was not run and nothing is claimed "
            "about what a Profile changes.",
            "",
        ]
        return "\n".join(lines)
    flagged = [r for r in run.results if r.not_ready]
    skipped = {r.task.id for r in flagged}
    kept_control, kept_arm = harness.common(
        [r for r in control if r.task.id not in skipped],
        [r for r in run.results if r.task.id not in skipped],
    )
    a, b = harness.summarize(kept_control), harness.summarize(kept_arm)
    enough_a, enough_b = capable_enough(kept_control), capable_enough(kept_arm)
    if not a.calls:
        lines += [
            "The two arms have no task in common that both answered, so they could not be "
            "compared and nothing is claimed.",
            "",
        ]
        return "\n".join(lines)
    lines += [
        "Both arms use the same Analyzer, Tiers, Policy and tasks. The control is the routed arm, "
        "which has an empty Profile. This arm declares `profile_sources` (the harness's "
        "instructions and the descriptions of the tools its workload uses) and starts only once "
        "the Profile is READY. The wait is not timed or counted.",
        "",
        f"Compared set: {a.calls} task(s) answered by both arms.",
        "",
    ]
    if flagged:
        lines += [
            f"{len(flagged)} call(s) served while the Profile was not ready were flagged and left "
            "out of the comparison, with their tasks.",
            "",
        ]
    lines += [
        "| | Empty Profile (control) | Real Profile |",
        "|---|---|---|",
        f"| Usefulness | {a.usefulness:.1f}% | {b.usefulness:.1f}% |",
        f"| **Total cost** | **{harness._money(a.total_cost)}** "
        f"| **{harness._money(b.total_cost)}** |",
        f"| Analyzer overhead, cost | {harness._money(a.analyzer_cost)} "
        f"| {harness._money(b.analyzer_cost)} |",
        f"| Analyzer overhead, added latency | {harness._seconds(a.analyzer_latency_avg)} "
        f"| {harness._seconds(b.analyzer_latency_avg)} |",
        f"| Sent to a Tier at least as capable as the task needed | {_share(enough_a)} "
        f"| {_share(enough_b)} |",
        "",
        "Tier chosen by difficulty:",
        "",
        "| Difficulty | Empty Profile | Real Profile |",
        "|---|---|---|",
        *[
            f"| {d} | {_tiers(kept_control, d)} | {_tiers(kept_arm, d)} |"
            for d in harness.DIFFICULTIES
        ],
        "",
        "Movement from the empty Profile to the real one:",
        "",
        f"- Total cost: {_cost_move(a, b)}",
        f"- Usefulness: {_points_move(a.usefulness, b.usefulness)}",
        f"- Analyzer cost: {_cost_move_plain(a.analyzer_cost, b.analyzer_cost)}",
        f"- Share sent to a Tier at least as capable as needed: {_share_move(enough_a, enough_b)}",
        "",
        "This reports how two runs of one workload differ. It does not say why, and one run has "
        f"no variance estimate: one task is {100 / a.calls:g} points of Usefulness.",
        "",
        *_warmup_lines(run.warmup),
        *_unknown_lines(kept_control, kept_arm),
    ]
    return "\n".join(lines)


def _warmup_lines(warmup: harness.CallResult | None) -> list[str]:
    cost = "n/a" if warmup is None else harness._money(warmup.total_cost)
    return [
        f"Warm-up call cost: {cost}. It and the Profile preparation, which the SDK's events do "
        "not price, are not included in the totals above, so the real-Profile arm's cost is a "
        "lower bound.",
        "",
    ]


def _unknown_lines(*arms: list[harness.CallResult]) -> list[str]:
    count = sum(unknown_tiers(rows) for rows in arms)
    if not count:
        return []
    return [f"{count} call(s) named an unrecognised Tier and are left out of the Tier shares.", ""]


def _cost_move_plain(before: float, after: float) -> str:
    if before <= 0:
        return "could not be compared"
    change = 100.0 * (before - after) / before
    if abs(change) < 0.05:
        return "did not change"
    return f"fell {change:.1f}%" if change > 0 else f"rose {-change:.1f}%"


def history_section(
    control: list[harness.CallResult], arm: list[harness.CallResult], offline: bool
) -> str:
    """The history experiment as a report section: multi-turn tasks only."""
    multi = [r for r in control if r.task.type == "multiturn"]
    kept_control, kept_arm = harness.common(multi, arm)
    a, b = harness.summarize(kept_control), harness.summarize(kept_arm)
    agree_a, agree_b = tier_agreement(kept_control), tier_agreement(kept_arm)
    lines = _head("History experiment", offline)
    if not a.calls:
        lines += [
            "No multi-turn task was answered by both arms, so they could not be compared and "
            "nothing is claimed.",
            "",
        ]
        return "\n".join(lines)
    lines += [
        "Only the multi-turn tasks are compared, since only their routing can depend on history. "
        "Both arms use the same worker setup, and the worker always receives the whole "
        "conversation; only what the Analyzer reads differs. The control shows the Analyzer up to "
        "3 earlier exchanges. This arm sets `context_exchanges` to 0.",
        "",
        f"Compared set: {a.calls} task(s) answered by both arms.",
        "",
        "| | 3 earlier exchanges (control) | No history |",
        "|---|---|---|",
        f"| Usefulness | {a.usefulness:.1f}% | {b.usefulness:.1f}% |",
        f"| **Total cost** | **{harness._money(a.total_cost)}** "
        f"| **{harness._money(b.total_cost)}** |",
        f"| Tier agreement (sent to exactly the Tier the task needed) | {_share(agree_a)} "
        f"| {_share(agree_b)} |",
        "",
        "Movement from 3 earlier exchanges to none:",
        "",
        f"- Total cost: {_cost_move(a, b)}",
        f"- Usefulness: {_points_move(a.usefulness, b.usefulness)}",
        f"- Tier agreement: {_share_move(agree_a, agree_b)}",
        "",
        f"Small sample: there are {a.calls} tasks, so one task is {100 / a.calls:g} points. This "
        "reports how two runs differ and does not say why.",
        "",
        *_unknown_lines(kept_control, kept_arm),
    ]
    return "\n".join(lines)


def experiments_data(
    profile: ProfileRun | None, history: list[harness.CallResult] | None
) -> dict[str, Any]:
    """The experiments as plain facts for the results file, in the same call format as the rest."""
    out: dict[str, Any] = {}
    if profile is not None:
        out["profile"] = {
            "status": "ran" if profile.ready else "not_run",
            "reason": profile.reason,
            "calls": [harness.call_record(r) for r in profile.results],
        }
    if history is not None:
        out["history"] = {
            "status": "ran",
            "reason": None,
            "calls": [harness.call_record(r) for r in history],
        }
    return out


def _any_answered(rows: list[harness.CallResult]) -> bool:
    return any(r.error is None for r in rows)


def run_requested(
    choice: str | None,
    config: harness.Config,
    control: list[harness.CallResult],
    key: str,
    transport: httpx2.BaseTransport | None,
    timeout: float,
    offline: bool,
) -> tuple[list[str], dict[str, Any], bool]:
    """Run what `choice` asks for against the control arm: sections, results, all ran."""
    if choice is None:
        return [], {}, True
    profile = run_profile(config, key, transport, timeout) if choice in ("profile", "all") else None
    history = run_history(config, key, transport) if choice in ("history", "all") else None
    sections = []
    if profile is not None:
        sections.append(profile_section(control, profile, offline))
    if history is not None:
        sections.append(history_section(control, history, offline))
    # An arm that answered nothing is a failed experiment, not an empty finding.
    answered = (profile is None or (profile.ready and _any_answered(profile.results))) and (
        history is None or _any_answered(history)
    )
    return sections, experiments_data(profile, history), answered
