"""Does Tier Routing save money without losing quality? The evaluation harness.

Runs the fixed workload twice, through the Tier Routing alias and straight to one premium model,
checks every reply or tool call by rule, and writes a report plus a machine-readable results file.
Run it with `uv run python evals/harness.py`, which needs OPENAI_API_KEY, or with `--offline` to
try the harness against a scripted stand-in.

Only the public `primagate` API is used, so this measures what a developer would get.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
from workload import TASKS, Task

from primagate import TIER_ROUTING_ALIAS, Event, PrimaGate, RoutingDecision, RoutingDegraded

# SM-4: cost down by at least this many percent, with Usefulness down by at most this many points.
COST_TARGET = 30.0
QUALITY_TOLERANCE = 2.0
# Below this share of answered calls in either arm, a pass rate says too little to call a result.
MIN_ANSWERED_SHARE = 0.9
# A confidence bucket with fewer calls than this is shown but not used for the trend.
MIN_BUCKET = 5
# Rounding slack so exactly 30% or exactly 2 points is not lost to floating point.
EPSILON = 1e-9
# A conversation with more earlier exchanges than this is counted as having its context omitted.
MAX_EXCHANGES = 3
# Fewer routed calls with a confidence than this say nothing about calibration.
MIN_CALIBRATION = 20
# A short answer is enough to check by rule and keeps a full run cheap.
MAX_TOKENS = 64
RESULTS_VERSION = 1

OFFLINE_BANNER = "OFFLINE STAND-IN: not evidence"
TYPES = ("oneshot", "tool", "multiturn")
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass(frozen=True)
class Config:
    small: str = "gpt-4o-mini"
    medium: str = "gpt-4.1-mini"
    powerful: str = "gpt-4o"
    analyzer: str = "gpt-4o-mini"
    baseline: str = "gpt-4o"
    threshold: float = 0.7


@dataclass(frozen=True)
class CallResult:
    """What one call did. It holds the reply only so the check can run, and it is never written."""

    task: Task
    arm: str
    reply: str | None = None
    # The first tool call as (name, JSON arguments).
    tool_call: tuple[str, str] | None = None
    # The class name of what was raised, never its text, which could echo a prompt.
    error: str | None = None
    # Seconds to the end of the stream and to its first chunk.
    latency: float = 0.0
    first_latency: float | None = None
    worker_cost: float | None = None
    analyzer_cost: float | None = None
    analyzer_latency: float | None = None
    tier: str | None = None
    # Why the call went to the Interim Binding instead of being routed, if it did.
    degraded_reason: str | None = None
    confidence: float | None = None
    # Attempts in the chain, skipped candidates not counted; more than one means a fallback.
    attempts: int = 1
    # Why the stream stopped; "length" means the reply was cut at max_tokens.
    finish_reason: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.task.check(self.reply, self.tool_call)

    @property
    def degraded(self) -> bool:
        return self.degraded_reason is not None

    @property
    def fallback(self) -> bool:
        return self.attempts > 1

    @property
    def long_history(self) -> bool:
        # A proxy only: the routed arm's context budget is what really trims, and it is set in
        # tokens. Only the routed arm has one; the fixed model sends everything.
        return self.arm == "routed" and len(self.task.history) > MAX_EXCHANGES

    @property
    def over_budget(self) -> bool:
        # The measured signal: routing gave up because the latest turn exceeded the budget.
        return self.arm == "routed" and self.degraded_reason == "over_budget"

    @property
    def omitted(self) -> bool:
        return self.long_history or self.over_budget

    @property
    def cut_off(self) -> bool:
        return self.error is None and self.finish_reason == "length"

    @property
    def cost_unknown(self) -> bool:
        return self.worker_cost is None or (
            # An Analyzer that ran but has no price would otherwise count as free.
            self.tier is not None and not self.degraded and self.analyzer_cost is None
        )

    @property
    def total_cost(self) -> float:
        return (self.worker_cost or 0.0) + (self.analyzer_cost or 0.0)


@dataclass(frozen=True)
class Slice:
    """The calls of one difficulty or type: how many were asked, passed and cost."""

    asked: int
    passed: int
    cost: float
    unknown: int

    @property
    def usefulness(self) -> float:
        return 100.0 * self.passed / self.asked if self.asked else 0.0


@dataclass(frozen=True)
class Summary:
    calls: int
    completed: int
    passes: int
    usefulness: float
    worker_cost: float
    analyzer_cost: float
    unknown_cost: tuple[str, ...]
    first_avg: float
    first_p95: float
    latency_avg: float
    latency_p95: float
    analyzer_latency_avg: float | None
    tiers: dict[str, int]
    degraded: int
    errors: int
    fallbacks: int
    omissions: int
    long_history: int
    over_budget: int
    cut_off: int
    by_difficulty: dict[str, Slice]
    by_type: dict[str, Slice]
    failed: tuple[str, ...]

    @property
    def total_cost(self) -> float:
        return self.worker_cost + self.analyzer_cost

    @property
    def cost_per_pass(self) -> float | None:
        return self.total_cost / self.passes if self.passes else None

    @property
    def fallback_rate(self) -> float:
        return self.fallbacks / self.completed if self.completed else 0.0


def percentile(values: list[float], share: float) -> float:
    """The nearest-rank percentile; 0 for no values."""
    ordered = sorted(values)
    return ordered[max(math.ceil(share * len(ordered)) - 1, 0)] if ordered else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _slices(results: list[CallResult], key: Any) -> dict[str, Slice]:
    groups: dict[str, list[CallResult]] = {}
    for r in results:
        groups.setdefault(key(r.task), []).append(r)
    return {
        name: Slice(
            asked=len(rows),
            passed=sum(1 for r in rows if r.passed),
            cost=sum(r.total_cost for r in rows if r.error is None),
            unknown=sum(1 for r in rows if r.error is None and r.cost_unknown),
        )
        for name, rows in groups.items()
    }


def summarize(results: list[CallResult]) -> Summary:
    completed = [r for r in results if r.error is None]
    passes = sum(1 for r in results if r.passed)
    # A degraded call ran no Analyzer, so any decision time it has is not Analyzer latency.
    analyzer_latencies = [
        r.analyzer_latency for r in completed if r.analyzer_latency is not None and not r.degraded
    ]
    tiers: dict[str, int] = {}
    for r in completed:
        if r.tier:
            tiers[r.tier] = tiers.get(r.tier, 0) + 1
    firsts = [r.first_latency for r in completed if r.first_latency is not None]
    latencies = [r.latency for r in completed]
    return Summary(
        calls=len(results),
        completed=len(completed),
        passes=passes,
        usefulness=100.0 * passes / len(results) if results else 0.0,
        worker_cost=sum(r.worker_cost for r in completed if r.worker_cost is not None),
        analyzer_cost=sum(r.analyzer_cost for r in completed if r.analyzer_cost is not None),
        # Listed by id and left out of the sums, so a missing price is never read as free.
        unknown_cost=tuple(r.task.id for r in completed if r.cost_unknown),
        first_avg=_mean(firsts),
        first_p95=percentile(firsts, 0.95),
        latency_avg=_mean(latencies),
        latency_p95=percentile(latencies, 0.95),
        analyzer_latency_avg=_mean(analyzer_latencies) if analyzer_latencies else None,
        tiers=tiers,
        degraded=sum(1 for r in completed if r.degraded),
        errors=len(results) - len(completed),
        fallbacks=sum(1 for r in completed if r.fallback),
        omissions=sum(1 for r in completed if r.omitted),
        long_history=sum(1 for r in completed if r.long_history),
        over_budget=sum(1 for r in completed if r.over_budget),
        cut_off=sum(1 for r in completed if r.cut_off),
        by_difficulty=_slices(results, lambda t: t.difficulty),
        by_type=_slices(results, lambda t: t.type),
        failed=tuple(r.task.id for r in results if not r.passed),
    )


def _saving(routed_cost: float, baseline_cost: float) -> float:
    return 100.0 * (baseline_cost - routed_cost) / baseline_cost


def common(
    routed: list[CallResult], baseline: list[CallResult]
) -> tuple[list[CallResult], list[CallResult]]:
    """Both arms restricted to the tasks each answered; a task one arm lost leaves both."""
    both = {r.task.id for r in routed if r.error is None} & {
        r.task.id for r in baseline if r.error is None
    }
    return (
        [r for r in routed if r.task.id in both],
        [r for r in baseline if r.task.id in both],
    )


def _verdict_of(routed: Summary, baseline: Summary) -> str:
    """The SM-4 line for two summaries over the same tasks."""
    if baseline.total_cost <= 0 or baseline.unknown_cost or routed.unknown_cost:
        return "INCONCLUSIVE: a cost is unknown or zero, so the saving cannot be computed"
    saving = _saving(routed.total_cost, baseline.total_cost)
    drop = baseline.usefulness - routed.usefulness
    met = saving >= COST_TARGET - EPSILON and drop <= QUALITY_TOLERANCE + EPSILON
    cost = f"cost fell {saving:.1f}%" if saving >= 0 else f"cost rose {-saving:.1f}%"
    use = (
        f"Usefulness {drop:.1f} points lower"
        if drop >= 0
        else f"Usefulness {-drop:.1f} points higher"
    )
    target = f"at least {COST_TARGET:.0f}% and at most {QUALITY_TOLERANCE:.0f} points"
    line = f"SM-4 {'MET' if met else 'NOT MET'}: {cost}, {use} (target: {target})"
    if routed.degraded:
        # Degraded calls went to the Interim Binding, so the routed arm was not all routing.
        line += f"; {routed.degraded} routed call(s) fell back to the Interim Binding"
    return line


def verdict(routed: list[CallResult], baseline: list[CallResult], offline: bool) -> str:
    """The one line that answers SM-4, over the tasks both arms answered."""
    if offline:
        return "INCONCLUSIVE: this run used the offline stand-in, which is not evidence"
    for name, rows in (("routed", routed), ("fixed", baseline)):
        answered = _answered(rows)
        if not rows or answered < MIN_ANSWERED_SHARE * len(rows) - EPSILON:
            return (
                f"INCONCLUSIVE: the {name} arm answered {answered} of {len(rows)} calls, under "
                f"{100 * MIN_ANSWERED_SHARE:.0f}%"
            )
    shared_routed, shared_baseline = common(routed, baseline)
    return _verdict_of(summarize(shared_routed), summarize(shared_baseline))


@dataclass(frozen=True)
class TypeRow:
    type: str
    # None when a cost in either arm is unknown or the baseline cost is zero.
    saving: float | None
    gap: float

    @property
    def saving_holds(self) -> bool:
        return self.saving is not None and self.saving >= COST_TARGET - EPSILON

    @property
    def gap_holds(self) -> bool:
        return self.gap <= QUALITY_TOLERANCE + EPSILON

    @property
    def holds(self) -> bool:
        return self.saving_holds and self.gap_holds


def type_rows(routed: list[CallResult], baseline: list[CallResult]) -> list[TypeRow]:
    """The saving and the Usefulness gap of each workload type, so a claim stays within its type."""
    routed, baseline = common(routed, baseline)
    mine, theirs = summarize(routed).by_type, summarize(baseline).by_type
    rows = []
    for name in TYPES:
        if name not in mine or name not in theirs:
            continue
        a, b = mine[name], theirs[name]
        known = not (a.unknown or b.unknown) and b.cost > 0
        rows.append(
            TypeRow(name, _saving(a.cost, b.cost) if known else None, b.usefulness - a.usefulness)
        )
    return rows


@dataclass(frozen=True)
class Bucket:
    label: str
    low: float
    high: float
    calls: int
    passes: int

    @property
    def rate(self) -> float | None:
        return self.passes / self.calls if self.calls else None

    @property
    def thin(self) -> bool:
        return 0 < self.calls < MIN_BUCKET


_EDGES = ((0.0, 0.5, "below 0.5"), (0.5, 0.7, "0.5 to 0.7"), (0.7, 0.9, "0.7 to 0.9"))


def calibration(routed: list[CallResult], threshold: float | None = None) -> list[Bucket]:
    """Routed calls by the Analyzer's confidence, with how often the chosen Tier was enough.

    A Tier is enough when the routed call passed. Left out: calls that raised (they say nothing
    about the Tier, only about the provider) and calls the verdict did not choose the Tier for,
    which degraded to the Interim Binding or, given a `threshold`, took a low-confidence downgrade.
    """
    rows = [
        r
        for r in routed
        if r.error is None
        and r.confidence is not None
        and not r.degraded
        and (threshold is None or r.confidence >= threshold)
    ]
    buckets = []
    for low, high, label in (*_EDGES, (0.9, 1.0, "0.9 and above")):
        # The top bucket includes 1.0; every other bucket holds its lower edge and not its upper.
        inside = [
            r
            for r in rows
            if low <= (r.confidence or 0.0) < high
            or (high == 1.0 and (r.confidence or 0.0) >= high)
        ]
        buckets.append(Bucket(label, low, high, len(inside), sum(1 for r in inside if r.passed)))
    return buckets


def calibration_statement(buckets: list[Bucket]) -> str:
    """Whether a higher confidence went with a higher pass rate in these data, plainly."""
    total = sum(b.calls for b in buckets)
    # A bucket of a few calls swings on one task, so it is shown but never used for the trend.
    used = [b.passes / b.calls for b in buckets if b.calls >= MIN_BUCKET]
    if total < MIN_CALIBRATION or len(used) < 2:
        return (
            f"Too few routed calls with a confidence to say ({total}, with {len(used)} bucket(s) "
            f"of at least {MIN_BUCKET}); calibration is not assumed."
        )
    if all(a < b for a, b in zip(used, used[1:], strict=False)):
        return (
            "Higher confidence went with a higher pass rate in these data. That is one run of "
            "one workload, so treat it as a check and not a guarantee."
        )
    return (
        "Higher confidence did not go with a higher pass rate in these data (a tie is not a "
        "rise), so the confidence cannot be trusted as calibrated here."
    )


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def _rate(level: Slice | None) -> str:
    return "n/a" if not level else f"{level.passed}/{level.asked}"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1000:.0f} ms"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}%"


def _points(gap: float) -> str:
    return f"{gap:.1f} points lower" if gap >= 0 else f"{-gap:.1f} points higher"


def _sm4_section(
    routed_sum: Summary, baseline_sum: Summary, rows: list[TypeRow], conclusive: bool
) -> list[str]:
    lines = [
        "## SM-4 by workload type",
        "",
        "| Type | Cost saving | Usefulness gap | Saving of 30% | Gap within 2 points | Holds |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.type} | {_percent(row.saving)} | {_points(row.gap)} "
            f"| {'yes' if row.saving_holds else 'no'} | {'yes' if row.gap_holds else 'no'} "
            f"| {'**yes**' if row.holds else 'no'} |"
        )
    lines.append("")
    comparable = (
        conclusive
        and baseline_sum.total_cost > 0
        and not (routed_sum.unknown_cost or baseline_sum.unknown_cost)
    )
    holding = [r.type for r in rows if r.holds]
    if not comparable:
        lines.append("No saving is claimed: the overall comparison is inconclusive.")
    else:
        saving = _saving(routed_sum.total_cost, baseline_sum.total_cost)
        where = f"holds for: {', '.join(holding)}" if holding else "holds for no workload type"
        lines.append(
            f"The measured overall cost change is {'a fall' if saving >= 0 else 'a rise'} of "
            f"{abs(saving):.1f}%. The target (cost down at least "
            f"{COST_TARGET:.0f}% with Usefulness at most {QUALITY_TOLERANCE:.0f} points lower) "
            f"{where}. No claim is made beyond these types and this workload."
        )
    return [*lines, ""]


def render_report(
    config: Config,
    routed: list[CallResult],
    baseline: list[CallResult],
    offline: bool,
    when: datetime,
) -> str:
    fa, fb = summarize(routed), summarize(baseline)
    shared_routed, shared_baseline = common(routed, baseline)
    # Everything below the counts is over the tasks both arms answered, so the arms are alike.
    a, b = summarize(shared_routed), summarize(shared_baseline)
    lines = [f"# Tier Routing evaluation {when:%Y-%m-%d %H:%M} UTC", ""]
    if offline:
        lines += [f"**{OFFLINE_BANNER}**", ""]
    lines += [
        f"**{verdict(routed, baseline, offline)}**",
        "",
        "Usefulness is the share of tasks whose reply or tool call passes an objective check. No "
        "model judged any answer.",
        "",
        f"Compared set: {a.calls} task(s) answered by both arms. Calls that raised: routed "
        f"{fa.errors}, fixed {fb.errors}. A raised call may still have been billed, so every cost "
        "here is a lower bound.",
        "",
        "## Setup",
        "",
        f"- Tiers: SMALL `{config.small}`, MEDIUM `{config.medium}`, POWERFUL `{config.powerful}`",
        f"- Analyzer: `{config.analyzer}`, confidence threshold {config.threshold}, "
        f"interim binding MEDIUM, context budget 2000 tokens",
        f"- Baseline (fixed premium model): `{config.baseline}`",
        f"- Workload: {len(TASKS)} tasks, one independent streamed call each per arm, caching off",
        "",
        "## Results",
        "",
        "| | Tier Routing | Fixed model |",
        "|---|---|---|",
        f"| Calls (answered) | {fa.calls} ({fa.completed}) | {fb.calls} ({fb.completed}) |",
        f"| Passed | {a.passes} | {b.passes} |",
        f"| Usefulness | {a.usefulness:.1f}% | {b.usefulness:.1f}% |",
        f"| Worker cost | {_money(a.worker_cost)} | {_money(b.worker_cost)} |",
        f"| Analyzer cost | {_money(a.analyzer_cost)} | - |",
        f"| **Total cost** | **{_money(a.total_cost)}** | **{_money(b.total_cost)}** |",
        f"| Cost per successful interaction | {_money(a.cost_per_pass)} "
        f"| {_money(b.cost_per_pass)} |",
        f"| First response, average | {_seconds(a.first_avg)} | {_seconds(b.first_avg)} |",
        f"| First response, 95th percentile (indicative) | {_seconds(a.first_p95)} "
        f"| {_seconds(b.first_p95)} |",
        f"| Completion, average | {_seconds(a.latency_avg)} | {_seconds(b.latency_avg)} |",
        f"| Completion, 95th percentile (indicative) | {_seconds(a.latency_p95)} "
        f"| {_seconds(b.latency_p95)} |",
        f"| Fallback rate | {100 * a.fallback_rate:.1f}% ({a.fallbacks}) "
        f"| {100 * b.fallback_rate:.1f}% ({b.fallbacks}) |",
        f"| Context omissions (proxy) | {a.omissions}: {a.long_history} with more than "
        f"{MAX_EXCHANGES} earlier exchanges, {a.over_budget} over budget | - |",
        f"| Interim Binding fallbacks | {a.degraded} | - |",
        "",
        "## Analyzer overhead (Tier Routing only)",
        "",
        f"- Cost: {_money(a.analyzer_cost)} of {_money(a.total_cost)} total",
        f"- Added latency, average: {_seconds(a.analyzer_latency_avg)} per call",
        f"- Calls degraded to the Interim Binding: {a.degraded}",
        "- Calls per Tier: "
        + (
            ", ".join(
                f"{k} {v} ({100.0 * v / a.completed:.0f}%)" for k, v in sorted(a.tiers.items())
            )
            or "none"
        ),
        "",
        "## Usefulness by difficulty",
        "",
        "| Difficulty | Tier Routing | Fixed model |",
        "|---|---|---|",
        *[
            f"| {name} | {_rate(a.by_difficulty.get(name))} | {_rate(b.by_difficulty.get(name))} |"
            for name in DIFFICULTIES
        ],
        "",
        "## Usefulness by type",
        "",
        "| Type | Tier Routing | Fixed model |",
        "|---|---|---|",
        *[
            f"| {name} | {_rate(a.by_type.get(name))} | {_rate(b.by_type.get(name))} |"
            for name in TYPES
        ],
        "",
        *_sm4_section(
            a,
            b,
            type_rows(routed, baseline),
            not verdict(routed, baseline, offline).startswith("INCONC"),
        ),
        "## Calibration",
        "",
        "Routed calls by the Analyzer's confidence, with how often the chosen Tier was enough "
        "(the routed call passed). Left out: calls that raised, calls with no verdict, calls that "
        f"degraded to the Interim Binding, and calls below the confidence threshold "
        f"({config.threshold}), whose Tier the verdict did not choose. Confidence is not "
        "stratified by difficulty, so a rising pass rate may only reflect easy tasks getting "
        "high confidence.",
        "",
        "| Confidence | Calls | Passed | Pass rate |",
        "|---|---|---|---|",
    ]
    buckets = calibration(routed, config.threshold)
    for bucket in buckets:
        rate = None if bucket.rate is None else 100 * bucket.rate
        mark = " (too few)" if bucket.thin else ""
        lines.append(
            f"| {bucket.label} | {bucket.calls} | {bucket.passes} | {_percent(rate)}{mark} |"
        )
    lines += ["", calibration_statement(buckets), ""]
    problems = [
        f"- Tier Routing: {fa.errors} call(s) raised" if fa.errors else "",
        f"- Fixed model: {fb.errors} call(s) raised" if fb.errors else "",
        f"- Failed in the routed arm: {', '.join(fa.failed)}" if fa.failed else "",
        f"- Failed in the fixed arm: {', '.join(fb.failed)}" if fb.failed else "",
        f"- Cut at max_tokens, counted as failed though not a wrong answer: routed "
        f"{fa.cut_off}, fixed {fb.cut_off}"
        if fa.cut_off or fb.cut_off
        else "",
        f"- Cost unknown, left out of the sums (routed): {', '.join(a.unknown_cost)}"
        if a.unknown_cost
        else "",
        f"- Cost unknown, left out of the sums (fixed): {', '.join(b.unknown_cost)}"
        if b.unknown_cost
        else "",
    ]
    problems = [p for p in problems if p]
    if problems:
        lines += ["## Notes", "", *problems, ""]
    return "\n".join(lines)


def results_data(
    config: Config,
    routed: list[CallResult],
    baseline: list[CallResult],
    offline: bool,
    when: datetime,
) -> dict[str, Any]:
    """Every call's outcome as plain facts, for the stories that build on this run.

    It holds no key and no prompt or reply text: only ids, numbers, names of tiers and errors.
    """
    calls = []
    for r in (*routed, *baseline):
        calls.append(
            {
                "task_id": r.task.id,
                "type": r.task.type,
                "difficulty": r.task.difficulty,
                "arm": r.arm,
                "passed": r.passed,
                "worker_cost": r.worker_cost,
                "analyzer_cost": r.analyzer_cost,
                "first_latency": r.first_latency,
                "latency": r.latency,
                "analyzer_latency": r.analyzer_latency,
                "tier": r.tier,
                "confidence": r.confidence,
                "degraded_reason": r.degraded_reason,
                "attempts": r.attempts,
                "fallback": r.fallback,
                "omitted": r.omitted,
                "over_budget": r.over_budget,
                "finish_reason": r.finish_reason,
                "error": r.error,
            }
        )
    return {
        "version": RESULTS_VERSION,
        "created": when.isoformat(),
        "offline": offline,
        "config": {
            "small": config.small,
            "medium": config.medium,
            "powerful": config.powerful,
            "analyzer": config.analyzer,
            "baseline": config.baseline,
            "threshold": config.threshold,
        },
        "calls": calls,
    }


class Probe:
    """Watches the events of one call, for the facts the response does not carry."""

    def __init__(self) -> None:
        self.decided_at: datetime | None = None
        self.degraded_reason: str | None = None
        self.confidence: float | None = None

    def reset(self) -> None:
        self.decided_at = None
        self.degraded_reason = None
        self.confidence = None

    def __call__(self, event: Event) -> None:
        if isinstance(event, RoutingDecision):
            if event.timestamp:
                # The decision is announced right after the Analyzer answers, so the time to it is
                # the time the Analyzer added. A bad stamp must never raise inside a handler.
                try:
                    self.decided_at = datetime.fromisoformat(event.timestamp)
                except ValueError:
                    self.decided_at = None
            self.confidence = event.confidence
        elif isinstance(event, RoutingDegraded):
            self.degraded_reason = event.reason


def _added(decided: datetime | None, started: datetime) -> float | None:
    """Seconds from the call starting to the routing decision, or None if it cannot be told."""
    if decided is None:
        return None
    try:
        # Never negative: a coarse or mismatched clock must not skew the average.
        return max(0.0, (decided - started).total_seconds())
    except TypeError:
        return None


def ask(gate: Any, probe: Probe, model: str, task: Task, arm: str) -> CallResult:
    """One streamed call: when the first chunk and the last arrived, what it said, what it cost."""
    probe.reset()
    started = datetime.now(UTC)
    clock = time.perf_counter()
    params: dict[str, Any] = {
        "model": model,
        "messages": task.messages(),
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "cache": False,
        "stream": True,
    }
    specs = task.tool_specs()
    if specs:
        params["tools"] = specs
    first: float | None = None
    text: list[str] = []
    calls: dict[int, list[str]] = {}
    meta = None
    finish: str | None = None
    try:
        for chunk in gate.completion(**params):
            if first is None:
                first = time.perf_counter() - clock
            if chunk.metadata is not None:
                meta = chunk.metadata
            for choice in chunk.choices:
                finish = choice.finish_reason or finish
                if choice.delta.content:
                    text.append(choice.delta.content)
                for part in choice.delta.tool_calls or []:
                    # [name, arguments], filled in as the deltas arrive. The name comes whole in
                    # the first delta that has one, and a repeat is never joined onto it.
                    slot = calls.setdefault(part.index, ["", ""])
                    if part.function is not None:
                        slot[0] = slot[0] or part.function.name or ""
                        slot[1] += part.function.arguments or ""
    except Exception as err:
        return CallResult(task, arm, error=type(err).__name__, latency=time.perf_counter() - clock)
    latency = time.perf_counter() - clock
    analyzer = meta.analyzer if meta else None
    # Parallel tool calls are scored on the first by index; the others are ignored.
    first_call = calls[min(calls)] if calls else None
    return CallResult(
        task=task,
        arm=arm,
        reply="".join(text) or None,
        tool_call=(first_call[0], first_call[1]) if first_call else None,
        latency=latency,
        first_latency=first,
        worker_cost=meta.cost.total if meta and meta.cost else None,
        analyzer_cost=analyzer.cost.total if analyzer and analyzer.cost else None,
        analyzer_latency=_added(probe.decided_at, started),
        tier=meta.tier if meta else None,
        degraded_reason=probe.degraded_reason,
        confidence=probe.confidence,
        attempts=sum(1 for a in meta.attempt_chain if a.skipped is None) if meta else 1,
        finish_reason=finish,
    )


_DIFFICULTY_TIER = {"easy": "SMALL", "medium": "MEDIUM", "hard": "POWERFUL"}
# What each model can answer in the stand-in, so the offline run has something to measure.
_SKILL = {
    "gpt-4o-mini": {"easy"},
    "gpt-4.1-mini": {"easy", "medium"},
    "gpt-4o": {"easy", "medium", "hard"},
}
# The stand-in Analyzer's confidence cycles over these, so every bucket has calls to count.
_CONFIDENCE = (0.95, 0.9, 0.8, 0.6, 0.4)


def _find(text: str) -> tuple[int, Task] | None:
    """The task whose final prompt ends `text`, with its position in the workload."""
    best: tuple[int, int, int] | None = None
    for index, task in enumerate(TASKS):
        at = text.rfind(task.prompt)
        if at >= 0 and (best is None or (at, len(task.prompt)) > best[:2]):
            best = (at, len(task.prompt), index)
    return None if best is None else (best[2], TASKS[best[2]])


def _sse(data: dict[str, Any] | str) -> bytes:
    return f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n".encode()


def _delta(model: str, delta: dict[str, Any], finish: str | None = None) -> bytes:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    return _sse({"id": "stand-in", "created": 0, "model": model, "choices": [choice]})


def _streamed(model: str, reply: str | None, call: tuple[str, str] | None, tokens: int) -> bytes:
    if call is not None:
        name, arguments = call
        tool = {"index": 0, "id": "call_0", "type": "function"}
        out = _delta(model, {"role": "assistant", "content": None})
        out += _delta(
            model, {"tool_calls": [{**tool, "function": {"name": name, "arguments": ""}}]}
        )
        cut = len(arguments) // 2
        for part in (arguments[:cut], arguments[cut:]):
            out += _delta(model, {"tool_calls": [{"index": 0, "function": {"arguments": part}}]})
        finish = "tool_calls"
    else:
        out = _delta(model, {"role": "assistant", "content": reply})
        finish = "stop"
    usage = {"prompt_tokens": 80, "completion_tokens": tokens, "total_tokens": 80 + tokens}
    tail = {"id": "stand-in", "created": 0, "model": model, "choices": [], "usage": usage}
    out += _delta(model, {}, finish) + _sse(tail) + _sse("[DONE]")
    return out


def stand_in(request: httpx2.Request) -> httpx2.Response:
    """A scripted OpenAI: tasks are answered by difficulty, with fixed token counts."""
    body: dict[str, Any] = json.loads(request.content)
    model = body["model"]
    messages = body["messages"]
    system = str(messages[0].get("content", ""))
    found = _find(str(messages[-1].get("content", "")))
    index, task = found if found else (0, None)
    if "Prompt version: analyzer" in system:
        tier = _DIFFICULTY_TIER[task.difficulty] if task else "MEDIUM"
        reply = f"{tier} {_CONFIDENCE[index % len(_CONFIDENCE)]}"
        usage = (300, 4)
        call = None
    else:
        known = task is not None and task.difficulty in _SKILL.get(model, {"easy"})
        reply = task.answer if task and known else "I am not sure."
        call = (
            (task.tool, json.dumps(task.args)) if task and known and task.type == "tool" else None
        )
        usage = (80, 6)
        if body.get("stream"):
            headers = {"content-type": "text/event-stream"}
            return httpx2.Response(
                200, content=_streamed(model, reply, call, usage[1]), headers=headers
            )
    answer = {
        "id": "stand-in",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": reply},
            }
        ],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[0] + usage[1],
        },
    }
    return httpx2.Response(200, json=answer)


def build_gate(
    config: Config, key: str, transport: httpx2.BaseTransport | None, probe: Probe
) -> PrimaGate:
    return PrimaGate(
        api_keys={"openai": key},
        transport=transport,
        tier_bindings={
            "SMALL": {"model": config.small},
            "MEDIUM": {"model": config.medium},
            "POWERFUL": {"model": config.powerful},
        },
        policy={
            "enabled_tiers": ["SMALL", "MEDIUM", "POWERFUL"],
            "confidence_threshold": config.threshold,
            "context_token_budget": 2000,
            "interim_binding": "MEDIUM",
        },
        analyzer=config.analyzer,
        event_handler=probe,
    )


def run(
    config: Config, key: str, transport: httpx2.BaseTransport | None
) -> tuple[list[CallResult], list[CallResult]]:
    probe = Probe()
    gate = build_gate(config, key, transport, probe)
    routed: list[CallResult] = []
    fixed: list[CallResult] = []
    for n, task in enumerate(TASKS):
        # Alternating tasks, and which arm goes first alternating too, so time of day, provider
        # load and any warm-up fall on both arms alike.
        if n % 2 == 0:
            routed.append(ask(gate, probe, TIER_ROUTING_ALIAS, task, "routed"))
            fixed.append(ask(gate, probe, config.baseline, task, "baseline"))
        else:
            fixed.append(ask(gate, probe, config.baseline, task, "baseline"))
            routed.append(ask(gate, probe, TIER_ROUTING_ALIAS, task, "routed"))
    return routed, fixed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Routed versus fixed-model evaluation.")
    parser.add_argument("--offline", action="store_true", help="use a scripted stand-in")
    parser.add_argument("--api-key", default=None, help="else OPENAI_API_KEY is read")
    parser.add_argument("--small", default=Config.small)
    parser.add_argument("--medium", default=Config.medium)
    parser.add_argument("--powerful", default=Config.powerful)
    parser.add_argument("--analyzer", default=Config.analyzer)
    parser.add_argument("--baseline", default=Config.baseline)
    parser.add_argument("--threshold", type=float, default=Config.threshold)
    parser.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    return parser


def _answered(results: Iterable[CallResult]) -> int:
    return sum(1 for r in results if r.error is None)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be above 0 and at most 1")
    config = Config(
        args.small, args.medium, args.powerful, args.analyzer, args.baseline, args.threshold
    )
    key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    if not args.offline and not key:
        # Before any call, so a missing key costs nothing.
        print(
            "No OpenAI key: set OPENAI_API_KEY or pass --api-key, or use --offline.",
            file=sys.stderr,
        )
        return 2
    transport = httpx2.MockTransport(stand_in) if args.offline else None
    routed, fixed = run(config, "offline-key" if args.offline else key, transport)
    when = datetime.now(UTC)
    report = render_report(config, routed, fixed, args.offline, when)
    # Printed first, so a result that cost money is never lost to a write that fails.
    print(report)
    models = "-".join((config.small, config.medium, config.powerful))
    name = re.sub(r"[^A-Za-z0-9.-]", "-", f"{config.baseline}-vs-{models}")
    stem = f"{'offline-' if args.offline else ''}harness-{name}-{when:%Y%m%d-%H%M%S}"
    # The model names hold dots, so the extension is appended and never swapped in.
    report_path = Path(args.out_dir) / f"{stem}.md"
    results_path = Path(args.out_dir) / f"{stem}.json"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8", newline="\n")
        data = results_data(config, routed, fixed, args.offline, when)
        results_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    except OSError as err:
        print(f"Could not write the report: {type(err).__name__}", file=sys.stderr)
        return 1
    print(f"Report written to {report_path}")
    print(f"Results written to {results_path}")
    silent = [name for name, rows in (("routed", routed), ("fixed", fixed)) if not _answered(rows)]
    if silent:
        # An arm that answered nothing is most likely a bad key or model name.
        print(
            f"No call was answered in the {' and '.join(silent)} arm. Check the key and the "
            "model names.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
