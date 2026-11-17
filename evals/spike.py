"""Does Tier Routing save money without losing quality? A first, rough measurement.

Runs the fixed workload twice, through the Tier Routing alias and straight to one premium model,
checks every reply by rule, and writes a report. Run it with `uv run python evals/spike.py`, which
needs OPENAI_API_KEY, or with `--offline` to try the harness against a scripted stand-in.

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
# Below this many answered calls per arm, a pass rate says too little to call a result.
MIN_COMPLETED = 30
# Rounding slack so exactly 30% or exactly 2 points is not lost to floating point.
EPSILON = 1e-9

OFFLINE_BANNER = "OFFLINE STAND-IN: not evidence"


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
    task: Task
    reply: str | None
    # The class name of what was raised, never its text, which could echo a prompt.
    error: str | None
    latency: float
    worker_cost: float | None = None
    analyzer_cost: float | None = None
    analyzer_latency: float | None = None
    tier: str | None = None
    degraded: bool = False

    @property
    def passed(self) -> bool:
        return self.error is None and self.task.check(self.reply)


@dataclass(frozen=True)
class Summary:
    calls: int
    completed: int
    passes: int
    usefulness: float
    worker_cost: float
    analyzer_cost: float
    unknown_cost: tuple[str, ...]
    latency_avg: float
    latency_p95: float
    analyzer_latency_avg: float | None
    tiers: dict[str, int]
    degraded: int
    errors: int
    # Pass counts per difficulty as (passed, asked), so a level a model cannot do shows.
    by_level: dict[str, tuple[int, int]]
    failed: tuple[str, ...]

    @property
    def total_cost(self) -> float:
        return self.worker_cost + self.analyzer_cost


def summarize(results: list[CallResult]) -> Summary:
    completed = [r for r in results if r.error is None]
    passes = sum(1 for r in results if r.passed)
    latencies = sorted(r.latency for r in completed)
    analyzer_latencies = [r.analyzer_latency for r in completed if r.analyzer_latency is not None]
    tiers: dict[str, int] = {}
    for r in completed:
        if r.tier:
            tiers[r.tier] = tiers.get(r.tier, 0) + 1
    return Summary(
        calls=len(results),
        completed=len(completed),
        passes=passes,
        usefulness=100.0 * passes / len(results) if results else 0.0,
        worker_cost=sum(r.worker_cost for r in completed if r.worker_cost is not None),
        analyzer_cost=sum(r.analyzer_cost for r in completed if r.analyzer_cost is not None),
        # Listed by id and left out of the sums, so a missing price is never read as free.
        unknown_cost=tuple(
            r.task.id
            for r in completed
            if r.worker_cost is None
            # An Analyzer that ran but has no price would otherwise count as free.
            or (r.tier is not None and not r.degraded and r.analyzer_cost is None)
        ),
        latency_avg=sum(latencies) / len(latencies) if latencies else 0.0,
        latency_p95=latencies[max(math.ceil(0.95 * len(latencies)) - 1, 0)] if latencies else 0.0,
        analyzer_latency_avg=(
            sum(analyzer_latencies) / len(analyzer_latencies) if analyzer_latencies else None
        ),
        tiers=tiers,
        degraded=sum(1 for r in completed if r.degraded),
        errors=len(results) - len(completed),
        by_level=_by_level(results),
        failed=tuple(r.task.id for r in results if not r.passed),
    )


def _by_level(results: list[CallResult]) -> dict[str, tuple[int, int]]:
    levels: dict[str, tuple[int, int]] = {}
    for r in results:
        passed, asked = levels.get(r.task.level, (0, 0))
        levels[r.task.level] = (passed + (1 if r.passed else 0), asked + 1)
    return levels


def verdict(routed: Summary, baseline: Summary, offline: bool) -> str:
    """The one line that answers SM-4, with the numbers it rests on."""
    if offline:
        return "SM-4 NOT EVALUATED: this run used the offline stand-in"
    if min(routed.completed, baseline.completed) < MIN_COMPLETED:
        return f"INCONCLUSIVE: fewer than {MIN_COMPLETED} completed calls in an arm"
    if baseline.total_cost <= 0 or baseline.unknown_cost or routed.unknown_cost:
        return "INCONCLUSIVE: a cost is unknown or zero, so the saving cannot be computed"
    saving = 100.0 * (baseline.total_cost - routed.total_cost) / baseline.total_cost
    drop = baseline.usefulness - routed.usefulness
    met = saving >= COST_TARGET - EPSILON and drop <= QUALITY_TOLERANCE + EPSILON
    shown = f"cost {saving:+.1f}% lower, Usefulness {drop:+.1f} points lower"
    target = f"at least {COST_TARGET:.0f}% and at most {QUALITY_TOLERANCE:.0f} points"
    line = f"SM-4 {'MET' if met else 'NOT MET'}: {shown} (target: {target})"
    if routed.degraded:
        # Degraded calls went to the Interim Binding, so the routed arm was not all routing.
        line += f"; {routed.degraded} routed call(s) fell back to the Interim Binding"
    return line


def _money(value: float) -> str:
    return f"${value:.6f}"


def _rate(level: tuple[int, int] | None) -> str:
    return "n/a" if not level else f"{level[0]}/{level[1]}"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1000:.0f} ms"


def render_report(
    config: Config, routed: Summary, baseline: Summary, offline: bool, when: datetime
) -> str:
    lines = [f"# Routing spike {when:%Y-%m-%d %H:%M} UTC", ""]
    if offline:
        lines += [f"**{OFFLINE_BANNER}**", ""]
    lines += [
        f"**{verdict(routed, baseline, offline)}**",
        "",
        "Usefulness is the share of tasks whose reply passes an objective check. No model judged "
        "any answer.",
        "",
        "## Setup",
        "",
        f"- Tiers: SMALL `{config.small}`, MEDIUM `{config.medium}`, POWERFUL `{config.powerful}`",
        f"- Analyzer: `{config.analyzer}`, confidence threshold {config.threshold}, "
        f"interim binding MEDIUM, context budget 2000 tokens",
        f"- Baseline (fixed premium model): `{config.baseline}`",
        f"- Workload: {len(TASKS)} tasks, one independent call each, caching off",
        "",
        "## Results",
        "",
        "| | Tier Routing | Fixed model |",
        "|---|---|---|",
        f"| Calls (answered) | {routed.calls} ({routed.completed}) "
        f"| {baseline.calls} ({baseline.completed}) |",
        f"| Passed | {routed.passes} | {baseline.passes} |",
        f"| Usefulness | {routed.usefulness:.1f}% | {baseline.usefulness:.1f}% |",
        f"| Worker cost | {_money(routed.worker_cost)} | {_money(baseline.worker_cost)} |",
        f"| Analyzer cost | {_money(routed.analyzer_cost)} | - |",
        f"| **Total cost** | **{_money(routed.total_cost)}** | **{_money(baseline.total_cost)}** |",
        f"| Latency, average | {_seconds(routed.latency_avg)} | {_seconds(baseline.latency_avg)} |",
        f"| Latency, 95th percentile (indicative) | {_seconds(routed.latency_p95)} "
        f"| {_seconds(baseline.latency_p95)} |",
        "",
        "## Analyzer overhead (Tier Routing only)",
        "",
        f"- Cost: {_money(routed.analyzer_cost)} of {_money(routed.total_cost)} total",
        f"- Added latency, average: {_seconds(routed.analyzer_latency_avg)} per call",
        f"- Calls degraded to the Interim Binding: {routed.degraded}",
        "- Calls per Tier: "
        + (
            ", ".join(
                f"{k} {v} ({100.0 * v / routed.completed:.0f}%)"
                for k, v in sorted(routed.tiers.items())
            )
            or "none"
        ),
        "",
        "## Pass rate by difficulty",
        "",
        "| Level | Tier Routing | Fixed model |",
        "|---|---|---|",
        *[
            f"| {level} | {_rate(routed.by_level.get(level))} "
            f"| {_rate(baseline.by_level.get(level))} |"
            for level in ("easy", "medium", "hard")
        ],
        "",
    ]
    problems = [
        f"- Tier Routing: {routed.errors} call(s) raised" if routed.errors else "",
        f"- Fixed model: {baseline.errors} call(s) raised" if baseline.errors else "",
        f"- Failed in the routed arm: {', '.join(routed.failed)}" if routed.failed else "",
        f"- Failed in the fixed arm: {', '.join(baseline.failed)}" if baseline.failed else "",
        f"- Cost unknown, left out of the sums (routed): {', '.join(routed.unknown_cost)}"
        if routed.unknown_cost
        else "",
        f"- Cost unknown, left out of the sums (fixed): {', '.join(baseline.unknown_cost)}"
        if baseline.unknown_cost
        else "",
    ]
    problems = [p for p in problems if p]
    if problems:
        lines += ["## Notes", "", *problems, ""]
    return "\n".join(lines)


class Probe:
    """Watches the events of one call, for the facts the response does not carry."""

    def __init__(self) -> None:
        self.decided_at: datetime | None = None
        self.degraded = False

    def reset(self) -> None:
        self.decided_at = None
        self.degraded = False

    def __call__(self, event: Event) -> None:
        if isinstance(event, RoutingDecision) and event.timestamp:
            # The decision is announced right after the Analyzer answers, so the time to it is
            # the time the Analyzer added.
            self.decided_at = datetime.fromisoformat(event.timestamp)
        elif isinstance(event, RoutingDegraded):
            self.degraded = True


def _messages(task: Task) -> list[dict[str, str]]:
    return [{"role": "user", "content": task.prompt}]


def _added(decided: datetime | None, started: datetime) -> float | None:
    """Seconds from the call starting to the routing decision, or None if it cannot be told."""
    if decided is None:
        return None
    try:
        # Never negative: a coarse or mismatched clock must not skew the average.
        return max(0.0, (decided - started).total_seconds())
    except TypeError:
        return None


def ask(gate: PrimaGate, probe: Probe, model: str, task: Task) -> CallResult:
    probe.reset()
    started = datetime.now(UTC)
    clock = time.perf_counter()
    try:
        response = gate.completion(
            model=model, messages=_messages(task), temperature=0, max_tokens=64, cache=False
        )
    except Exception as err:
        return CallResult(task, None, type(err).__name__, time.perf_counter() - clock)
    latency = time.perf_counter() - clock
    meta = response.metadata
    analyzer = meta.analyzer if meta else None
    added = _added(probe.decided_at, started)
    return CallResult(
        task=task,
        reply=response.choices[0].message.content if response.choices else None,
        error=None,
        latency=latency,
        worker_cost=meta.cost.total if meta and meta.cost else None,
        analyzer_cost=analyzer.cost.total if analyzer and analyzer.cost else None,
        analyzer_latency=added,
        tier=meta.tier if meta else None,
        degraded=probe.degraded,
    )


_LEVEL_TIER = {"easy": "SMALL", "medium": "MEDIUM", "hard": "POWERFUL"}
# What each model can answer in the stand-in, so the offline run has something to measure.
_SKILL = {
    "gpt-4o-mini": {"easy"},
    "gpt-4.1-mini": {"easy", "medium"},
    "gpt-4o": {"easy", "medium", "hard"},
}


def stand_in(request: httpx2.Request) -> httpx2.Response:
    """A scripted OpenAI: tasks are answered by difficulty, with fixed token counts."""
    body: dict[str, Any] = json.loads(request.content)
    messages = body["messages"]
    system = str(messages[0].get("content", ""))
    text = str(messages[-1].get("content", ""))
    task = next((t for t in TASKS if t.prompt in text), None)
    if "Prompt version: analyzer" in system:
        reply = f"{_LEVEL_TIER.get(task.level if task else 'easy', 'MEDIUM')} 0.9"
        usage = (300, 4)
    else:
        known = task is not None and task.level in _SKILL.get(body["model"], {"easy"})
        reply = task.answer if task and known else "I am not sure."
        usage = (80, 6)
    answer = {
        "id": "stand-in",
        "created": 0,
        "model": body["model"],
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
) -> tuple[Summary, Summary]:
    probe = Probe()
    gate = build_gate(config, key, transport, probe)
    routed: list[CallResult] = []
    fixed: list[CallResult] = []
    for task in TASKS:
        # Alternating, so time of day and provider load fall on both arms alike.
        routed.append(ask(gate, probe, TIER_ROUTING_ALIAS, task))
        fixed.append(ask(gate, probe, config.baseline, task))
    return summarize(routed), summarize(fixed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Routed versus fixed-model spike.")
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
    path = (
        Path(args.out_dir)
        / f"{'offline-' if args.offline else ''}spike-{name}-{when:%Y%m%d-%H%M%S}.md"
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report, encoding="utf-8")
    except OSError as err:
        print(f"Could not write the report: {type(err).__name__}", file=sys.stderr)
        return 1
    print(f"Report written to {path}")
    if routed.completed == 0 and fixed.completed == 0:
        # Nothing was answered, most likely a bad key or model name.
        print("No call was answered. Check the key and the model names.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
