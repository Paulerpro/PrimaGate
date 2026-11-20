"""Which Analyzer should route my calls? Runs the routed arm once per candidate Analyzer and reports
what each costs, how fast and how reliably it answers, how well its Tier choices follow the task's
difficulty, and how its confidence relates to the pass rate, then turns that into a table for
setting `confidence_threshold`.

Only the Analyzer changes between candidates: the same tasks, Tiers, Policy, worker models and
threshold, caching off. No baseline arm runs here, nothing is judged by a model, and no Analyzer is
named best, since the right one depends on how you weigh cost, speed and accuracy.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import experiments
import harness
import httpx2
from workload import TASKS

from primagate import TIER_ROUTING_ALIAS

DEFAULT_CANDIDATES = ("gpt-4o-mini", "gpt-4.1-mini")
# The environment variable that holds the key of each provider.
KEY_VARIABLES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
# The suggestion wants kept calls to pass at least this often, and to be at least this share of all.
TARGET_PASS = 0.95
MIN_KEPT = 0.5
# Fewer kept calls than this and a pass rate swings on one task.
THIN_KEPT = 20
# An Analyzer answer the SDK could not use, so the call took the Interim Binding.
INVALID_REASONS = ("malformed_output", "provider_error", "tier_not_enabled")
UNVALIDATED = "early access, unvalidated"
# More than this is more likely a typo than a plan, and each candidate costs real money.
MAX_CANDIDATES = 6
_DIFFICULTY_RANK = {"easy": 0, "medium": 1, "hard": 2}
_TIE = 1e-9


@dataclass
class Candidate:
    """One candidate Analyzer: its routed calls, or why it could not be run."""

    name: str
    results: list[harness.CallResult] = field(default_factory=list)
    error: str | None = None

    @property
    def label(self) -> str:
        name = _cell(self.name)
        return f"{name} ({UNVALIDATED})" if is_early_access(self.name) else name


def is_early_access(name: str) -> bool:
    """Whether a candidate is labelled early access: decided by name, in this one place."""
    return "jev" in name.lower()


def _cell(name: str) -> str:
    """A name safe inside a markdown table cell or code span."""
    return name.replace("|", "\\|").replace("`", "'")


def parse_candidates(text: str) -> list[str]:
    """The comma separated names: at least two, each once."""
    names = [part.strip() for part in text.split(",") if part.strip()]
    if len(names) < 2:
        raise ValueError("--compare-analyzers needs at least two candidates, for example a,b")
    if len(set(names)) != len(names):
        raise ValueError("--compare-analyzers names a candidate more than once")
    if len(names) > MAX_CANDIDATES:
        raise ValueError(f"--compare-analyzers takes at most {MAX_CANDIDATES} candidates")
    return names


def provider_of(model: str) -> str | None:
    """The provider the SDK would pick for a model name, or None if the prefix is unknown."""
    prefix, sep, _ = model.partition("/")
    if not sep:
        return "anthropic" if model.startswith("claude-") else "openai"
    return prefix if prefix in KEY_VARIABLES else None


def missing_keys(
    candidates: Sequence[str], workers: Sequence[str], env: Mapping[str, str]
) -> list[str]:
    """Names of the key variables that are empty for a provider the run will call."""
    needed = {provider_of(model) for model in (*candidates, *workers)}
    return [
        var for provider, var in KEY_VARIABLES.items() if provider in needed and not env.get(var)
    ]


def rotation(names: Sequence[str], task_number: int) -> list[str]:
    """The names in the order to ask them for one task, so none is always first."""
    if not names:
        return []
    shift = task_number % len(names)
    return [*names[shift:], *names[:shift]]


def cost_estimate() -> tuple[int, float, float]:
    """Calls per candidate (a worker and an Analyzer call per task) and its cost range in cents."""
    calls = 2 * len(TASKS)
    low, high = (calls * cents / harness.MAIN_RUN_CALLS for cents in harness.MAIN_RUN_CENTS)
    return calls, low, high


def cost_note() -> str:
    calls, low, high = cost_estimate()
    return (
        f"about {calls} calls per candidate, roughly {low:.0f} to {high:.0f} US cents each at "
        "the default models (an estimate, not a bill)"
    )


def run_comparison(
    config: harness.Config,
    candidates: Sequence[str],
    keys: Mapping[str, str],
    transport: httpx2.BaseTransport | None,
    base_urls: Mapping[str, str] | None = None,
) -> list[Candidate]:
    """Ask every task through each candidate's gate, task by task, with the order rotated."""
    out = {name: Candidate(name) for name in candidates}
    gates: dict[str, Any] = {}
    probes: dict[str, harness.Probe] = {}
    for name in candidates:
        probe = harness.Probe()
        try:
            gates[name] = harness.build_gate(
                replace(config, analyzer=name),
                "",
                transport,
                probe,
                api_keys=dict(keys),
                base_urls=dict(base_urls) if base_urls else None,
            )
        except Exception as err:
            # One candidate that cannot be set up must not stop the others being compared.
            out[name].error = type(err).__name__
        else:
            probes[name] = probe
    try:
        for number, task in enumerate(TASKS):
            for name in rotation(list(gates), number):
                try:
                    row = harness.ask(gates[name], probes[name], TIER_ROUTING_ALIAS, task, "routed")
                except Exception as err:
                    # One candidate's failure must not lose the others' results.
                    out[name].error = type(err).__name__
                    del gates[name]
                    continue
                out[name].results.append(row)
    finally:
        for gate in gates.values():
            gate.close()
    for item in out.values():
        if item.error is not None or not item.results:
            continue
        answered = [r for r in item.results if r.error is None]
        if not answered:
            item.error = "NoCallAnswered"
        elif all(r.degraded for r in answered):
            # It never gave a usable answer, so there is nothing to compare it on.
            item.error = "NoUsableAnswer"
    return list(out.values())


def all_answered(run: Sequence[Candidate]) -> bool:
    return all(c.error is None for c in run)


def _ranks(values: Sequence[float]) -> list[float]:
    """Positions from 1, with tied values sharing the average of the positions they cover."""
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for i in range(start, end + 1):
            ranks[order[i]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation with ties handled, or None when it is undefined."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    sy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sx < _TIE or sy < _TIE:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True)) / (sx * sy)


@dataclass(frozen=True)
class Criteria:
    calls: int
    # Calls that asked the Analyzer: those with a usable answer and those with an unusable one.
    asked: int
    valid: int
    invalid: dict[str, int]
    # Degraded for a reason that has nothing to do with the Analyzer's answer, so left out.
    other_degraded: int
    requests: int
    # None when an Analyzer request that should have a price has none.
    cost_per_call: float | None
    cost_share: float | None
    latency_avg: float | None
    latency_p95: float | None
    exact: float | None
    at_least: float | None
    rank: float | None
    usefulness: float
    total_cost: float
    cost_per_pass: float | None

    @property
    def valid_rate(self) -> float | None:
        return self.valid / self.asked if self.asked else None


def criteria(results: list[harness.CallResult], threshold: float) -> Criteria:
    """The five criteria, and the cost and Usefulness of the routed run, for one candidate."""
    done = [r for r in results if r.error is None]
    summary = harness.summarize(results)
    invalid = {
        reason: n
        for reason in INVALID_REASONS
        if (n := sum(1 for r in done if r.degraded_reason == reason))
    }
    valid = [r for r in done if not r.degraded]
    other = sum(1 for r in done if r.degraded and r.degraded_reason not in INVALID_REASONS)
    # An unusable answer was still an Analyzer request, and a provider error was one too.
    requests = len(valid) + sum(invalid.values())
    # A provider error has no usage, so no cost is expected of it; every other request should.
    unpriced = any(
        r.analyzer_cost is None
        for r in done
        if r.degraded_reason != "provider_error"
        and (not r.degraded or r.degraded_reason in INVALID_REASONS)
    )
    spent = sum(r.analyzer_cost or 0.0 for r in done)
    worker = sum(r.worker_cost or 0.0 for r in done)
    # A request that failed or was unusable still took the Analyzer's time.
    asked = [r for r in done if not r.degraded or r.degraded_reason in INVALID_REASONS]
    latencies = [r.analyzer_latency for r in asked if r.analyzer_latency is not None]
    graded = [r for r in valid if r.tier in experiments._RANK]
    rank = spearman(
        [_DIFFICULTY_RANK[r.task.difficulty] for r in graded],
        [experiments._RANK[r.tier or ""] for r in graded],
    )
    known = any(r.analyzer_cost is not None for r in asked) and not unpriced
    return Criteria(
        calls=len(results),
        asked=requests,
        valid=len(valid),
        invalid=invalid,
        other_degraded=other,
        requests=requests,
        cost_per_call=spent / requests if known else None,
        cost_share=spent / (spent + worker) if known and spent + worker > 0 else None,
        latency_avg=sum(latencies) / len(latencies) if latencies else None,
        latency_p95=harness.percentile(latencies, 0.95) if latencies else None,
        exact=experiments.tier_agreement(valid),
        at_least=experiments.capable_enough(valid),
        rank=rank,
        usefulness=summary.usefulness,
        total_cost=summary.total_cost,
        cost_per_pass=summary.cost_per_pass,
    )


@dataclass(frozen=True)
class ThresholdRow:
    threshold: float
    kept: int
    kept_share: float | None
    kept_pass: float | None
    rest: int
    rest_pass: float | None
    # Under the threshold the run used: calls just below it already ran on the default Tier.
    approximate: bool = False

    @property
    def thin(self) -> bool:
        return self.kept < THIN_KEPT


def threshold_table(
    results: list[harness.CallResult], run_threshold: float = 0.0
) -> list[ThresholdRow]:
    """For each threshold, how many routed calls it keeps and how often each side passed.

    It reads the recorded outcomes, so it costs nothing and describes exactly the calls measured.
    """
    # Every answered call is in the base, so a call the Analyzer failed on counts as not kept and a
    # flaky Analyzer cannot look good by being judged only on the calls it got right.
    rows = [r for r in results if r.error is None]
    table = []
    for threshold in THRESHOLDS:
        sure = [r for r in rows if not r.degraded and r.confidence is not None]
        kept = [r for r in sure if (r.confidence or 0.0) >= threshold - _TIE]
        rest = [r for r in rows if r not in kept]
        table.append(
            ThresholdRow(
                threshold,
                len(kept),
                len(kept) / len(rows) if rows else None,
                sum(1 for r in kept if r.passed) / len(kept) if kept else None,
                len(rest),
                sum(1 for r in rest if r.passed) / len(rest) if rest else None,
                threshold < run_threshold - _TIE,
            )
        )
    return table


def _qualifies(row: ThresholdRow) -> bool:
    return (
        row.kept_share is not None
        and row.kept_pass is not None
        and row.kept_share >= MIN_KEPT - _TIE
        and row.kept_pass >= TARGET_PASS - _TIE
    )


def suggestion(table: list[ThresholdRow], run_threshold: float = 0.0) -> str:
    """One line naming the lowest qualifying threshold, or saying none qualifies.

    Only thresholds at or above the one the run used are offered, since the pass rates under it
    describe calls that ran on the default Tier.
    """
    chosen = next(
        (row for row in table if _qualifies(row) and row.threshold >= run_threshold - _TIE), None
    )
    wanted = (
        f"kept calls passing at least {100 * TARGET_PASS:.0f}% while keeping at least "
        f"{100 * MIN_KEPT:.0f}% of the calls"
    )
    if chosen is None:
        return (
            f"No threshold from {THRESHOLDS[0]} to {THRESHOLDS[-1]} gives {wanted} in these data, "
            "so no figure is suggested."
        )
    line = (
        f"The lowest threshold with {wanted} is {chosen.threshold}. Treat it as a data-based "
        "starting point, not a guarantee: it comes from one run of one workload."
    )
    if chosen.thin:
        line += (
            f" The counts are thin: only {chosen.kept} call(s) would be kept, under {THIN_KEPT}, "
            "so one task moves the pass rate a lot."
        )
    return line


# What each leaderboard criterion reads and which direction is better.
_LEAD = (
    ("Cost per routing call", lambda c: c.cost_per_call, min),
    ("Added latency (average)", lambda c: c.latency_avg, min),
    ("Valid-output rate", lambda c: c.valid_rate, max),
    ("Tier-to-difficulty rank correlation", lambda c: c.rank, max),
    ("Usefulness of the routed run", lambda c: c.usefulness, max),
    ("Total cost of the routed run", lambda c: c.total_cost, min),
)


def leaders(run: Sequence[Candidate], threshold: float) -> dict[str, list[str]]:
    """Per criterion, the candidate(s) ahead on it. Calibration is a status, not a ranking."""
    scored = {
        c.name: criteria(c.results, threshold)
        for c in run
        if c.error is None and criteria(c.results, threshold).valid
    }
    out: dict[str, list[str]] = {}
    for title, read, best in _LEAD:
        values = {n: read(c) for n, c in scored.items() if read(c) is not None}
        if not values:
            out[title] = []
            continue
        top = best(values.values())
        out[title] = [n for n, v in values.items() if abs(v - top) <= _TIE]
    return out


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _coefficient(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _invalid(c: Criteria) -> str:
    return ", ".join(f"{k} {v}" for k, v in c.invalid.items()) or "none"


def comparison_section(config: harness.Config, run: Sequence[Candidate], offline: bool) -> str:
    """The comparison as a report section."""
    lines = ["## Analyzer comparison", ""]
    if offline:
        lines += [f"**{harness.OFFLINE_BANNER}**", ""]
    live = [c for c in run if c.error is None]
    failed = [c for c in run if c.error is not None]
    lines += [
        "Every candidate ran the same tasks through the same Tiers, Policy, worker models and "
        f"confidence threshold ({config.threshold}), caching off. Only the Analyzer differed, and "
        "the order of the candidates rotated task by task. No baseline arm ran. No model judged "
        "any answer.",
        "",
    ]
    for c in failed:
        lines.append(f"- `{_cell(c.name)}` failed ({c.error}) and is not compared.")
    if failed:
        lines.append("")
    if not live:
        return "\n".join([*lines, "No candidate answered, so nothing is compared.", ""])
    crit = {c.name: criteria(c.results, config.threshold) for c in live}
    names = [c.name for c in live]
    lines += [
        "| Criterion | " + " | ".join(f"`{_cell(n)}`" for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]

    def row(title: str, read: Any) -> None:
        lines.append(f"| {title} | " + " | ".join(read(crit[n]) for n in names) + " |")

    row("Cost per routing call", lambda c: harness._money(c.cost_per_call))
    row("Share of routed cost", lambda c: _percent(c.cost_share))
    row("Added latency, average", lambda c: harness._seconds(c.latency_avg))
    row("Added latency, 95th percentile (indicative)", lambda c: harness._seconds(c.latency_p95))
    row("Valid-output rate", lambda c: f"{_percent(c.valid_rate)} ({c.valid}/{c.asked})")
    row("Invalid answers by reason", _invalid)
    row("Routed to the Tier the difficulty calls for", lambda c: _percent(c.exact))
    row("Routed to that Tier or a more capable one", lambda c: _percent(c.at_least))
    row("Rank correlation, difficulty to Tier", lambda c: _coefficient(c.rank))
    row("Usefulness of the routed run", lambda c: f"{c.usefulness:.1f}%")
    row("Total cost of the routed run", lambda c: harness._money(c.total_cost))
    row("Cost per successful interaction", lambda c: harness._money(c.cost_per_pass))
    lines += [
        "",
        "Valid-output rate counts routed calls that asked the Analyzer; a call that went to the "
        "Interim Binding for a reason unrelated to its answer (such as an over-budget context) is "
        "left out. Cost per routing call is the Analyzer's cost over its requests; a request that "
        "failed has no cost, so the figure is a lower bound. The correlation is `n/a` when a "
        "candidate chose one Tier for everything.",
        "",
        "### Who leads on what",
        "",
    ]
    for title, who in leaders(run, config.threshold).items():
        lines.append(f"- Leads on {title}: " + (", ".join(f"`{_cell(n)}`" for n in who) or "n/a"))
    lines += [
        "",
        "No Analyzer is named as best. Which one suits you depends on how you weigh cost, speed "
        "and accuracy.",
        "",
    ]
    for c in live:
        lines += _candidate_section(config, c)
    return "\n".join(lines)


def _candidate_section(config: harness.Config, c: Candidate) -> list[str]:
    lines = [f"### {c.label}", ""]
    buckets = harness.calibration(c.results, config.threshold)
    lines += [
        "Calibration, from this candidate's own calls:",
        "",
        "| Confidence | Calls | Passed | Pass rate |",
        "|---|---|---|---|",
    ]
    for b in buckets:
        mark = " (too few)" if b.thin else ""
        rate = None if b.rate is None else 100 * b.rate
        lines.append(f"| {b.label} | {b.calls} | {b.passes} | {harness._percent(rate)}{mark} |")
    table = threshold_table(c.results, config.threshold)
    lines += [
        "",
        harness.calibration_statement(buckets),
        "",
        "Setting `confidence_threshold` from this run's recorded outcomes. Kept calls keep their "
        "chosen Tier; the rest would take `ambiguous_context_default`. Calls below the threshold "
        f"this run used ({config.threshold}) already ran on that default, so rows under it are "
        "approximate.",
        "",
        "| Threshold | Kept share | Kept pass rate | Rest pass rate |",
        "|---|---|---|---|",
    ]
    for t in table:
        thin = " (thin)" if t.thin else ""
        thin += " (approximate)" if t.approximate else ""
        lines.append(
            f"| {t.threshold} | {_percent(t.kept_share)} ({t.kept}){thin} "
            f"| {_percent(t.kept_pass)} | {_percent(t.rest_pass)} |"
        )
    return [*lines, "", suggestion(table, config.threshold), ""]


def _numbers(c: Criteria) -> dict[str, Any]:
    return {
        "calls": c.calls,
        "asked": c.asked,
        "valid": c.valid,
        "valid_rate": c.valid_rate,
        "invalid": c.invalid,
        "other_degraded": c.other_degraded,
        "requests": c.requests,
        "cost_per_call": c.cost_per_call,
        "cost_share": c.cost_share,
        "latency_avg": c.latency_avg,
        "latency_p95": c.latency_p95,
        "tier_exact": c.exact,
        "tier_at_least": c.at_least,
        "rank_correlation": c.rank,
        "usefulness": c.usefulness,
        "total_cost": c.total_cost,
        "cost_per_pass": c.cost_per_pass,
    }


def comparison_data(
    config: harness.Config, run: Sequence[Candidate], offline: bool
) -> dict[str, Any]:
    """The comparison as plain facts for the results file: numbers, names and call records."""
    candidates: dict[str, Any] = {}
    for c in run:
        if c.error is not None:
            candidates[c.name] = {
                "failed": True,
                "error": c.error,
                "unvalidated": is_early_access(c.name),
                "criteria": {},
                "thresholds": [],
                "suggestion": None,
                "calibration": [],
                "calls": [harness.call_record(r) for r in c.results],
            }
            continue
        table = threshold_table(c.results, config.threshold)
        candidates[c.name] = {
            "failed": False,
            "error": None,
            "unvalidated": is_early_access(c.name),
            "criteria": _numbers(criteria(c.results, config.threshold)),
            "thresholds": [
                {
                    "threshold": t.threshold,
                    "kept": t.kept,
                    "kept_share": t.kept_share,
                    "kept_pass": t.kept_pass,
                    "approximate": t.approximate,
                    "rest": t.rest,
                    "rest_pass": t.rest_pass,
                }
                for t in table
            ],
            "suggestion": suggestion(table, config.threshold),
            "calibration": [
                {"label": b.label, "calls": b.calls, "passes": b.passes}
                for b in harness.calibration(c.results, config.threshold)
            ],
            "calls": [harness.call_record(r) for r in c.results],
        }
    return {
        "status": "ran",
        "offline": offline,
        "threshold": config.threshold,
        "target_pass": TARGET_PASS,
        "min_kept": MIN_KEPT,
        "candidates": candidates,
        "leaders": leaders(run, config.threshold),
    }
