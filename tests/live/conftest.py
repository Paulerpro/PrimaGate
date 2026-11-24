"""Live conformance: the suite's contracts checked against real OpenAI and Anthropic replies.

Opt-in only: `uv run pytest -m live`. The default run deselects these tests, so it never spends
money or needs a key. See docs/live-conformance.md.

This file is also the pytest plugin that

- reads each provider's key from its environment variable, skipping that provider when the key
  is absent, or failing it when `PRIMAGATE_LIVE_REQUIRE` names the provider;
- classifies each failed live test as a provider outage or an SDK regression, and reports a
  regression that is a key or model problem separately, as configuration; and
- writes `live-report.md` (or the path in `PRIMAGATE_LIVE_REPORT`).

The report holds test ids, providers, models, outcomes and error type names only. Error messages
are left out because a provider's error text can quote the request.
"""

from __future__ import annotations

import os
import re
from collections.abc import Generator, Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from primagate import (
    APIConnectionError,
    APIError,
    AttemptsExhaustedError,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from .support import (
    INCONCLUSIVE,
    KEY_ENV,
    PROVIDERS,
    Live,
    LiveCaller,
    LiveConfigError,
    LiveOutage,
    MissingKeyError,
    model_for,
)

__all__ = ["LiveConfigError", "MissingKeyError"]

REQUIRE_ENV = "PRIMAGATE_LIVE_REQUIRE"
REPORT_ENV = "PRIMAGATE_LIVE_REPORT"
REPORT_NAME = "live-report.md"

OUTAGE = "outage"
REGRESSION = "regression"
CONFIGURATION = "regression (configuration)"
# The provider was briefly unable to answer: nothing in the SDK needs to change.
_TRANSIENT = (Timeout, RateLimitError, ServiceUnavailableError, APIConnectionError, LiveOutage)
_SKIP_REASON = re.compile(rf"[A-Z]+_API_KEY not set|{re.escape(INCONCLUSIVE)}")
# A rejected key or an unknown model is the operator's to fix, not the SDK's. It still fails the
# run, but the report counts it apart from SDK regressions.
_CONFIGURATION = (AuthenticationError, PermissionDeniedError, NotFoundError, LiveConfigError)
# How bad each classification is, so a test that failed twice keeps its worst.
_SEVERITY = {OUTAGE: 0, CONFIGURATION: 1, REGRESSION: 2}


# --- keys ---


def required_providers(value: str | None) -> frozenset[str]:
    names = {name.strip().lower() for name in (value or "").split(",") if name.strip()}
    unknown = sorted(names - set(PROVIDERS))
    if unknown:
        # A misspelt name would otherwise quietly turn the gate's failure back into a skip.
        raise LiveConfigError(
            f"{REQUIRE_ENV} names unknown providers: {', '.join(unknown)}; "
            f"known: {', '.join(PROVIDERS)}"
        )
    return frozenset(names)


def key_for(provider: str, env: Mapping[str, str]) -> str:
    """The provider's key; a missing one skips the test, or fails it when the gate requires it."""
    required = required_providers(env.get(REQUIRE_ENV))
    key = env.get(KEY_ENV[provider], "").strip()
    if key:
        return key
    reason = f"{KEY_ENV[provider]} not set"
    if provider in required:
        raise MissingKeyError(f"{reason}, and {REQUIRE_ENV} requires {provider}")
    pytest.skip(reason)


# --- fixtures ---


@pytest.fixture(params=PROVIDERS)
def provider(request: pytest.FixtureRequest) -> str:
    name: str = request.param
    key_for(name, os.environ)
    return name


@pytest.fixture
def live(provider: str, request: pytest.FixtureRequest) -> Iterator[Live]:
    key = key_for(provider, os.environ)
    session = Live(provider, key, model_for(provider), model_for(provider, medium=True))
    yield session
    session.close()
    sent = request.config.stash.setdefault(_REQUESTS, {})
    sent[provider] = sent.get(provider, 0) + session.events.requests


@pytest.fixture(params=["sync", "async"])
def call(request: pytest.FixtureRequest) -> LiveCaller:
    return LiveCaller(request.param)


# --- classification ---


def classify(error: BaseException | None) -> str:
    """Outage or regression, from the error's type alone, never its message."""
    if isinstance(error, AttemptsExhaustedError):
        error = error.last_error
    if isinstance(error, _TRANSIENT):
        return OUTAGE
    # No status means the reply broke off, such as a stream that ended before its last event.
    if isinstance(error, APIError) and (error.status_code is None or error.status_code >= 500):
        return OUTAGE
    if isinstance(error, _CONFIGURATION):
        return CONFIGURATION
    return REGRESSION


def error_type(error: BaseException | None) -> str:
    if error is None:
        return "unknown"
    name = type(error).__name__
    if isinstance(error, AttemptsExhaustedError):
        return f"{name} ({type(error.last_error).__name__})"
    return name


# --- results ---


@dataclass
class Result:
    nodeid: str
    provider: str
    # passed, failed or skipped
    outcome: str
    classification: str | None = None
    error_type: str | None = None
    skip_reason: str | None = None


def record(
    results: dict[str, Result],
    nodeid: str,
    provider: str,
    when: str,
    outcome: str,
    error: BaseException | None,
) -> None:
    """Fold one phase's report into the test's result; the worst failure is the one kept."""
    current = results.get(nodeid)
    if outcome == "failed":
        found = Result(nodeid, provider, "failed", classify(error), error_type(error))
        if current is None or current.outcome != "failed" or _worse(found, current):
            results[nodeid] = found
        return
    if current is not None and current.outcome == "failed":
        return
    if outcome == "skipped":
        message = str(getattr(error, "msg", "") or "")
        # Only known reasons are kept, since a skip message could hold anything.
        reason = message if _SKIP_REASON.fullmatch(message) else "other"
        results[nodeid] = Result(nodeid, provider, "skipped", skip_reason=reason)
    elif when == "call":
        results[nodeid] = Result(nodeid, provider, "passed")


def _worse(found: Result, current: Result) -> bool:
    before = _SEVERITY.get(current.classification or REGRESSION, 2)
    return _SEVERITY.get(found.classification or REGRESSION, 2) > before


_RESULTS = pytest.StashKey[dict[str, Result]]()
_REQUESTS = pytest.StashKey[dict[str, int]]()
_WRITTEN = pytest.StashKey[Path]()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if item.get_closest_marker("live") is not None:
        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec is not None else {}
        error = call.excinfo.value if call.excinfo is not None else None
        results = item.config.stash.setdefault(_RESULTS, {})
        record(
            results,
            item.nodeid,
            str(params.get("provider", "unknown")),
            report.when,
            report.outcome,
            error,
        )
    return report


def pytest_sessionfinish(session: pytest.Session) -> None:
    results = session.config.stash.get(_RESULTS, None)
    # Nothing live ran (the default run, or a collection), so there is nothing to report.
    if not results:
        return
    path = report_path(os.environ, Path.cwd())
    path.parent.mkdir(parents=True, exist_ok=True)
    requests = session.config.stash.get(_REQUESTS, {})
    text = render(results.values(), os.environ, datetime.now(UTC), requests)
    path.write_text(text, encoding="utf-8", newline="\n")
    session.config.stash[_WRITTEN] = path


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    path = terminalreporter.config.stash.get(_WRITTEN, None)
    if path is not None:
        terminalreporter.write_line(f"live conformance report: {path}")


# --- the report ---


def report_path(env: Mapping[str, str], cwd: Path) -> Path:
    chosen = env.get(REPORT_ENV, "").strip()
    if not chosen:
        return cwd / REPORT_NAME
    # Only Windows reads a trailing backslash as a separator; elsewhere it would end up in the
    # folder's name, so it is dropped before the path is built.
    path = Path(chosen.rstrip("\\") or chosen)
    if not path.is_absolute():
        path = cwd / path
    # A trailing separator, an existing folder, or a new name without a suffix is a folder.
    trailing = chosen.endswith(("/", "\\"))
    folder = trailing or path.is_dir() or (not path.exists() and not path.suffix)
    return path / REPORT_NAME if folder else path


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _providers(results: list[Result]) -> list[str]:
    seen = list(PROVIDERS)
    seen += sorted({r.provider for r in results} - set(PROVIDERS))
    return seen


def _failures(results: Iterable[Result]) -> tuple[int, int, int]:
    """Failed tests as (SDK regressions, configuration problems, provider outages)."""
    kinds = [r.classification for r in results if r.outcome == "failed"]
    return (
        sum(1 for k in kinds if k not in (OUTAGE, CONFIGURATION)),
        kinds.count(CONFIGURATION),
        kinds.count(OUTAGE),
    )


def _models(name: str, env: Mapping[str, str]) -> str:
    if name not in PROVIDERS:
        return "unknown"
    try:
        return f"{model_for(name, env)}, MEDIUM {model_for(name, env, medium=True)}"
    except LiveConfigError:
        return "invalid model override"


def verdict(results: Iterable[Result]) -> str:
    """One line: whether the run passed and, if not, which kind of failure it was."""
    results = list(results)
    regressions, configuration, outages = _failures(results)
    clauses = [
        _plural(count, noun)
        for count, noun in (
            (regressions, "SDK regression"),
            (configuration, "configuration problem"),
            (outages, "provider outage"),
        )
        if count
    ]
    if outages and len(clauses) == 1:
        return f"FAIL: {clauses[0]} and no SDK regression; rerun once the provider recovers."
    if clauses:
        joined = (
            clauses[-1] if len(clauses) == 1 else f"{', '.join(clauses[:-1])} and {clauses[-1]}"
        )
        return f"FAIL: {joined}."
    ran = {r.provider for r in results if r.outcome == "passed"}
    verified = [p for p in _providers(results) if p in ran]
    unverified = [p for p in _providers(results) if p not in ran]
    if not verified:
        return "NOT VERIFIED: no live test ran; every one was skipped."
    line = f"PASS: {' and '.join(verified)} verified"
    if unverified:
        line += f"; {' and '.join(unverified)} not verified (skipped)"
    return line + "."


def render(
    results: Iterable[Result],
    env: Mapping[str, str],
    now: datetime,
    requests: Mapping[str, int] | None = None,
) -> str:
    results = sorted(results, key=lambda r: r.nodeid)
    try:
        required = sorted(required_providers(env.get(REQUIRE_ENV)), key=_providers(results).index)
        problem = None
    except LiveConfigError:
        required, problem = [], f"{REQUIRE_ENV} is not valid; see the failures below."
    missing = [KEY_ENV[p] for p in required if not env.get(KEY_ENV[p], "").strip()]
    lines = [
        "# Live conformance report",
        "",
        f"Verdict: {verdict(results)}",
        "",
        f"Run at: {now.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Required providers: {', '.join(required) or 'none'}",
    ]
    if problem:
        lines.append(problem)
    if missing:
        lines.append(f"Missing required keys: {', '.join(missing)}")
    lines += [
        "",
        "## Counts",
        "",
        "| Provider | Models | Requests sent | Passed | Failed (outage) "
        "| Failed (configuration) | Failed (regression) | Skipped |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in _providers(results):
        mine = [r for r in results if r.provider == name]
        regressions, configuration, outages = _failures(mine)
        passed = sum(1 for r in mine if r.outcome == "passed")
        skipped = sum(1 for r in mine if r.outcome == "skipped")
        sent = (requests or {}).get(name, 0)
        lines.append(
            f"| {name} | {_models(name, env)} | {sent} | {passed} | {outages} | {configuration} "
            f"| {regressions} | {skipped} |"
        )
    failures = [r for r in results if r.outcome == "failed"]
    lines += ["", "## Failures", ""]
    if failures:
        lines += ["| Test | Provider | Classification | Error type |", "|---|---|---|---|"]
        lines += [
            f"| {r.nodeid} | {r.provider} | {r.classification} | {r.error_type} |" for r in failures
        ]
    else:
        lines.append("None.")
    reasons: dict[str, int] = {}
    for r in results:
        if r.outcome == "skipped":
            reasons[r.skip_reason or "other"] = reasons.get(r.skip_reason or "other", 0) + 1
    lines += ["", "## Skipped", ""]
    lines += [f"- {reason}: {count}" for reason, count in sorted(reasons.items())] or ["None."]
    lines += [
        "",
        "An outage is a timeout, rate limit, unavailable service, lost connection, broken-off",
        "reply or 5xx error from the provider. A configuration problem is a missing or rejected",
        "key or an unknown model. Anything else, a failed assertion included, is an SDK",
        "regression. Every failure fails the run. Requests sent counts worker dispatches and",
        "Analyzer questions. See docs/live-conformance.md.",
    ]
    return "\n".join(lines) + "\n"
