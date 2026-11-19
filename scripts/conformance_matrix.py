"""Build the FR-to-test conformance matrix and check the suite against FR41.

Run `uv run python scripts/conformance_matrix.py` to rewrite docs/conformance-matrix.md.
tests/conformance/test_coverage_audit.py imports this file to apply the same checks.
"""

from __future__ import annotations

import io
import os
import re
import sys
import warnings
from collections import defaultdict
from collections.abc import Iterable
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "tests" / "conformance"
MATRIX = ROOT / "docs" / "conformance-matrix.md"
REGENERATE = "uv run python scripts/conformance_matrix.py"

# The PRD's FR headings, word for word. This list and REQUIRED must change when the PRD does.
FR_TITLES: dict[int, str] = {
    1: "One call across Providers",
    2: "Tools, structured output and streaming in the Unified Schema",
    3: "Launch Providers",
    4: "Raise on unsupported features",
    5: "Opt-in `drop_params`",
    6: "Drops are recorded",
    7: "Declare fallbacks",
    8: "The Fallback List is optional",
    9: "Each entry gets full reliability handling",
    10: "Skip incapable fallbacks",
    11: "One aggregate error",
    12: "Retry then redundant endpoint",
    13: "Direct dispatch when off",
    14: "Declare redundant endpoints",
    15: "Tier Routing Alias",
    16: "Declare Tier Bindings",
    17: "Declare Profile sources once",
    18: "Independent, logged routing per call",
    19: "User-chosen Analyzer",
    20: "Strict Analyzer output",
    21: "Policy",
    22: "Never block, degrade to the Interim Binding",
    23: "Background Profile preparation",
    24: "Profile and Binding data contract",
    25: "Compatibility Filter and eligibility",
    26: "Bounded routing context",
    27: "Caller cancellation",
    28: "Capacity limits",
    29: "Store interfaces",
    30: "MEDIUM-only Tier Failover",
    31: "Exact-match hit before routing",
    32: "Opt-out and streaming",
    33: "Estimates on every response",
    34: "Token counter",
    35: "Advisory spending ceiling",
    36: "Decision events",
    37: "Event delivery",
    38: "Payload retention is opt-in",
    39: "Clean install",
    40: "MIT-licensed public repository",
    41: "Conformance Suite",
    42: "Forced-failure demonstration",
    43: "Tier Routing evaluation harness",
    44: "Analyzer selection criteria",
}
LAST_FR = max(FR_TITLES)

# The FRs that FR41 says must each have at least one passing conformance test.
REQUIRED: frozenset[int] = frozenset([1, 2, *range(4, 15), 16, 20, 21, 22, 25, 27, *range(30, 39)])

MODES = frozenset({"sync", "async"})
# The parameters through which the suite really runs a test in sync or async mode. An id that
# merely reads "sync" or "async" proves nothing, so ids are never consulted.
MODE_PARAMS = frozenset({"call", "stream_call", "mode", "stopper"})
_FR_TAG = re.compile(r"FR([1-9][0-9]?)")


@dataclass(frozen=True)
class Item:
    """One collected test item, reduced to what the audit reads."""

    nodeid: str
    fr_tags: tuple[object, ...] = ()
    # None when the item has no mode_independent marker; otherwise each marker's reason.
    mode_reasons: tuple[object, ...] | None = None
    # The modes this item runs in, taken from its mode parameters.
    modes: tuple[str, ...] = ()
    # Its other parameters, so the sync and async halves of one variant can be paired.
    variant: tuple[str, ...] = ()


@dataclass
class SuiteTest:
    """One test function with all its parametrized variants."""

    nodeid: str
    items: list[Item] = field(default_factory=list)

    @property
    def module(self) -> str:
        return self.nodeid.split("::", 1)[0]

    @property
    def fr_tags(self) -> list[object]:
        seen: list[object] = []
        for item in self.items:
            seen.extend(tag for tag in item.fr_tags if tag not in seen)
        return seen

    @property
    def frs(self) -> set[int]:
        """The FRs tagged on every variant; a tag on only some variants claims nothing."""
        per_item = [{n for t in i.fr_tags if (n := fr_number(t)) is not None} for i in self.items]
        return set.intersection(*per_item) if per_item else set()


def fr_number(tag: object) -> int | None:
    """The FR number a tag names, or None when it names no FR in the PRD."""
    if not isinstance(tag, str):
        return None
    match = _FR_TAG.fullmatch(tag)
    if match is None:
        return None
    number = int(match.group(1))
    return number if number in FR_TITLES else None


def group(items: Iterable[Item]) -> list[SuiteTest]:
    tests: dict[str, SuiteTest] = {}
    for item in items:
        base = item.nodeid.partition("[")[0]
        tests.setdefault(base, SuiteTest(base)).items.append(item)
    return sorted(tests.values(), key=lambda t: t.nodeid)


def missing_required(tests: Iterable[SuiteTest]) -> list[str]:
    covered = set().union(*(t.frs for t in tests))
    return [f"FR{n}" for n in sorted(REQUIRED - covered)]


def unknown_tags(tests: Iterable[SuiteTest]) -> list[str]:
    found = []
    for test in tests:
        bad = [tag for tag in test.fr_tags if fr_number(tag) is None]
        found.extend(f"{test.nodeid}: {tag!r}" for tag in bad)
    return found


def partial_tags(tests: Iterable[SuiteTest]) -> list[str]:
    found = []
    for test in tests:
        known = {n for tag in test.fr_tags if (n := fr_number(tag)) is not None}
        found.extend(
            f"{test.nodeid}: FR{n} is on only some variants" for n in sorted(known - test.frs)
        )
    return found


def _has_reason(reason: object) -> bool:
    return isinstance(reason, str) and bool(reason.strip())


def mode_violations(tests: Iterable[SuiteTest]) -> list[str]:
    found = []
    for test in tests:
        problems: list[str] = []
        variants: dict[tuple[str, ...], list[Item]] = defaultdict(list)
        for item in test.items:
            variants[item.variant].append(item)
            if item.mode_reasons is not None and not (
                item.mode_reasons and all(_has_reason(r) for r in item.mode_reasons)
            ):
                problems.append("mode_independent needs a non-empty reason")
        for items in variants.values():
            unmarked = [i for i in items if i.mode_reasons is None]
            modes = {m for i in items for m in i.modes}
            if unmarked and {m for i in unmarked for m in i.modes} != MODES:
                problems.append("runs in one mode without mode_independent")
            elif not unmarked and modes == MODES:
                problems.append("marked mode_independent but runs in both modes")
        found.extend(f"{test.nodeid}: {p}" for p in dict.fromkeys(problems))
    return found


def _names_by_module(tests: Iterable[SuiteTest]) -> dict[str, list[str]]:
    modules: dict[str, list[str]] = defaultdict(list)
    for test in sorted(tests, key=lambda t: t.nodeid):
        modules[test.module].append(test.nodeid.split("::", 1)[1])
    return dict(sorted(modules.items()))


def render(tests: Iterable[SuiteTest]) -> str:
    tests = list(tests)
    by_fr: dict[int, list[SuiteTest]] = defaultdict(list)
    for test in tests:
        for number in test.frs:
            by_fr[number].append(test)
    covered = sum(1 for n in REQUIRED if by_fr.get(n))
    lines = [
        "# Conformance matrix",
        "",
        "<!-- Generated by scripts/conformance_matrix.py. Do not edit by hand. -->",
        "",
        "This file is generated from the `@pytest.mark.fr` markers in `tests/conformance/`.",
        "Do not edit it by hand. After you add, move or retag a conformance test, regenerate it:",
        "",
        "```bash",
        REGENERATE,
        "```",
        "",
        "`tests/conformance/test_coverage_audit.py` fails when this file is out of date, when an",
        "FR that FR41 requires has no tagged test, when a tag names an FR outside FR1 to",
        f"FR{LAST_FR} or sits on only some of a test's variants, or when a test is not run in both",
        "sync and async mode and carries no `mode_independent` reason.",
        "",
        "A test counts once, however many parametrized variants it has. A tag means the test",
        "asserts that FR's behaviour, not merely that it runs the code.",
        "",
        f"Required FRs covered: {covered} of {len(REQUIRED)}.",
        "",
        "| FR | Title | Required | Tests | Tests by module |",
        "|---|---|---|---|---|",
    ]
    for number, title in FR_TITLES.items():
        tagged = by_fr.get(number, [])
        cell = "<br>".join(
            f"`{module}`: " + ", ".join(f"`{name}`" for name in names)
            for module, names in _names_by_module(tagged).items()
        )
        required = "yes" if number in REQUIRED else "no"
        lines.append(f"| FR{number} | {title} | {required} | {len(tagged)} | {cell or 'none'} |")
    untagged = _names_by_module(t for t in tests if not t.frs)
    lines += [
        "",
        "## Untagged conformance tests",
        "",
        "These tests carry no FR tag. They are listed so that a test which loses its tag shows",
        "up here when the matrix is regenerated.",
        "",
    ]
    lines += [
        f"- `{module}`: " + ", ".join(f"`{name}`" for name in names)
        for module, names in untagged.items()
    ] or ["None."]
    return "\n".join(lines) + "\n"


def stale_message(committed: str | None, generated: str) -> str | None:
    if committed is not None and committed.replace("\r\n", "\n") == generated:
        return None
    return f"{MATRIX.relative_to(ROOT).as_posix()} is out of date; run: {REGENERATE}"


class _Collector:
    def __init__(self) -> None:
        self.items: list[Item] = []

    def pytest_collection_modifyitems(self, items: list[pytest.Item]) -> None:
        for item in items:
            # A bare fr names nothing, so it is kept as None and reported as an unknown tag.
            tags = tuple(arg for mark in item.iter_markers("fr") for arg in (mark.args or (None,)))
            marks = list(item.iter_markers("mode_independent"))
            reasons = (
                tuple(m.args[0] if m.args else m.kwargs.get("reason", "") for m in marks)
                if marks
                else None
            )
            callspec = getattr(item, "callspec", None)
            params = dict(callspec.params) if callspec is not None else {}
            modes = tuple(sorted({v for k, v in params.items() if k in MODE_PARAMS and v in MODES}))
            # Indices number the whole product, so the values pair the two halves instead.
            variant = tuple(f"{k}={v!r}" for k, v in sorted(params.items()) if k not in MODE_PARAMS)
            self.items.append(Item(item.nodeid, tags, reasons, modes, variant))


def collect(suite: Path = SUITE) -> list[SuiteTest]:
    """Collect the suite in this process, without running it."""
    collector = _Collector()
    args = [
        "--collect-only",
        "-qq",
        "--strict-markers",
        "-p",
        "no:cacheprovider",
        "--rootdir",
        str(ROOT),
        "-c",
        str(ROOT / "pyproject.toml"),
        str(suite),
    ]
    # The caller's own options must not change what is collected.
    addopts = os.environ.pop("PYTEST_ADDOPTS", None)
    # The nested session configures its plugins again, and their setup warnings would land
    # on whichever test triggered the collection.
    report = io.StringIO()
    try:
        with warnings.catch_warnings(), redirect_stdout(report):
            warnings.simplefilter("ignore")
            code = pytest.main(args, plugins=[collector])
    finally:
        if addopts is not None:
            os.environ["PYTEST_ADDOPTS"] = addopts
    if code != pytest.ExitCode.OK:
        raise RuntimeError(f"collecting {suite} failed ({code}):\n{report.getvalue()}")
    return group(collector.items)


def main(suite: Path = SUITE, matrix: Path = MATRIX) -> int:
    tests = collect(suite)
    matrix.write_text(render(tests), encoding="utf-8", newline="\n")
    problems = (
        missing_required(tests) + unknown_tags(tests) + partial_tags(tests) + mode_violations(tests)
    )
    print(f"wrote {matrix}")
    for problem in problems:
        print(f"problem: {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
