"""A human-rated sample: export responses to rate blind, then score the ratings.

The rating file shows a person only a neutral item id, the conversation and the response, so they
rate the response and not the system that made it. A separate key file maps each id to its arm and
task, and is read only when the ratings are scored. No model rates anything.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import harness

SAMPLE_SIZE = 50
# Fixed, so the same results always give the same sample.
SEED = 4217
FIELDS = ("item", "prompt", "response", "useful")
KEY_VERSION = 1
VALID = {"yes": True, "no": False}


class RatingError(Exception):
    """A rating file or key that cannot be scored; the message names what to fix."""


@dataclass(frozen=True)
class Exported:
    csv: Path
    key: Path
    count: int
    note: str


@dataclass
class ArmScore:
    rated: int = 0
    useful: int = 0
    # How many of the same items the automatic check passed, for a side-by-side reading.
    automatic: int = 0

    @property
    def usefulness(self) -> float:
        return 100.0 * self.useful / self.rated if self.rated else 0.0


@dataclass
class Scored:
    per_arm: dict[str, ArmScore] = field(default_factory=dict)
    agreed: int = 0
    rated: int = 0
    offline: bool = False
    differing: list[tuple[str, str]] = field(default_factory=list)
    # Items in the key that the file has no row for: never counted as rated.
    missing: list[str] = field(default_factory=list)
    exported: int = 0

    @property
    def disagreements(self) -> list[str]:
        return [task for task, _ in self.differing]

    def render(self) -> str:
        lines = ["# Human-rated sample", ""]
        if self.offline:
            lines += [f"**{harness.OFFLINE_BANNER}**", ""]
        lines += [
            "A person rated each response yes or no for whether it is useful, without being shown "
            "which arm made it. No judge model was used.",
            "",
            "| Arm | Rated | Rated useful | Human Usefulness | Automatic check, same items |",
            "|---|---|---|---|---|",
        ]
        for arm, score in sorted(self.per_arm.items()):
            lines.append(
                f"| {arm} | {score.rated} | {score.useful} | {score.usefulness:.1f}% "
                f"| {score.automatic} of {score.rated} passed |"
            )
        if self.missing:
            lines += [
                "",
                f"Only {self.rated} of {self.exported} exported items were rated. No row for: "
                f"{', '.join(self.missing)}",
            ]
        share = 100.0 * self.agreed / self.rated if self.rated else 0.0
        lines += [
            "",
            f"Agreement between the human rating and the automatic check: {self.agreed} of "
            f"{self.rated} ({share:.1f}%).",
        ]
        if self.differing:
            shown = ", ".join(f"{task} ({arm})" for task, arm in self.differing)
            lines.append(f"They disagreed on: {shown}")
        else:
            lines.append("They disagreed on no item.")
        lines += [
            "",
            "A sample of this size is a check on the automatic rules, not a measure of the "
            "product.",
            "",
        ]
        return "\n".join(lines)


# A spreadsheet runs a cell that starts with one of these as a formula.
_FORMULA = ("=", "+", "-", "@", chr(9), chr(13))


def neutral(text: str) -> str:
    """The text with an apostrophe in front when a spreadsheet would run it as a formula."""
    return "'" + text if text.startswith(_FORMULA) else text


def _prompt(call: harness.CallResult) -> str:
    lines = [f"{m['role'].capitalize()}: {m['content']}" for m in call.task.messages()]
    if call.task.tools:
        lines.append(f"[Tools offered: {', '.join(call.task.tools)}]")
    return "\n".join(lines)


def _response(call: harness.CallResult) -> str:
    if call.tool_call is not None:
        name, arguments = call.tool_call
        return f"Tool call: {name} {arguments}"
    return (call.reply or "").replace("\r\n", "\n") or "(no response)"


def _allocate(sizes: dict[tuple[str, str], int], total: int) -> dict[tuple[str, str], int]:
    """How many to take from each stratum: proportional, with the leftovers to the largest parts."""
    pool = sum(sizes.values())
    if pool <= total:
        return dict(sizes)
    exact = {k: total * v / pool for k, v in sizes.items()}
    taken = {k: int(x) for k, x in exact.items()}
    left = total - sum(taken.values())
    for k in sorted(sizes, key=lambda k: (-(exact[k] - taken[k]), k))[:left]:
        taken[k] += 1
    return taken


def sample(
    routed: list[harness.CallResult],
    baseline: list[harness.CallResult],
    size: int = SAMPLE_SIZE,
    seed: int = SEED,
) -> list[harness.CallResult]:
    """Up to `size` answered calls from both arms, spread over type and difficulty, shuffled."""
    pool = sorted(
        (r for r in (*routed, *baseline) if r.error is None),
        key=lambda r: (r.arm, r.task.id),
    )
    strata: dict[tuple[str, str], list[harness.CallResult]] = {}
    for r in pool:
        strata.setdefault((r.task.type, r.task.difficulty), []).append(r)
    rng = random.Random(seed)
    take = _allocate({k: len(v) for k, v in strata.items()}, size)
    chosen: list[harness.CallResult] = []
    for stratum in sorted(strata):
        chosen += rng.sample(strata[stratum], take[stratum])
    # Shuffled, so the order gives away neither the type nor the arm.
    rng.shuffle(chosen)
    return chosen


def export(
    routed: list[harness.CallResult],
    baseline: list[harness.CallResult],
    out_dir: Path,
    offline: bool,
    when: datetime,
    results_name: str,
    size: int = SAMPLE_SIZE,
    seed: int = SEED,
) -> Exported:
    """Write the rating file and, apart from it, the key that says which arm made each item."""
    chosen = sample(routed, baseline, size, seed)
    if not chosen:
        raise RatingError("There are no answered calls to export.")
    stem = f"rating-{when:%Y%m%d-%H%M%S}"
    csv_path, key_path = out_dir / f"{stem}.csv", out_dir / f"{stem}.key.json"
    items = {f"item-{n:03d}": call for n, call in enumerate(chosen, start=1)}
    key: dict[str, Any] = {
        "version": KEY_VERSION,
        "offline": offline,
        "seed": seed,
        "results": results_name,
        "items": {item: {"arm": call.arm, "task_id": call.task.id} for item, call in items.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    key_path.write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8", newline="\n")
    try:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(FIELDS)
            for item, call in items.items():
                writer.writerow([item, neutral(_prompt(call)), neutral(_response(call)), ""])
    except OSError:
        # A key with no file to go with it would only mislead.
        key_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
        raise
    note = f"Exported {len(chosen)} response(s) to rate."
    if len(chosen) < size:
        note = f"Only {len(chosen)} response(s) were available, so {note.lower()}"
    return Exported(csv_path, key_path, len(chosen), note)


def _read_key(path: Path) -> dict[str, Any]:
    key_path = path.with_suffix(".key.json")
    unreadable = RatingError(f"The key file {key_path} cannot be read.")
    try:
        key = json.loads(key_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RatingError(f"The key file {key_path} is missing; it is needed to score.") from None
    except (OSError, ValueError):
        raise unreadable from None
    if not isinstance(key, dict) or not isinstance(key.get("items"), dict):
        raise unreadable
    if not isinstance(key.get("results"), str) or not key["results"]:
        raise unreadable
    for entry in key["items"].values():
        if not (
            isinstance(entry, dict)
            and isinstance(entry.get("arm"), str)
            and isinstance(entry.get("task_id"), str)
        ):
            raise unreadable
    return key


def _automatic(key: dict[str, Any], path: Path) -> dict[tuple[str, str], bool]:
    # Only the name is used, so a key can never point outside the folder it sits in.
    results = path.with_name(Path(key["results"].replace("\\", "/")).name)
    try:
        calls = json.loads(results.read_text(encoding="utf-8"))["calls"]
        return {(c["arm"], c["task_id"]): bool(c["passed"]) for c in calls}
    except (OSError, ValueError, KeyError, TypeError):
        raise RatingError(
            f"The results file {results} that the key points at cannot be read."
        ) from None


def score(path: Path) -> Scored:
    """Score a rated file. Any bad row stops it, so a partial score is never reported."""
    key = _read_key(path)
    automatic = _automatic(key, path)
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as err:
        raise RatingError(f"Could not read {path}: {type(err).__name__}") from None
    names = reader.fieldnames or []
    if len(names) == 1 and ";" in names[0]:
        raise RatingError(
            f"{path} looks semicolon separated; save it as a comma separated CSV (UTF-8)."
        )
    if not {"item", "useful"} <= set(names):
        raise RatingError(f"{path} needs the columns item and useful.")
    problems: list[str] = []
    seen: set[str] = set()
    ratings: dict[str, bool] = {}
    for number, row in enumerate(rows, start=2):
        item = (row.get("item") or "").strip().lstrip("'")
        if item not in key["items"]:
            problems.append(f"row {number}: item {item!r} is not in the key")
            continue
        if item in seen:
            problems.append(f"row {number}: item {item} appears more than once")
            continue
        seen.add(item)
        value = (row.get("useful") or "").strip().lstrip("'").strip().lower()
        if value not in VALID:
            problems.append(f"row {number} (item {item}): the rating must be yes or no")
            continue
        ratings[item] = VALID[value]
    if problems:
        raise RatingError("Nothing was scored:\n" + "\n".join(problems))
    missing = sorted(set(key["items"]) - seen)
    scored = Scored(offline=bool(key.get("offline")), missing=missing, exported=len(key["items"]))
    for item, useful in ratings.items():
        arm, task = key["items"][item]["arm"], key["items"][item]["task_id"]
        passed = automatic.get((arm, task))
        if passed is None:
            raise RatingError(f"The results file has no {arm} call for task {task}.")
        row_score = scored.per_arm.setdefault(arm, ArmScore())
        row_score.rated += 1
        row_score.useful += useful
        row_score.automatic += passed
        scored.rated += 1
        if useful == passed:
            scored.agreed += 1
        else:
            scored.differing.append((task, arm))
    scored.differing.sort()
    return scored
