"""Check the documentation set: every required topic is findable and every relative link resolves.

Run `uv run python scripts/check_docs.py`. It exits 1 and names each miss. External links are
never fetched, so the audit needs no network and cannot fail on someone else's outage.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Each topic is (name, file, heading pattern, phrase that must also appear in that file). A
# heading pattern matches the text after the `#` marks; the phrase pins down a topic that lives
# under a broader heading.
TOPICS: list[tuple[str, str, str, str | None]] = [
    ("quickstart", "README.md", r"Quickstart", None),
    ("error handling", "README.md", r"\d+\. Handle failures", "AttemptsExhaustedError"),
    ("privacy disclosure", "README.md", r"\d+\. What the Analyzer reads", "Profile"),
    ("truststore and proxy note", "README.md", r"Install notes", "truststore"),
    (
        "offline tiktoken cache pre-seeding",
        "README.md",
        r"\d+\. Count tokens when the provider does not",
        "TIKTOKEN_CACHE_DIR",
    ),
    ("fork safety", "README.md", r"\d+\. Background preparation and closing", "Fork safety"),
    ("reliability guide", "README.md", r"\d+\. Handle failures", "attempt_chain"),
    ("reliability demo", "examples/README.md", r"Forced-failure demo", None),
    ("Analyzer guide", "docs/analyzer-guide.md", r"Choosing an Analyzer", None),
    ("versioning page", "docs/versioning.md", r"Versioning", None),
    ("scope statement", "README.md", r"Scope", "Stage 2"),
    ("release notes", "docs/release-notes.md", r"Release notes", None),
]

# Files a topic needs besides its heading, so the demo cannot be documented but missing.
REQUIRED_FILES = ["examples/forced_failure.py"]

# Where the README must point, since a reader starts there.
README_MENTIONS = ["docs/release-notes.md"]

FENCE = re.compile(r"^```.*?^```[^\n]*$", re.MULTILINE | re.DOTALL)
CODE_SPAN = re.compile(r"`[^`\n]*`")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.MULTILINE)
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
SCHEME = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//)")


def doc_files(root: Path) -> list[Path]:
    files = [root / "README.md", root / "CONTRIBUTING.md", root / "examples" / "README.md"]
    files += sorted((root / "docs").glob("*.md"))
    return [f for f in files if f.is_file()]


def prose(text: str) -> str:
    """The text a reader sees as prose: no fenced code, whose `#` lines are not headings."""
    return FENCE.sub("", text)


def slug(heading: str) -> str:
    # GitHub's rule: lowercase, drop punctuation, spaces to hyphens. Code marks drop out as
    # punctuation, so `confidence_threshold` keeps its underscore like the rendered anchor.
    return re.sub(r"[^\w -]", "", heading.lower()).strip().replace(" ", "-")


def anchors(text: str) -> set[str]:
    seen: dict[str, int] = {}
    result: set[str] = set()
    for heading in HEADING.findall(prose(text)):
        base = slug(heading)
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.add(base if count == 0 else f"{base}-{count}")
    return result


def check_links(root: Path, path: Path) -> list[str]:
    text = CODE_SPAN.sub("", prose(path.read_text(encoding="utf-8")))
    label = path.relative_to(root).as_posix()
    problems = []
    for target in LINK.findall(text):
        target = target.strip("<>")
        if SCHEME.match(target):
            continue
        file_part, _, fragment = target.partition("#")
        destination = path if not file_part else (path.parent / file_part).resolve()
        if not destination.exists():
            problems.append(f"{label}: link {target!r} points to a file that does not exist")
            continue
        if fragment and destination.suffix == ".md":
            if fragment not in anchors(destination.read_text(encoding="utf-8")):
                problems.append(f"{label}: link {target!r} points to a heading that does not exist")
    return problems


def check_topics(root: Path) -> list[str]:
    problems = []
    for name, file, heading, phrase in TOPICS:
        path = root / file
        if not path.is_file():
            problems.append(f"topic {name!r}: {file} does not exist")
            continue
        text = path.read_text(encoding="utf-8")
        if not re.search(rf"^#+\s+{heading}\s*$", prose(text), re.MULTILINE):
            problems.append(f"topic {name!r}: {file} has no heading matching {heading!r}")
        elif phrase is not None and phrase not in text:
            problems.append(f"topic {name!r}: {file} never mentions {phrase!r}")
    for file in REQUIRED_FILES:
        if not (root / file).is_file():
            problems.append(f"required file {file} does not exist")
    readme = root / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8")
        problems += [f"README.md never links {m}" for m in README_MENTIONS if m not in text]
    return problems


def check(root: Path = ROOT) -> list[str]:
    problems = check_topics(root)
    for path in doc_files(root):
        problems += check_links(root, path)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root to check")
    args = parser.parse_args(argv)
    problems = check(args.root.resolve())
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    print("docs check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
