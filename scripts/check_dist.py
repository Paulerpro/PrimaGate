"""Check a release before it is uploaded: the tag, and the built wheel and sdist.

Run `uv build && uv run python scripts/check_dist.py --tag v0.1.0`. The release workflow runs it
in the build job, so a bad tag or an unbounded dependency fails before anything is published.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "primagate" / "__init__.py"

# Digits only: v1 publishes no pre-release suffix, so an rc tag is refused rather than shipped.
TAG = re.compile(r"v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)")
VERSION_LINE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)
REQUIRES = re.compile(r"^Requires-Dist:\s*(.+?)\s*$", re.MULTILINE)
UPPER_CAP = re.compile(r"^<\s*\d")
LOWER_BOUND = re.compile(r"^>=\s*\d")


def package_version(init: Path = INIT) -> str:
    # Read as text: importing the package would check the checkout, not what was built.
    match = VERSION_LINE.search(init.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"no __version__ found in {init}")
    return match.group(1)


def check_tag(tag: str, version: str) -> list[str]:
    if not TAG.fullmatch(tag):
        return [f"tag {tag!r} is not semantic versioning: expected v<MAJOR>.<MINOR>.<PATCH>"]
    if tag != f"v{version}":
        return [f"tag {tag!r} does not match the package version: expected 'v{version}'"]
    return []


def check_requirement(requirement: str) -> str | None:
    """Return a problem with one `Requires-Dist` value, or None when it is bounded."""
    spec = requirement.split(";", 1)[0].strip()
    name_match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", spec)
    name = name_match.group(0) if name_match else spec
    rest = spec[len(name) :].strip()
    rest = re.sub(r"^\[[^\]]*\]", "", rest).strip()
    if rest.startswith("(") and rest.endswith(")"):
        rest = rest[1:-1]
    parts = [p.strip() for p in rest.split(",") if p.strip()]
    missing = []
    if not any(LOWER_BOUND.match(p) for p in parts):
        missing.append("a lower bound (>=)")
    if not any(UPPER_CAP.match(p) for p in parts):
        missing.append("an upper major cap (<N)")
    if missing:
        return f"dependency {name!r} ({requirement}) lacks " + " and ".join(missing)
    return None


def check_metadata(text: str, label: str) -> list[str]:
    problems = []
    for requirement in REQUIRES.findall(text):
        problem = check_requirement(requirement)
        if problem:
            problems.append(f"{label}: {problem}")
    return problems


def check_wheel(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as wheel:
        names = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
        if len(names) != 1:
            return [f"{path.name}: expected one METADATA file, found {len(names)}"]
        text = wheel.read(names[0]).decode("utf-8")
    return check_metadata(text, path.name)


def check_sdist(path: Path) -> list[str]:
    problems = []
    with tarfile.open(path, "r:gz") as sdist:
        members = sdist.getnames()
        if any(Path(n).name == "uv.lock" for n in members):
            problems.append(f"{path.name}: the sdist contains uv.lock")
        info = [n for n in members if n.count("/") == 1 and n.endswith("/PKG-INFO")]
        if len(info) != 1:
            problems.append(f"{path.name}: expected one top-level PKG-INFO, found {len(info)}")
        else:
            handle = sdist.extractfile(info[0])
            assert handle is not None
            problems += check_metadata(handle.read().decode("utf-8"), path.name)
    return problems


def check_dist(dist: Path) -> list[str]:
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        found = f"{len(wheels)} wheels and {len(sdists)} sdists"
        return [f"{dist}: expected exactly one wheel and one sdist, found {found}"]
    return check_wheel(wheels[0]) + check_sdist(sdists[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="the release tag to check against the package version")
    parser.add_argument("--dist", type=Path, default=ROOT / "dist", help="directory of built files")
    args = parser.parse_args(argv)

    problems: list[str] = []
    if args.tag is not None:
        problems += check_tag(args.tag, package_version())
    problems += check_dist(args.dist)
    for problem in problems:
        print(f"error: {problem}", file=sys.stderr)
    if problems:
        return 1
    print("release check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
