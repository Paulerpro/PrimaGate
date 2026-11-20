"""scripts/check_dist.py, tested on synthetic metadata, tiny archives and a real build."""

import importlib.util
import io
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_dist.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_dist", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_dist"] = module
    spec.loader.exec_module(module)
    return module


cd = _load()

OK = ["httpx2<3,>=2.13"]


def test_good_tag() -> None:
    assert cd.check_tag("v0.1.0", "0.1.0") == []


def test_mismatched_tag_names_both() -> None:
    (problem,) = cd.check_tag("v0.2.0", "0.1.0")
    assert "v0.2.0" in problem and "v0.1.0" in problem


@pytest.mark.parametrize("tag", ["v0.1", "0.1.0", "v0.1.0-rc1", "v01.1.0", "v0.1.0.1", ""])
def test_not_semver_tag(tag: str) -> None:
    assert cd.check_tag(tag, "0.1.0")


@pytest.mark.parametrize(
    "requirement",
    [
        "httpx2<3,>=2.13",
        "httpx2>=2.13,<3",
        "httpx2 (>=2.13,<3)",
        "tiktoken>=0.14,<1; extra == 'tokens'",
    ],
)
def test_capped_requirement_passes(requirement: str) -> None:
    assert cd.check_requirement(requirement) is None


@pytest.mark.parametrize(
    ("requirement", "name"),
    [
        ("httpx2", "httpx2"),
        ("httpx2>=2.13", "httpx2"),
        ("httpx2<3", "httpx2"),
        ("httpx2<=3,>=2", "httpx2"),
        ("tiktoken>=1; extra == 'tokens'", "tiktoken"),
    ],
)
def test_unbounded_requirement_fails_naming_it(requirement: str, name: str) -> None:
    problem = cd.check_requirement(requirement)
    assert problem is not None and f"'{name}'" in problem


def test_metadata_checks_every_requires_dist() -> None:
    text = "Name: x\nRequires-Dist: a>=1,<2\nRequires-Dist: b>=1\n"
    (problem,) = cd.check_metadata(text, "x.whl")
    assert "'b'" in problem


def _wheel(path: Path, requires: list[str]) -> None:
    body = "Name: p\n" + "".join(f"Requires-Dist: {r}\n" for r in requires)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("p-0.1.0.dist-info/METADATA", body)


def _sdist(path: Path, requires: list[str], extra: tuple[str, ...] = ()) -> None:
    body = "Name: p\n" + "".join(f"Requires-Dist: {r}\n" for r in requires)
    files = {"p-0.1.0/PKG-INFO": body, **{f"p-0.1.0/{e}": "" for e in extra}}
    with tarfile.open(path, "w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            raw = data.encode()
            info.size = len(raw)
            t.addfile(info, io.BytesIO(raw))


def test_good_dist_passes(tmp_path: Path) -> None:
    _wheel(tmp_path / "p-0.1.0-py3-none-any.whl", OK)
    _sdist(tmp_path / "p-0.1.0.tar.gz", OK)
    assert cd.check_dist(tmp_path) == []


def test_lockfile_in_sdist_fails(tmp_path: Path) -> None:
    _wheel(tmp_path / "p-0.1.0-py3-none-any.whl", OK)
    _sdist(tmp_path / "p-0.1.0.tar.gz", OK, extra=("uv.lock",))
    assert any("uv.lock" in p for p in cd.check_dist(tmp_path))


def test_unbounded_dep_in_sdist_or_wheel_fails(tmp_path: Path) -> None:
    _wheel(tmp_path / "p-0.1.0-py3-none-any.whl", OK)
    _sdist(tmp_path / "p-0.1.0.tar.gz", ["httpx2"])
    assert any("'httpx2'" in p for p in cd.check_dist(tmp_path))

    _sdist(tmp_path / "p-0.1.0.tar.gz", OK)
    _wheel(tmp_path / "p-0.1.0-py3-none-any.whl", ["tiktoken>=1; extra == 'tokens'"])
    assert any("'tiktoken'" in p for p in cd.check_dist(tmp_path))


def test_main_exit_codes(tmp_path: Path) -> None:
    _wheel(tmp_path / "p-0.1.0-py3-none-any.whl", OK)
    _sdist(tmp_path / "p-0.1.0.tar.gz", OK)
    version = cd.package_version()
    assert cd.main(["--dist", str(tmp_path), "--tag", f"v{version}"]) == 0
    assert cd.main(["--dist", str(tmp_path), "--tag", "v9.9.9"]) == 1
    assert cd.main(["--dist", str(tmp_path)]) == 0


def test_real_build_passes(tmp_path: Path) -> None:
    subprocess.run(
        ["uv", "build", "--out-dir", str(tmp_path)], cwd=ROOT, check=True, capture_output=True
    )
    assert cd.check_dist(tmp_path) == []
