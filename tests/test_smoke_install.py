"""scripts/smoke_install.py, run against a built wheel in a fresh venv, as a new user has it."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "smoke_install.py"

if not SCRIPT.is_file():
    # The sdist leaves out scripts/, so there is nothing to run when tested from it.
    pytest.skip("scripts/smoke_install.py is not part of the sdist", allow_module_level=True)


def _bare_env() -> dict[str, str]:
    # No keys and no inherited settings; Windows still needs its system root to start sockets.
    return {k: v for k, v in os.environ.items() if k.upper() in ("SYSTEMROOT", "WINDIR")}


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out)], cwd=ROOT, check=True, capture_output=True
    )
    (built,) = out.glob("*.whl")
    return built


def _fresh_venv(tmp_path: Path) -> Path:
    venv = tmp_path / "venv"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(venv)], check=True)
    return _venv_python(venv)


def test_the_built_wheel_passes_the_smoke_in_a_fresh_venv(wheel: Path, tmp_path: Path) -> None:
    python = _fresh_venv(tmp_path)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), str(wheel)],
        check=True,
        capture_output=True,
    )
    # Run from an empty directory so the checkout cannot shadow the installed package.
    run = subprocess.run(
        [str(python), "-I", str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=_bare_env(),
    )
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "ok"


def test_the_smoke_fails_when_the_package_is_not_installed(tmp_path: Path) -> None:
    python = _fresh_venv(tmp_path)
    run = subprocess.run(
        [str(python), "-I", str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=_bare_env(),
    )
    assert run.returncode != 0
    assert "ok" not in run.stdout
    assert "ModuleNotFoundError" in run.stderr
