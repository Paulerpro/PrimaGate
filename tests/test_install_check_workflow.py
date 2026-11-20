"""The install-check workflow's shape, asserted on its text (no YAML parser in the dev set)."""

import re
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
if not (WORKFLOWS / "install-check.yml").is_file():
    # The sdist leaves out .github, so these tests have nothing to read when run from it.
    pytest.skip("install-check.yml is not part of the sdist", allow_module_level=True)
TEXT = (WORKFLOWS / "install-check.yml").read_text(encoding="utf-8")


def test_runs_by_hand_only_with_a_version_input() -> None:
    on = TEXT.split("jobs:")[0]
    assert "workflow_dispatch:" in on
    assert re.search(r"^      version:\n(?:        .*\n)*?        required: true", on, re.MULTILINE)
    for trigger in ("pull_request", "push", "schedule", "workflow_call", "tags"):
        assert trigger not in on


def test_matrix_covers_three_systems_and_four_pythons() -> None:
    for os_name in ("ubuntu-latest", "macos-latest", "windows-latest"):
        assert os_name in TEXT
    assert 'python: ["3.11", "3.12", "3.13", "3.14"]' in TEXT
    assert "fail-fast: false" in TEXT


def test_installs_the_input_version_with_pip_into_a_fresh_venv() -> None:
    assert "uv venv --seed" in TEXT
    assert '-m pip install "primagate==$VERSION"' in TEXT
    assert "VERSION: ${{ inputs.version }}" in TEXT
    # The input reaches the shell through env, never spliced into the script text.
    assert 'inputs.version }}"' not in TEXT
    assert "uv pip install" not in TEXT and "uv sync" not in TEXT


def test_runs_the_smoke_isolated_from_the_checkout() -> None:
    assert '-I "$GITHUB_WORKSPACE/scripts/smoke_install.py"' in TEXT
    assert "working-directory: ${{ runner.temp }}" in TEXT


def test_actions_pinned_by_sha() -> None:
    uses = re.findall(r"^\s+- uses: (\S+)", TEXT, re.MULTILINE)
    assert len(uses) == 2
    for ref in uses:
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), ref


def test_read_only_and_no_secrets() -> None:
    assert "contents: read" in TEXT
    assert "secrets." not in TEXT and "id-token" not in TEXT
