"""The release workflow's shape, asserted on its text (there is no YAML parser in the dev set)."""

import re
from pathlib import Path

import pytest

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
if not (WORKFLOWS / "release.yml").is_file():
    # The sdist leaves out .github, so these tests have nothing to read when run from it.
    pytest.skip("release.yml is not part of the sdist", allow_module_level=True)
TEXT = (WORKFLOWS / "release.yml").read_text(encoding="utf-8")


def _job(name: str) -> str:
    """The text of one top-level job, up to the next job."""
    match = re.search(rf"^  {name}:\n(.*?)(?=^  \w[\w-]*:\n|\Z)", TEXT, re.MULTILINE | re.DOTALL)
    assert match, name
    return match.group(1)


def test_triggers_on_version_tags_only() -> None:
    on = TEXT.split("jobs:")[0]
    assert 'tags: ["v*"]' in on
    assert "pull_request" not in on and "branches" not in on


def test_gate_calls_live_workflow_with_inherited_secrets() -> None:
    gate = _job("gate")
    assert "uses: ./.github/workflows/live.yml" in gate
    assert "secrets: inherit" in gate
    assert "require:" not in gate  # the default requires every provider


def test_job_order() -> None:
    assert "needs: gate" in _job("build")
    assert "needs: build" in _job("publish")
    assert "needs" not in _job("gate")


def test_id_token_only_in_publish() -> None:
    assert TEXT.count("id-token: write") == 1
    assert "id-token: write" in _job("publish")
    assert "contents: read" in _job("build")
    assert "id-token" not in TEXT.split("jobs:")[0]


def test_publish_uses_pypi_environment_and_never_builds() -> None:
    publish = _job("publish")
    assert re.search(r"^    environment: pypi$", publish, re.MULTILINE)
    assert "uv build" not in publish
    assert "download-artifact" in publish


def test_build_checks_then_uploads() -> None:
    build = _job("build")
    assert build.index("uv build") < build.index("check_dist.py") < build.index("upload-artifact")
    assert "--tag" in build


def test_no_pypi_credentials() -> None:
    assert "PYPI_" not in TEXT
    assert "password" not in TEXT.lower()
    assert "secrets." not in TEXT


def test_actions_pinned_by_sha() -> None:
    uses = re.findall(r"^\s+- uses: (\S+)", TEXT, re.MULTILINE)
    assert len(uses) >= 5
    for ref in uses:
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), ref


def test_build_passes_the_pushed_tag_and_cannot_be_waved_through() -> None:
    build = _job("build")
    assert 'check_dist.py --tag "${GITHUB_REF_NAME}"' in build
    assert "continue-on-error" not in TEXT
