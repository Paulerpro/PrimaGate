import re
from importlib.metadata import metadata
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
REPOSITORY = "https://github.com/Paulerpro/PrimaGate"


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", ["LICENSE", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md"])
def test_the_root_files_exist(name: str) -> None:
    assert (ROOT / name).is_file()


def test_the_license_is_mit() -> None:
    assert read("LICENSE").startswith("MIT License")


def test_contributing_states_the_contribution_model() -> None:
    text = read("CONTRIBUTING.md")
    assert "owner reviews and merges every change" in text
    assert f"{REPOSITORY}/issues" in text
    assert "No DCO and no CLA" in text
    assert "CODE_OF_CONDUCT.md" in text
    for command in (
        "uv sync --locked",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run mypy primagate",
        "uv run pytest",
        "uv run pytest tests/<file> -k <name>",
    ):
        assert command in text
    assert "CI does not run it yet" in text


def test_contributing_sends_security_issues_privately() -> None:
    text = read("CONTRIBUTING.md")
    assert "Report security issues privately to nrpaulsilas@gmail.com, not in an issue." in text


def test_contributing_asks_for_the_lockfile_with_a_dependency_change() -> None:
    text = read("CONTRIBUTING.md")
    assert "If you add or change a dependency, run `uv lock` and commit `uv.lock`" in text


def test_contributing_relative_links_resolve() -> None:
    targets = re.findall(r"\]\(([^)]+)\)", read("CONTRIBUTING.md"))
    relative = [t for t in targets if not t.startswith(("http://", "https://", "mailto:", "#"))]
    assert relative
    for target in relative:
        assert (ROOT / target.split("#")[0]).is_file(), target


def test_the_code_of_conduct_names_its_standard_and_contact() -> None:
    text = read("CODE_OF_CONDUCT.md")
    assert "Contributor Covenant" in text and "version 2.1" in text
    assert "CC BY 4.0" in text
    assert "nrpaulsilas@gmail.com" in text
    assert "[INSERT CONTACT METHOD]" not in text
    assert "## Enforcement Guidelines" in text


def test_the_installed_metadata_has_the_license_and_repository() -> None:
    meta = metadata("primagate")
    assert meta["License-Expression"] == "MIT"
    assert sorted(meta.get_all("Project-URL") or []) == sorted(
        [
            f"Homepage, {REPOSITORY}",
            f"Repository, {REPOSITORY}",
            f"Issues, {REPOSITORY}/issues",
            f"Documentation, {REPOSITORY}/blob/main/README.md",
        ]
    )


PUBLIC_SURFACE = [
    "`completion`",
    "`acompletion`",
    "`PrimaGate`",
    "Policy",
    "Tier Binding",
    "`ChatCompletion`",
    "`ChatCompletionChunk`",
    "`Metadata`",
    "event types",
    "exceptions",
    "`profile_store`",
    "`token_counter`",
    "`event_handler`",
]


@pytest.mark.parametrize("item", PUBLIC_SURFACE)
def test_the_versioning_page_names_the_public_surface(item: str) -> None:
    assert item in read("docs/versioning.md")


def test_the_versioning_page_states_semver_and_the_open_deprecation_period() -> None:
    text = read("docs/versioning.md")
    assert "MAJOR.MINOR.PATCH" in text
    assert "While the version is below 1.0, a minor version may break the public surface." in text
    assert "The deprecation period is not yet set." in text
    assert "Anything under a module path not re-exported from `primagate` is internal" in text


VERSIONING_RULES = [
    "The module-level `primagate.drop_params` setting is covered.",
    "The `transport` setting is a test seam and is not covered.",
    "Adding a field, an event type or an exception subclass.",
    "Your event handlers should ignore event types and fields they do not know.",
    "Removing or renaming a field, an event type or an exception.",
    "Changing a field's type or meaning.",
    "Changing which exception a failure raises.",
    "Changing a documented default.",
    "Dropping a supported Python version.",
]


@pytest.mark.parametrize("sentence", VERSIONING_RULES)
def test_the_versioning_page_pins_its_rules(sentence: str) -> None:
    assert sentence in read("docs/versioning.md")


def test_the_versioning_page_separates_breaking_from_not_breaking() -> None:
    text = read("docs/versioning.md")
    safe, rest = text.split("These changes are breaking:")
    breaking = rest.split("## What is not covered")[0]
    assert "These changes are not breaking:" in safe
    assert "Adding a field" in safe and "Adding a field" not in breaking
    assert "Dropping a supported Python version." in breaking


@pytest.mark.parametrize("name", ["CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "docs/versioning.md"])
def test_the_readme_links_each_file_absolutely(name: str) -> None:
    # The README is also the PyPI description, where relative links do not resolve.
    assert f"]({REPOSITORY}/blob/main/{name})" in read("README.md")
    assert (ROOT / name).is_file()
