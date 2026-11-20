"""scripts/check_docs.py, run on this repository and on copies of it with one thing broken."""

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_docs.py"

if not SCRIPT.is_file() or not (ROOT / "docs").is_dir():
    # The sdist leaves these out, so there is nothing to audit when run from it.
    pytest.skip("the repository docs are not part of the sdist", allow_module_level=True)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_docs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_docs"] = module
    spec.loader.exec_module(module)
    return module


cd = _load()


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    """A copy of the documentation set, free to break."""
    for name in ("README.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md", "LICENSE"):
        shutil.copy(ROOT / name, tmp_path / name)
    (tmp_path / "evals").mkdir()
    shutil.copy(ROOT / "evals" / "README.md", tmp_path / "evals" / "README.md")
    shutil.copytree(ROOT / "docs", tmp_path / "docs")
    (tmp_path / "examples").mkdir()
    shutil.copy(ROOT / "examples" / "README.md", tmp_path / "examples" / "README.md")
    shutil.copy(ROOT / "examples" / "forced_failure.py", tmp_path / "examples")
    return tmp_path


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_the_repository_passes() -> None:
    assert cd.check(ROOT) == []


def test_the_copy_passes_before_it_is_broken(docs: Path) -> None:
    assert cd.check(docs) == []


def test_broken_file_link_names_file_and_link(docs: Path) -> None:
    (docs / "docs" / "extra.md").write_text("# Extra\n\n[x](missing.md)\n", encoding="utf-8")
    (problem,) = cd.check(docs)
    assert "extra.md" in problem and "missing.md" in problem


def test_broken_anchor_in_the_same_file(docs: Path) -> None:
    (docs / "docs" / "extra.md").write_text("# Extra\n\n[x](#no-such-heading)\n", encoding="utf-8")
    (problem,) = cd.check(docs)
    assert "extra.md" in problem and "#no-such-heading" in problem


def test_broken_anchor_in_another_file(docs: Path) -> None:
    (docs / "docs" / "extra.md").write_text("[x](versioning.md#nope)\n", encoding="utf-8")
    (problem,) = cd.check(docs)
    assert "versioning.md#nope" in problem


def test_good_links_and_anchors_pass(docs: Path) -> None:
    body = (
        "# Extra\n\n[a](#extra) [b](versioning.md#deprecation) "
        "[c](../README.md#7-handle-failures)\n"
    )
    (docs / "docs" / "extra.md").write_text(body, encoding="utf-8")
    assert cd.check(docs) == []


def test_external_links_are_not_checked(docs: Path) -> None:
    body = "[a](https://example.invalid/nope) [b](mailto:a@b.c)\n"
    (docs / "docs" / "extra.md").write_text(body, encoding="utf-8")
    assert cd.check(docs) == []


def test_links_inside_code_are_ignored(docs: Path) -> None:
    body = "```\n[x](missing.md)\n```\n\nInline `[y](gone.md)` too.\n"
    (docs / "docs" / "extra.md").write_text(body, encoding="utf-8")
    assert cd.check(docs) == []


def test_a_heading_inside_a_code_block_is_not_an_anchor(docs: Path) -> None:
    body = "```\n# Hidden\n```\n\n[x](#hidden)\n"
    (docs / "docs" / "extra.md").write_text(body, encoding="utf-8")
    assert len(cd.check(docs)) == 1


def test_missing_heading_names_the_topic(docs: Path) -> None:
    _edit(docs / "docs" / "analyzer-guide.md", "# Choosing an Analyzer", "# Something else")
    problems = cd.check(docs)
    assert len(problems) == 1 and "Analyzer guide" in problems[0]


def test_missing_file_names_the_topic(docs: Path) -> None:
    (docs / "docs" / "versioning.md").unlink()
    problems = cd.check(docs)
    assert any("versioning page" in p for p in problems)


def test_missing_phrase_names_the_topic(docs: Path) -> None:
    text = (docs / "README.md").read_text(encoding="utf-8").replace("TIKTOKEN_CACHE_DIR", "X")
    (docs / "README.md").write_text(text, encoding="utf-8")
    assert any("offline tiktoken" in p for p in cd.check(docs))


def test_missing_required_file_names_it(docs: Path) -> None:
    (docs / "examples" / "forced_failure.py").unlink()
    assert any("forced_failure.py" in p for p in cd.check(docs))


def test_readme_without_the_release_notes_link_fails(docs: Path) -> None:
    text = (docs / "README.md").read_text(encoding="utf-8").replace("release-notes.md", "x.md")
    (docs / "README.md").write_text(text, encoding="utf-8")
    assert any("release-notes.md" in p for p in cd.check(docs))


def test_main_exit_codes(docs: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cd.main(["--root", str(docs)]) == 0
    assert "passed" in capsys.readouterr().out
    (docs / "docs" / "extra.md").write_text("[x](missing.md)\n", encoding="utf-8")
    assert cd.main(["--root", str(docs)]) == 1
    assert "missing.md" in capsys.readouterr().err
