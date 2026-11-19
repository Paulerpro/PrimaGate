import importlib.util
import random
import re
import sys
import textwrap
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = [
    pytest.mark.fr("FR41"),
    pytest.mark.mode_independent("audits how the suite is collected; makes no call"),
]

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "conformance_matrix.py"
X = "tests/conformance/test_x.py"


def _load() -> ModuleType:
    # The script lives outside the package and outside any importable path.
    spec = importlib.util.spec_from_file_location("conformance_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cm = _load()


@pytest.fixture(scope="module")
def suite() -> list[Any]:
    return list(cm.collect())


def _both(name: str, *tags: object, variant: tuple[str, ...] = ()) -> list[Any]:
    suffix = "-".join(variant)
    return [
        cm.Item(
            f"{X}::{name}[{mode}{'-' + suffix if suffix else ''}]", tags, None, (mode,), variant
        )
        for mode in ("sync", "async")
    ]


def _covering_everything() -> list[Any]:
    return [item for n in sorted(cm.REQUIRED) for item in _both(f"test_fr{n}", f"FR{n}")]


def _rows(matrix: str) -> list[list[str]]:
    return [r.split(" | ") for r in matrix.splitlines() if re.match(r"\| FR\d", r)]


# --- the real suite ---


def test_every_required_fr_has_a_tagged_test(suite: list[Any]) -> None:
    missing = cm.missing_required(suite)
    assert not missing, f"required FRs with no tagged conformance test: {', '.join(missing)}"


def test_every_tag_names_an_fr_in_the_prd_and_sits_on_every_variant(suite: list[Any]) -> None:
    wrong = cm.unknown_tags(suite) + cm.partial_tags(suite)
    assert not wrong, "tags naming no PRD FR, or on only some variants:\n" + "\n".join(wrong)


def test_every_test_runs_in_both_modes_or_says_why_not(suite: list[Any]) -> None:
    broken = cm.mode_violations(suite)
    assert not broken, "tests breaking the mode rule:\n" + "\n".join(broken)


def test_the_committed_matrix_is_current(suite: list[Any]) -> None:
    committed = cm.MATRIX.read_text(encoding="utf-8") if cm.MATRIX.exists() else None
    problem = cm.stale_message(committed, cm.render(suite))
    assert problem is None, problem


def test_the_collection_matches_the_committed_matrix_counts(suite: list[Any]) -> None:
    # Guards the checks above against passing on an empty or untagged collection.
    committed = cm.MATRIX.read_text(encoding="utf-8")
    counted = {row[0][2:]: int(row[3]) for row in _rows(committed)}
    collected = {f"FR{n}": sum(1 for t in suite if n in t.frs) for n in cm.FR_TITLES}
    assert collected == counted
    assert all(collected[f"FR{n}"] > 0 for n in cm.REQUIRED)
    names = {t.nodeid for t in suite}
    assert "tests/conformance/test_coverage_audit.py::test_the_committed_matrix_is_current" in names


# --- a small suite collected for real ---


def _mini(root: Path, body: str) -> Path:
    """Write one test module under a fresh directory and return the directory."""
    folder = root / uuid.uuid4().hex
    folder.mkdir()
    (folder / f"test_mini_{folder.name}.py").write_text(textwrap.dedent(body), encoding="utf-8")
    return folder


@pytest.fixture
def mini(tmp_path: Path) -> Any:
    before = set(sys.modules)
    yield lambda body: _mini(tmp_path, body)
    for name in set(sys.modules) - before:
        if name.startswith("test_mini_"):
            del sys.modules[name]


MINI_HEAD = """
    import pytest

    @pytest.fixture(params=["sync", "async"])
    def call(request):
        return request.param
"""

MINI = (
    MINI_HEAD
    + """
    @pytest.mark.fr("FR7")
    def test_both(call):
        pass

    @pytest.mark.fr
    def test_bare_tag(call):
        pass

    @pytest.mark.mode_independent(reason="makes no call")
    def test_reason_by_keyword():
        pass

    @pytest.mark.mode_independent
    def test_bare_marker():
        pass

    @pytest.mark.parametrize("x", [1, pytest.param(2, marks=pytest.mark.fr("FR9"))])
    @pytest.mark.fr("FR7")
    def test_tag_on_one_variant(call, x):
        pass

    @pytest.mark.parametrize("x", [1, pytest.param(2, marks=pytest.mark.mode_independent("why"))])
    def test_marker_on_one_variant(x):
        pass

    @pytest.mark.parametrize("how", ["close", "aclose"], ids=["sync", "async"])
    def test_relabelled(how):
        pass
"""
)


def test_a_collected_suite_is_judged_by_its_real_marks_and_parameters(mini: Any) -> None:
    tests = {t.nodeid.split("::")[1]: t for t in cm.collect(mini(MINI))}
    assert tests["test_both"].frs == {7}
    bare = [w for w in cm.unknown_tags(tests.values()) if "test_bare_tag" in w]
    assert bare == [f"{tests['test_bare_tag'].nodeid}: None"]
    assert tests["test_tag_on_one_variant"].frs == {7}
    assert cm.partial_tags(tests.values()) == [
        f"{tests['test_tag_on_one_variant'].nodeid}: FR9 is on only some variants"
    ]
    broken = {w.split("::")[1] for w in cm.mode_violations(tests.values())}
    assert broken == {
        "test_bare_marker: mode_independent needs a non-empty reason",
        "test_marker_on_one_variant: runs in one mode without mode_independent",
        "test_relabelled: runs in one mode without mode_independent",
    }


def test_a_mistyped_marker_fails_the_collection(mini: Any) -> None:
    body = MINI_HEAD + '\n    @pytest.mark.frr("FR7")\n    def test_typo(call):\n        pass\n'
    with pytest.raises(RuntimeError, match="collecting"):
        cm.collect(mini(body))


def test_main_exits_1_with_its_problems_and_0_on_a_sound_suite(
    mini: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    matrix = tmp_path / "matrix.md"
    assert cm.main(mini(MINI), matrix) == 1
    problems = [line for line in capsys.readouterr().err.splitlines() if line]
    assert problems and all(line.startswith("problem: ") for line in problems)
    assert any("FR1" in line for line in problems) and matrix.exists()

    tags = ", ".join(f'"FR{n}"' for n in sorted(cm.REQUIRED))
    sound = MINI_HEAD + f"\n    @pytest.mark.fr({tags})\n    def test_all(call):\n        pass\n"
    assert cm.main(mini(sound), matrix) == 0
    assert capsys.readouterr().err == ""
    assert "Required FRs covered: 28 of 28." in matrix.read_text(encoding="utf-8")


def test_the_callers_pytest_options_do_not_change_the_collection(
    mini: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = mini(MINI_HEAD + "\n    def test_one(call):\n        pass\n")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k nothing_matches")
    assert [t.nodeid.split("::")[1] for t in cm.collect(folder)] == ["test_one"]
    assert cm.os.environ["PYTEST_ADDOPTS"] == "-k nothing_matches"


# --- the checks on hand-made items ---


def test_a_required_fr_that_loses_its_tags_is_named() -> None:
    tests = cm.group(i for i in _covering_everything() if "FR12" not in i.fr_tags)
    assert cm.missing_required(tests) == ["FR12"]
    assert cm.missing_required(cm.group(_covering_everything())) == []


@pytest.mark.parametrize("tag", ["FR99", "FR0", "FR45", "fr7", "FR07", 7, None])
def test_a_tag_outside_the_prd_is_named_with_its_test(tag: object) -> None:
    tests = cm.group(_both("test_odd", "FR7", tag))
    assert cm.unknown_tags(tests) == [f"{X}::test_odd: {tag!r}"]


def test_every_prd_fr_is_a_known_tag() -> None:
    assert sorted(cm.FR_TITLES) == list(range(1, 45))
    assert cm.unknown_tags(cm.group(_both("test_new", "FR43", "FR44"))) == []


def test_a_single_mode_test_without_the_marker_is_named() -> None:
    items = [
        cm.Item(f"{X}::test_plain"),
        cm.Item(f"{X}::test_sync_only[sync]", (), None, ("sync",)),
        cm.Item(f"{X}::test_half[sync-openai]", (), None, ("sync",), ("provider=0",)),
        cm.Item(f"{X}::test_half[async-openai]", (), None, ("async",), ("provider=0",)),
        cm.Item(f"{X}::test_half[sync-anthropic]", (), None, ("sync",), ("provider=1",)),
        *_both("test_fine"),
    ]
    broken = cm.mode_violations(cm.group(items))
    assert {b.split("::")[1].split(":")[0] for b in broken} == {
        "test_plain",
        "test_sync_only",
        "test_half",
    }


def test_two_mode_fixtures_together_still_count_as_both_modes() -> None:
    items = [
        cm.Item(f"{X}::test_mixed[{a}-{b}]", (), None, tuple(sorted({a, b})))
        for a in ("sync", "async")
        for b in ("sync", "async")
    ]
    assert cm.mode_violations(cm.group(items)) == []


@pytest.mark.parametrize("reasons", [("",), ("   ",), (None,), ()])
def test_a_marker_without_a_reason_is_named(reasons: tuple[object, ...]) -> None:
    tests = cm.group([cm.Item(f"{X}::test_bare", (), reasons)])
    assert cm.mode_violations(tests) == [
        f"{X}::test_bare: mode_independent needs a non-empty reason"
    ]


def test_a_marked_test_that_runs_both_modes_is_named() -> None:
    items = [
        cm.Item(f"{X}::test_both[{mode}]", (), ("no call",), (mode,)) for mode in ("sync", "async")
    ]
    assert cm.mode_violations(cm.group(items)) == [
        f"{X}::test_both: marked mode_independent but runs in both modes"
    ]
    single = cm.group([cm.Item(f"{X}::test_one", (), ("no call",))])
    assert cm.mode_violations(single) == []


def test_a_stale_or_missing_matrix_names_the_regenerate_command() -> None:
    generated = cm.render(cm.group(_covering_everything()))
    for committed in (None, generated + "edited by hand\n", ""):
        problem = cm.stale_message(committed, generated)
        assert problem is not None and "uv run python scripts/conformance_matrix.py" in problem
    assert cm.stale_message(generated, generated) is None
    assert cm.stale_message(generated.replace("\n", "\r\n"), generated) is None


def test_the_matrix_is_the_same_whatever_order_tests_are_collected_in() -> None:
    items = _covering_everything() + _both("test_shared", "FR41", "FR3")
    shuffled = items[:]
    random.Random(7).shuffle(shuffled)
    assert cm.render(cm.group(items)) == cm.render(cm.group(shuffled))


def test_the_matrix_has_one_row_per_fr_with_its_tests_by_module() -> None:
    items = [
        *_both("test_b", "FR7"),
        *_both("test_a", "FR7", "FR9"),
        cm.Item("tests/conformance/test_y.py::test_c[sync]", ("FR7",), None, ("sync",)),
        cm.Item("tests/conformance/test_y.py::test_c[async]", ("FR7",), None, ("async",)),
    ]
    rows = _rows(cm.render(cm.group(items)))
    assert [r[0] for r in rows] == [f"| FR{n}" for n in range(1, 45)]
    assert rows[6][2:4] == ["yes", "3"]
    assert rows[6][4] == (f"`{X}`: `test_a`, `test_b`<br>`tests/conformance/test_y.py`: `test_c` |")
    assert rows[2][2:5] == ["no", "0", "none |"]


def test_untagged_tests_are_listed_by_module() -> None:
    items = [*_both("test_tagged", "FR7"), *_both("test_loose"), *_both("test_partial")]
    items.append(cm.Item(f"{X}::test_partial[extra]", ("FR7",), ("no call",)))
    matrix = cm.render(cm.group(items))
    untagged = matrix.split("## Untagged conformance tests", 1)[1]
    assert f"- `{X}`: `test_loose`, `test_partial`" in untagged
    assert "test_tagged" not in untagged
    assert "None." in cm.render(cm.group(_both("test_tagged", "FR7")))
