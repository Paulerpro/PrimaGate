import ast
from pathlib import Path

import primagate

ROOT = Path(primagate.__file__).parent

ALLOWED: dict[str, set[str]] = {
    "client": {"drivers", "schema"},
    "drivers": {"core", "adapters", "transport", "stores", "schema"},
    "stores": {"core", "schema"},
    "core": {"schema"},
    "adapters": {"schema"},
    "transport": {"schema"},
    "schema": set(),
}
FORBIDDEN_EXTERNAL = {"pydantic", "openai", "anthropic"}


def absolute_imports(source: str, package: list[str]) -> list[list[str]]:
    """Resolve every import in `source` to absolute dotted parts.

    `package` is the importing module's package as parts, e.g. ["primagate", "core"].
    `from . import x` yields the `x` submodule so sibling-layer imports are caught.
    """
    found: list[list[str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.extend(alias.name.split(".") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[: len(package) - (node.level - 1)]
            else:
                base = []
            module = base + (node.module.split(".") if node.module else [])
            if node.module is None or module == ["primagate"]:
                found.extend(module + [alias.name] for alias in node.names)
            else:
                found.append(module)
    return found


def external_violations(source: str) -> list[str]:
    return [
        parts[0]
        for parts in absolute_imports(source, ["primagate"])
        if parts[0] in FORBIDDEN_EXTERNAL
    ]


def layer_violations(layer: str, source: str, package: list[str]) -> list[str]:
    bad: list[str] = []
    for parts in absolute_imports(source, package):
        if parts[0] != "primagate" or len(parts) < 2:
            continue
        target = parts[1]
        if target != layer and target not in ALLOWED[layer]:
            bad.append(target)
    return bad


def _modules() -> list[tuple[str, Path, list[str]]]:
    out: list[tuple[str, Path, list[str]]] = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if len(rel.parts) == 1 and rel.name != "client.py":
            continue
        layer = rel.parts[0].removesuffix(".py")
        out.append((layer, path, ["primagate", *rel.parts[:-1]]))
    return out


def test_root_modules_are_known() -> None:
    # A root module outside this set would be skipped by the layer scan.
    root_modules = {p.name for p in ROOT.glob("*.py")}
    assert root_modules == {"__init__.py", "client.py"}


def test_layers_respect_allowed_import_directions() -> None:
    violations = [
        f"{path.relative_to(ROOT)} imports {target}"
        for layer, path, package in _modules()
        for target in layer_violations(layer, path.read_text(encoding="utf-8"), package)
    ]
    assert not violations, violations


def test_no_forbidden_external_imports() -> None:
    violations = [
        f"{path.relative_to(ROOT)} imports {name}"
        for path in sorted(ROOT.rglob("*.py"))
        for name in external_violations(path.read_text(encoding="utf-8"))
    ]
    assert not violations, violations


def test_detector_flags_forbidden_imports() -> None:
    pkg = ["primagate", "core"]
    assert layer_violations("core", "from primagate.drivers import x", pkg) == ["drivers"]
    assert layer_violations("core", "import primagate.adapters.openai", pkg) == ["adapters"]
    assert layer_violations("core", "from ..transport import t", pkg) == ["transport"]
    assert layer_violations("core", "from .. import stores", pkg) == ["stores"]
    assert layer_violations("core", "from primagate.schema.types import T", pkg) == []
    assert layer_violations("core", "from . import steps", pkg) == []
    assert layer_violations("core", "from primagate.utils import u", pkg) == ["utils"]
    assert layer_violations("client", "from primagate.core import x", ["primagate"]) == ["core"]
    assert layer_violations("client", "from primagate.drivers import d", ["primagate"]) == []
    low = ["primagate", "drivers"]
    assert layer_violations("drivers", "from primagate.client import c", low) == ["client"]


def test_detector_flags_forbidden_external_imports() -> None:
    for src in ("import pydantic", "from openai import x", "import anthropic.types"):
        assert external_violations(src) != [], src
    assert external_violations("import httpx2") == []
    assert external_violations("from primagate.schema import x") == []
