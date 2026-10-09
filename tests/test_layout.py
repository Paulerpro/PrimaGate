from pathlib import Path

import primagate

ROOT = Path(primagate.__file__).parent


def test_version_is_str() -> None:
    assert isinstance(primagate.__version__, str)


def test_layer_folders_exist() -> None:
    for layer in ("schema", "core", "adapters", "transport", "stores", "drivers"):
        assert (ROOT / layer / "__init__.py").is_file(), layer
    assert (ROOT / "client.py").is_file()


def test_typed_marker_present() -> None:
    assert (ROOT / "py.typed").is_file()
