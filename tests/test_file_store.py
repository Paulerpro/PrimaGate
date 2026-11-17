import json
import os
import stat
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import primagate
from primagate.core.ports import ProfileStore
from primagate.schema.errors import BadRequestError
from primagate.schema.profile import PROFILE_SCHEMA_VERSION, Profile, ProfileContent
from primagate.stores.file import FileProfileStore

DOCS = Path(__file__).parents[1] / "docs" / "profile-store.md"
README = Path(__file__).parents[1] / "README.md"


def _profile(n: int, fingerprint: str = "fp-1") -> Profile:
    return Profile(
        profile_id=f"profile-{n}",
        input_fingerprint=fingerprint,
        schema_version=PROFILE_SCHEMA_VERSION,
        created_at="2026-11-17T10:00:00Z",
        content=ProfileContent(
            purpose="Answers order questions, en français aussi.",
            interactions=("chat",),
            capabilities=("lookup_order: finds an order",),
            unknowns=("refund policy",),
        ),
        source_ref="setup",
    )


def _leftovers(directory: Path, keep: str) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name != keep)


def test_it_meets_the_port_and_declares_itself_blocking(tmp_path: Path) -> None:
    store: ProfileStore = FileProfileStore(tmp_path / "profile.json")
    assert store.non_blocking is False


def test_a_missing_file_is_a_miss(tmp_path: Path) -> None:
    assert FileProfileStore(tmp_path / "profile.json").read() is None


def test_a_written_profile_reads_back_equal(tmp_path: Path) -> None:
    store, profile = FileProfileStore(tmp_path / "profile.json"), _profile(1)
    store.write(profile)
    assert store.read() == profile


def test_a_new_store_on_the_same_file_reads_the_profile_back(tmp_path: Path) -> None:
    FileProfileStore(tmp_path / "profile.json").write(_profile(1))
    assert FileProfileStore(str(tmp_path / "profile.json")).read() == _profile(1)


def test_the_file_is_exactly_the_sorted_indented_profile_dict(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    FileProfileStore(path).write(_profile(1))
    data = path.read_bytes()
    assert json.loads(data.decode("utf-8")) == _profile(1).to_dict()
    text = data.decode("utf-8")
    assert text == json.dumps(_profile(1).to_dict(), sort_keys=True, indent=2) + "\n"


def test_a_later_write_replaces_the_earlier_one(tmp_path: Path) -> None:
    store = FileProfileStore(tmp_path / "profile.json")
    store.write(_profile(1))
    store.write(_profile(2, "fp-2"))
    assert store.read() == _profile(2, "fp-2")
    assert _leftovers(tmp_path, "profile.json") == []


def test_a_profile_of_another_schema_is_read_as_written(tmp_path: Path) -> None:
    # Whether it still fits is core's call, so the store hands it over unjudged.
    store = FileProfileStore(tmp_path / "profile.json")
    old = Profile(**{**_profile(1).__dict__, "schema_version": PROFILE_SCHEMA_VERSION + 1})
    store.write(old)
    assert store.read() == old


def test_text_with_a_lone_surrogate_round_trips(tmp_path: Path) -> None:
    store = FileProfileStore(tmp_path / "profile.json")
    odd = Profile(**{**_profile(1).__dict__, "source_ref": "setup \ud800"})
    store.write(odd)
    assert store.read() == odd


def test_a_relative_path_is_fixed_when_the_store_is_built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home, elsewhere = tmp_path / "home", tmp_path / "elsewhere"
    home.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(home)
    store = FileProfileStore("profile.json")
    monkeypatch.chdir(elsewhere)
    store.write(_profile(1))
    assert store.read() == _profile(1)
    assert (home / "profile.json").is_file()
    assert list(elsewhere.iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
def test_every_write_leaves_the_file_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    store = FileProfileStore(path)
    store.write(_profile(1))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    path.chmod(0o644)
    store.write(_profile(2, "fp-2"))
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def _unknown_field() -> bytes:
    raw = _profile(1).to_dict()
    raw["extra"] = "x"
    return json.dumps(raw).encode("utf-8")


def _missing_unknowns() -> bytes:
    raw = _profile(1).to_dict()
    del raw["content"]["unknowns"]
    return json.dumps(raw).encode("utf-8")


CORRUPT: dict[str, bytes] = {
    "empty": b"",
    "truncated": json.dumps(_profile(1).to_dict()).encode("utf-8")[:40],
    "invalid utf-8": b'{"profile_id": "\xff\xfe"}',
    "byte order mark": b"\xef\xbb\xbf" + json.dumps(_profile(1).to_dict()).encode("utf-8"),
    "not a mapping": b"[1, 2, 3]",
    "unknown field": _unknown_field(),
    "missing unknowns": _missing_unknowns(),
    "deeply nested": b"[" * 100_000 + b"]" * 100_000,
    "huge number": b'{"schema_version": ' + b"9" * 10_000 + b"}",
}


@pytest.mark.parametrize("content", CORRUPT.values(), ids=CORRUPT.keys())
def test_unreadable_content_is_a_miss(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "profile.json"
    path.write_bytes(content)
    assert FileProfileStore(path).read() is None


@pytest.mark.parametrize("content", CORRUPT.values(), ids=CORRUPT.keys())
def test_a_write_after_a_miss_overwrites_it(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "profile.json"
    path.write_bytes(content)
    store = FileProfileStore(path)
    store.write(_profile(1))
    assert store.read() == _profile(1)


def test_a_missing_parent_directory_is_created(tmp_path: Path) -> None:
    path = tmp_path / "dir" / "sub" / "profile.json"
    store = FileProfileStore(path)
    assert store.read() is None
    store.write(_profile(1))
    assert store.read() == _profile(1)


def test_a_directory_at_the_path_is_an_error_not_a_miss(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    path.mkdir()
    store = FileProfileStore(path)
    with pytest.raises(OSError):
        store.read()
    with pytest.raises(OSError):
        store.write(_profile(1))
    assert _leftovers(tmp_path, "profile.json") == []


def _seed(tmp_path: Path) -> tuple[FileProfileStore, Path, bytes]:
    path = tmp_path / "profile.json"
    store = FileProfileStore(path)
    store.write(_profile(1))
    return store, path, path.read_bytes()


class Interrupted(Exception):
    pass


def test_a_failed_replace_leaves_the_old_file_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, path, before = _seed(tmp_path)

    def fail(*args: Any) -> None:
        raise Interrupted("replace")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(Interrupted, match="replace"):
        store.write(_profile(2, "fp-2"))
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert _leftovers(tmp_path, "profile.json") == []
    assert store.read() == _profile(1)


def test_a_write_cut_off_midway_leaves_the_old_file_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, path, before = _seed(tmp_path)
    real_fdopen = os.fdopen

    class HalfWriter:
        def __init__(self, fd: int, *args: Any, **kwargs: Any) -> None:
            self._file = real_fdopen(fd, *args, **kwargs)

        def __enter__(self) -> "HalfWriter":
            return self

        def __exit__(self, *exc: object) -> None:
            self._file.close()

        def write(self, data: bytes) -> int:
            self._file.write(data[: len(data) // 2])
            self._file.flush()
            raise Interrupted("write")

        def __getattr__(self, name: str) -> Any:
            return getattr(self._file, name)

    monkeypatch.setattr(os, "fdopen", HalfWriter)
    with pytest.raises(Interrupted, match="write"):
        store.write(_profile(2, "fp-2"))
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert _leftovers(tmp_path, "profile.json") == []


def test_a_failed_fsync_leaves_the_old_file_and_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, path, before = _seed(tmp_path)

    def fail(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="disk full"):
        store.write(_profile(2, "fp-2"))
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert _leftovers(tmp_path, "profile.json") == []


def test_an_interrupt_during_the_write_still_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, path, before = _seed(tmp_path)

    def fail(*args: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(KeyboardInterrupt):
        store.write(_profile(2, "fp-2"))
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert _leftovers(tmp_path, "profile.json") == []


def test_threads_reading_and_writing_one_store_see_whole_profiles(tmp_path: Path) -> None:
    store = FileProfileStore(tmp_path / "profile.json")
    written = [_profile(n, f"fp-{n}") for n in range(40)]
    seen: list[Profile | None] = []
    start = threading.Barrier(8)
    errors: list[BaseException] = []

    def guarded(work: Any) -> Any:
        def run(*args: Any) -> None:
            try:
                start.wait()
                work(*args)
            except BaseException as exc:
                errors.append(exc)

        return run

    def writer(chunk: list[Profile]) -> None:
        for profile in chunk:
            store.write(profile)

    def reader() -> None:
        for _ in range(100):
            seen.append(store.read())

    threads = [threading.Thread(target=guarded(writer), args=(written[i::4],)) for i in range(4)]
    threads += [threading.Thread(target=guarded(reader)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert all(p is None or p in written for p in seen)
    assert store.read() in written
    assert _leftovers(tmp_path, "profile.json") == []


class PathOf:
    def __init__(self, value: object) -> None:
        self._value = value

    def __fspath__(self) -> Any:
        return self._value


def test_a_path_like_object_is_accepted(tmp_path: Path) -> None:
    store = FileProfileStore(PathOf(str(tmp_path / "profile.json")))
    store.write(_profile(1))
    assert FileProfileStore(tmp_path / "profile.json").read() == _profile(1)


@pytest.mark.parametrize(
    "path",
    ["", None, 5, b"profile.json", PathOf(""), PathOf(b"profile.json"), PathOf(5)],
    ids=["empty", "none", "int", "bytes", "empty-pathlike", "bytes-pathlike", "int-pathlike"],
)
def test_a_bad_path_is_rejected_at_construction(path: object) -> None:
    with pytest.raises(BadRequestError):
        FileProfileStore(path)  # type: ignore[arg-type]


def test_the_file_store_is_the_one_public_store_name() -> None:
    assert primagate.FileProfileStore is FileProfileStore
    assert "FileProfileStore" in primagate.__all__


def test_the_docs_cover_format_versioning_and_several_processes() -> None:
    docs = DOCS.read_text(encoding="utf-8")
    for phrase in (
        "Profile.to_dict()",
        "schema_version",
        "os.replace",
        "power loss",
        "Last writer wins",
        "one writer per file",
        "Windows",
    ):
        assert phrase in docs, phrase
    assert "docs/profile-store.md" in README.read_text(encoding="utf-8")
