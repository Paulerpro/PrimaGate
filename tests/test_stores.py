import dataclasses
import inspect
import threading
from typing import Protocol

import pytest

from primagate.core import ports
from primagate.core.ports import ModelRegistry, ProfileStore
from primagate.schema.errors import BadRequestError
from primagate.schema.profile import PROFILE_SCHEMA_VERSION, Profile, ProfileContent
from primagate.schema.tiers import FEATURES
from primagate.stores.memory import InMemoryModelRegistry, InMemoryProfileStore

CONTENT = ProfileContent(purpose="Answers order questions.", unknowns=())


def _profile(n: int, fingerprint: str = "fp-1") -> Profile:
    return Profile(
        profile_id=f"profile-{n}",
        input_fingerprint=fingerprint,
        schema_version=PROFILE_SCHEMA_VERSION,
        created_at="2026-11-17T10:00:00Z",
        content=CONTENT,
        source_ref="setup",
    )


@pytest.mark.parametrize("port", [ProfileStore, ModelRegistry])
def test_ports_are_synchronous_protocols(port: type) -> None:
    assert Protocol in port.__mro__ or getattr(port, "_is_protocol", False)
    for name, member in vars(port).items():
        if callable(member) and not name.startswith("_"):
            assert not inspect.iscoroutinefunction(member), name
    assert "non_blocking" in port.__annotations__


def test_ports_module_names_no_store_implementation() -> None:
    assert not any("InMemory" in name for name in vars(ports))


def test_in_memory_store_meets_the_port_and_never_blocks() -> None:
    store: ProfileStore = InMemoryProfileStore()
    assert store.non_blocking is True


def test_a_fresh_store_is_empty() -> None:
    assert InMemoryProfileStore().read() is None


def test_a_written_profile_reads_back_as_the_same_object() -> None:
    store, profile = InMemoryProfileStore(), _profile(1)
    store.write(profile)
    assert store.read() is profile


def test_a_later_write_replaces_the_one_slot() -> None:
    store, first, second = InMemoryProfileStore(), _profile(1), _profile(2, "fp-2")
    store.write(first)
    store.write(second)
    assert store.read() is second


def test_two_stores_never_share_a_slot() -> None:
    one, two = InMemoryProfileStore(), InMemoryProfileStore()
    one.write(_profile(1))
    assert two.read() is None


def test_concurrent_readers_see_none_or_one_whole_written_profile() -> None:
    store = InMemoryProfileStore()
    written = [_profile(n, f"fp-{n}") for n in range(50)]
    seen: list[Profile | None] = []
    start = threading.Barrier(8)

    def writer(chunk: list[Profile]) -> None:
        start.wait()
        for profile in chunk:
            store.write(profile)

    def reader() -> None:
        start.wait()
        for _ in range(2000):
            seen.append(store.read())

    threads = [threading.Thread(target=writer, args=(written[i::4],)) for i in range(4)]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    ids = {id(p) for p in written}
    assert all(p is None or id(p) in ids for p in seen)
    assert store.read() in written


def test_registry_meets_the_port_and_never_blocks() -> None:
    registry: ModelRegistry = InMemoryModelRegistry({})
    assert registry.non_blocking is True


def test_registry_answers_a_known_model_and_misses_an_unknown_one() -> None:
    registry = InMemoryModelRegistry({("openai", "gpt-4o"): {"tools", "images"}})
    assert registry.supported_features("openai", "gpt-4o") == frozenset({"tools", "images"})
    assert registry.supported_features("openai", "gpt-4o-mini") is None
    assert registry.supported_features("anthropic", "gpt-4o") is None


def test_registry_accepts_every_known_feature_and_an_empty_set() -> None:
    registry = InMemoryModelRegistry({("a", "all"): FEATURES, ("a", "none"): ()})
    assert registry.supported_features("a", "all") == frozenset(FEATURES)
    assert registry.supported_features("a", "none") == frozenset()


@pytest.mark.parametrize(
    "features",
    [
        {"telepathy"},
        {"tools", "telepathy"},
        "tools",
        b"tools",
        None,
        5,
        {1},
        {"tools": False},
    ],
)
def test_registry_rejects_an_unknown_feature_at_build(features: object) -> None:
    with pytest.raises(BadRequestError):
        InMemoryModelRegistry({("openai", "gpt-4o"): features})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    "key",
    [
        ("openai",),
        ("openai", "gpt-4o", "extra"),
        "openai/gpt-4o",
        ("openai", ""),
        ("", "m"),
        ("openai", 4),
        (1, 2),
    ],
)
def test_registry_rejects_a_key_that_is_not_provider_and_model(key: object) -> None:
    with pytest.raises(BadRequestError):
        InMemoryModelRegistry({key: {"tools"}})  # type: ignore[dict-item]


def test_registry_rejects_a_non_mapping() -> None:
    with pytest.raises(BadRequestError):
        InMemoryModelRegistry([(("openai", "gpt-4o"), {"tools"})])  # type: ignore[arg-type]


def test_registry_keeps_its_own_read_only_copy() -> None:
    features = {"tools"}
    source = {("openai", "gpt-4o"): features}
    registry = InMemoryModelRegistry(source)
    features.add("images")
    source[("openai", "gpt-4o-mini")] = {"tools"}
    assert registry.supported_features("openai", "gpt-4o") == frozenset({"tools"})
    assert registry.supported_features("openai", "gpt-4o-mini") is None


def test_a_stored_profile_cannot_be_changed_in_place() -> None:
    store, profile = InMemoryProfileStore(), _profile(1)
    store.write(profile)
    with pytest.raises(dataclasses.FrozenInstanceError):
        store.read().profile_id = "other"  # type: ignore[union-attr, misc]
