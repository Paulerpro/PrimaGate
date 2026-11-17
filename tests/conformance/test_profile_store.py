import asyncio
import threading
from collections.abc import Iterator
from typing import Any

import pytest

import primagate
from primagate import PrimaGate
from primagate.core.call import advance
from primagate.core.profile import read_profile, write_profile
from primagate.core.steps import Action, Raise
from primagate.drivers.async_ import AsyncDriver
from primagate.drivers.sync import SyncDriver
from primagate.schema.errors import PrimaGateError
from primagate.schema.profile import PROFILE_SCHEMA_VERSION, Profile, ProfileContent
from primagate.stores.memory import InMemoryProfileStore

FP = "fp-current"
PROFILE = Profile(
    profile_id="profile-1",
    input_fingerprint=FP,
    schema_version=PROFILE_SCHEMA_VERSION,
    created_at="2026-11-17T10:00:00Z",
    content=ProfileContent(purpose="Answers order questions.", unknowns=()),
    source_ref="setup",
)
_NEXT = Raise(PrimaGateError("next"))


class Boom(Exception):
    pass


class RecordingStore:
    """A store that notes which thread ran it, and can be told to fail."""

    def __init__(self, non_blocking: bool, fail: bool = False) -> None:
        self.non_blocking = non_blocking
        self.fail = fail
        self.threads: list[int] = []
        self.slot: Profile | None = None

    def read(self) -> Profile | None:
        self.threads.append(threading.get_ident())
        if self.fail:
            raise Boom("read")
        return self.slot

    def write(self, profile: Profile) -> None:
        self.threads.append(threading.get_ident())
        if self.fail:
            raise Boom("write")
        self.slot = profile


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _driver(mode: str, store: Any = None) -> SyncDriver | AsyncDriver:
    cls = SyncDriver if mode == "sync" else AsyncDriver
    if store is None:
        return cls(adapters={}, settings={}, transport=None)  # type: ignore[arg-type]
    return cls(adapters={}, settings={}, transport=None, profile_store=store)  # type: ignore[arg-type]


def _step(driver: SyncDriver | AsyncDriver, action: Action) -> Action:
    """Perform one action and return what core says comes next."""
    if isinstance(driver, SyncDriver):
        outcome = driver.perform(action)
    else:
        outcome = asyncio.run(driver.perform(action))
    return advance(action, outcome)  # type: ignore[arg-type]


def _read(driver: SyncDriver | AsyncDriver, fingerprint: str = FP) -> list[Profile | None]:
    got: list[Profile | None] = []

    def then(found: Profile | None) -> Action:
        got.append(found)
        return _NEXT

    assert _step(driver, read_profile(fingerprint, then)) is _NEXT
    return got


def _write(driver: SyncDriver | AsyncDriver, profile: Profile) -> None:
    assert _step(driver, write_profile(profile, _NEXT)) is _NEXT


def test_a_driver_built_without_a_store_starts_with_an_empty_one(mode: str) -> None:
    assert _read(_driver(mode)) == [None]


def test_a_written_profile_is_read_back_through_the_driver(mode: str) -> None:
    driver = _driver(mode, InMemoryProfileStore())
    _write(driver, PROFILE)
    assert _read(driver) == [PROFILE]


def test_a_later_write_replaces_the_earlier_one(mode: str) -> None:
    driver = _driver(mode, InMemoryProfileStore())
    second = Profile(**{**PROFILE.__dict__, "profile_id": "profile-2"})
    _write(driver, PROFILE)
    _write(driver, second)
    assert _read(driver)[0] is second


def test_a_stale_or_old_schema_profile_reaches_core_as_a_miss(mode: str) -> None:
    store = InMemoryProfileStore()
    driver = _driver(mode, store)
    _write(driver, PROFILE)
    assert _read(driver, "fp-moved") == [None]
    for version in (PROFILE_SCHEMA_VERSION + 1, PROFILE_SCHEMA_VERSION - 1):
        old = Profile(**{**PROFILE.__dict__, "schema_version": version})
        _write(driver, old)
        assert _read(driver) == [None]
        assert store.read() is old


@pytest.mark.parametrize("non_blocking", [True, False])
def test_the_sync_driver_runs_any_store_on_the_calling_thread(non_blocking: bool) -> None:
    store = RecordingStore(non_blocking)
    driver = _driver("sync", store)
    _write(driver, PROFILE)
    _read(driver)
    assert store.threads == [threading.get_ident()] * 2


def _async_threads(store: RecordingStore) -> tuple[int, list[int]]:
    driver = AsyncDriver(adapters={}, settings={}, transport=None, profile_store=store)  # type: ignore[arg-type]

    async def go() -> int:
        for action in (write_profile(PROFILE, _NEXT), read_profile(FP, lambda _: _NEXT)):
            advance(action, await driver.perform(action))  # type: ignore[arg-type]
        return threading.get_ident()

    loop_thread = asyncio.run(go())
    return loop_thread, store.threads


def test_the_async_driver_runs_a_non_blocking_store_on_the_loop_thread() -> None:
    loop_thread, ran_on = _async_threads(RecordingStore(non_blocking=True))
    assert ran_on == [loop_thread] * 2


def test_the_async_driver_runs_a_blocking_store_in_a_worker_thread() -> None:
    loop_thread, ran_on = _async_threads(RecordingStore(non_blocking=False))
    assert len(ran_on) == 2
    assert loop_thread not in ran_on


@pytest.mark.parametrize("non_blocking", [True, False])
def test_a_store_error_propagates_unchanged(mode: str, non_blocking: bool) -> None:
    driver = _driver(mode, RecordingStore(non_blocking, fail=True))
    with pytest.raises(Boom, match="read"):
        _read(driver)
    with pytest.raises(Boom, match="write"):
        _write(driver, PROFILE)


@pytest.fixture
def gates() -> Iterator[list[PrimaGate]]:
    made: list[PrimaGate] = []
    yield made
    for gate in made:
        asyncio.run(gate.aclose())


def _gate(gates: list[PrimaGate], **kwargs: Any) -> PrimaGate:
    gate = PrimaGate(**kwargs)
    gates.append(gate)
    return gate


def _driver_of(gate: PrimaGate, mode: str) -> SyncDriver | AsyncDriver:
    return gate._get_driver() if mode == "sync" else gate._get_async_driver()


def test_a_new_instance_starts_with_an_empty_store(gates: list[PrimaGate], mode: str) -> None:
    before = _gate(gates)
    _write(_driver_of(before, mode), PROFILE)
    assert _read(_driver_of(before, mode)) == [PROFILE]
    after = _gate(gates)
    assert _read(_driver_of(after, mode)) == [None]


def test_one_instance_shares_its_store_between_sync_and_async(gates: list[PrimaGate]) -> None:
    gate = _gate(gates)
    _write(gate._get_driver(), PROFILE)
    assert _read(gate._get_async_driver()) == [PROFILE]
    second = Profile(**{**PROFILE.__dict__, "profile_id": "p2"})
    _write(gate._get_async_driver(), second)
    assert _read(gate._get_driver())[0] is second


def test_a_caller_store_serves_both_modes(gates: list[PrimaGate]) -> None:
    store = RecordingStore(non_blocking=True)
    gate = _gate(gates, profile_store=store)
    _write(gate._get_driver(), PROFILE)
    assert store.slot is PROFILE
    assert _read(gate._get_async_driver()) == [PROFILE]
    assert len(store.threads) == 2


def test_the_store_and_registry_add_no_public_name() -> None:
    for name in (
        "ProfileStore",
        "ModelRegistry",
        "InMemoryProfileStore",
        "InMemoryModelRegistry",
        "default_profile_store",
    ):
        assert not hasattr(primagate, name), name
