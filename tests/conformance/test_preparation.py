import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from test_attempt_chain import DATE, _json_ok
from test_events import Recorder
from test_retry_routing import _answer as worker_answer

from primagate import PrimaGate
from primagate.core.profile import PREP_PROMPT_VERSION, parse_profile
from primagate.drivers._shared import Prepared
from primagate.drivers.preparation import PreparationStopped
from primagate.schema import errors, events
from primagate.schema.profile import PROFILE_SCHEMA_VERSION, PrepState, Profile, ProfileSources
from primagate.stores.memory import InMemoryProfileStore

KEYS = {"openai": "sk-o", "anthropic": "sk-a"}
TWO = {"SMALL": {"model": "gpt-4o-mini"}, "MEDIUM": {"model": "gpt-4o"}}
POLICY: dict[str, Any] = {
    "enabled_tiers": ["SMALL", "MEDIUM"],
    "confidence_threshold": 0.8,
    "context_token_budget": 1000,
    "interim_binding": "MEDIUM",
    "ambiguous_context_default": "SMALL",
}
SECRET = "Only ever discuss the Zanzibar shipping ledger"
DECLARED = {"instructions": SECRET, "tool_descriptions": ["lookup_order: finds an order"]}
GOOD = {
    "purpose": "Answers questions about orders.",
    "interactions": ["order status questions"],
    "constraints": [],
    "capabilities": ["look up an order"],
    "unknowns": ["refund policy"],
}

Reply = str | dict[str, Any] | Callable[[], httpx2.Response]


def _reply(text: str) -> httpx2.Response:
    body = _json_ok("openai")
    body["choices"][0]["message"]["content"] = text
    body["usage"] = {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200}
    return httpx2.Response(200, json=body, headers=DATE)


def _status(code: int) -> Callable[[], httpx2.Response]:
    return lambda: httpx2.Response(code, json={"error": {"message": "x", "type": "server_error"}})


class Prep:
    """A gate whose mock answers preparation questions from a script and calls as a worker."""

    def __init__(
        self,
        mode: str,
        replies: list[Reply],
        policy: dict[str, Any] | None = None,
        analyzer: str | None = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        self.mode = mode
        self.replies = list(replies)
        self.prepared: list[dict[str, Any]] = []
        self.others: list[dict[str, Any]] = []
        self.handler = Recorder()
        self.gate = PrimaGate(
            api_keys=KEYS,
            transport=httpx2.MockTransport(self._handle),
            tier_bindings=TWO,
            policy=policy or POLICY,
            analyzer=analyzer,
            event_handler=self.handler,
            **kwargs,
        )

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        system = str(body.get("messages", [{}])[0].get("content", ""))
        if f"Prompt version: {PREP_PROMPT_VERSION}" in system:
            self.prepared.append(body)
            reply = self.replies.pop(0)
            if callable(reply):
                return reply()
            return _reply(reply if isinstance(reply, str) else json.dumps(reply))
        self.others.append(body)
        if "Prompt version: analyzer" in system:
            return _reply("SMALL 0.9")
        return worker_answer("openai", bool(body.get("stream")))

    @property
    def fingerprint(self) -> str:
        tiers = self.gate._tier_routing
        assert tiers is not None and tiers.fingerprint is not None
        return tiers.fingerprint

    @property
    def sources(self) -> ProfileSources:
        tiers = self.gate._tier_routing
        assert tiers is not None and tiers.sources is not None
        return tiers.sources

    def prepare(self) -> Prepared:
        if self.mode == "sync":
            return self.gate._get_driver().prepare(self.sources, self.fingerprint)
        driver = self.gate._get_async_driver()
        return asyncio.run(driver.prepare(self.sources, self.fingerprint))

    def call(self, **kwargs: Any) -> Any:
        if self.mode == "sync":
            return self.gate.completion(**kwargs)
        return asyncio.run(self.gate.acompletion(**kwargs))

    @property
    def stored(self) -> Any:
        return self.gate._profile_store.read()

    @property
    def types(self) -> list[str]:
        return self.handler.types

    def failures(self) -> list[tuple[str, int, bool]]:
        return [
            (e.reason, e.attempt, e.final)
            for e in self.handler.seen
            if isinstance(e, events.PreparationFailed)
        ]


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def _prep(mode: str, replies: list[Reply], **kwargs: Any) -> Prep:
    kwargs.setdefault("profile_sources", DECLARED)
    return Prep(mode, replies, **kwargs)


@pytest.mark.fr("FR23")
def test_a_valid_answer_is_stored_and_reported_ready(mode: str, delays: list[float]) -> None:
    run = _prep(mode, [GOOD])
    result = run.prepare()
    assert result.state is PrepState.READY and result.profile is not None
    assert run.stored == result.profile
    assert run.types == ["preparation_started", "profile_ready"]
    assert len(run.prepared) == 1 and delays == []
    sent = run.prepared[0]
    assert sent["model"] == "gpt-4o-mini" and sent["temperature"] == 0
    assert SECRET in sent["messages"][1]["content"]


@pytest.mark.fr("FR24")
def test_the_stored_profile_has_its_prep_id_and_utc_time(mode: str) -> None:
    run = _prep(mode, [GOOD])
    profile = run.prepare().profile
    assert profile is not None
    assert profile.profile_id.startswith("prep-") and len(profile.profile_id) > 5
    assert profile.input_fingerprint == run.fingerprint
    assert profile.schema_version == PROFILE_SCHEMA_VERSION
    assert profile.created_at.endswith("+00:00")
    assert profile.source_ref == f"{PREP_PROMPT_VERSION};gpt-4o-mini"
    # Written with the driver's real timestamp, it must read back under the stored-Profile rules.
    assert parse_profile(profile.to_dict()) == profile
    (ready,) = [e for e in run.handler.seen if isinstance(e, events.ProfileReady)]
    assert ready.profile_id == profile.profile_id


@pytest.mark.fr("FR23")
def test_every_preparation_event_has_a_prep_request_id(mode: str, delays: list[float]) -> None:
    run = _prep(mode, ["prose", GOOD])
    run.prepare()
    ids = {e.request_id for e in run.handler.seen}
    assert len(ids) == 1 and next(iter(ids)).startswith("prep-")
    assert all(e.timestamp for e in run.handler.seen)


@pytest.mark.fr("FR24")
def test_oversized_sources_fail_without_a_dispatch(mode: str) -> None:
    run = _prep(mode, [], profile_sources={"instructions": "word " * 20_000})
    result = run.prepare()
    assert result == Prepared(PrepState.FAILED, None)
    assert run.prepared == [] and run.stored is None
    assert run.failures() == [("overflow", 0, True)]


@pytest.mark.fr("FR23")
def test_a_malformed_answer_then_a_valid_one_is_ready(mode: str, delays: list[float]) -> None:
    run = _prep(mode, ["Here is your profile.", GOOD])
    assert run.prepare().state is PrepState.READY
    assert run.failures() == [("malformed_output", 1, False)]
    # The fixture's jitter is full scale, so the first wait is the top of its range.
    assert delays == [1.0]


@pytest.mark.fr("FR23")
def test_a_503_three_times_fails_with_no_write(mode: str, delays: list[float]) -> None:
    run = _prep(mode, [_status(503)] * 3)
    assert run.prepare() == Prepared(PrepState.FAILED, None)
    assert len(run.prepared) == 3 and delays == [1.0, 2.0] and run.stored is None
    assert run.failures() == [
        ("provider_error", 1, False),
        ("provider_error", 2, False),
        ("provider_error", 3, True),
    ]


@pytest.mark.fr("FR23")
def test_a_401_fails_after_one_attempt(mode: str, delays: list[float]) -> None:
    run = _prep(mode, [_status(401)])
    assert run.prepare().state is PrepState.FAILED
    assert len(run.prepared) == 1 and delays == [] and run.stored is None
    assert run.failures() == [("provider_error", 1, True)]


@pytest.mark.parametrize(
    "reply",
    [
        {k: v for k, v in GOOD.items() if k != "unknowns"},
        {"purpose": "", "interactions": [], "constraints": [], "capabilities": [], "unknowns": []},
    ],
    ids=["missing_unknowns", "empty_content"],
)
@pytest.mark.fr("FR24")
def test_content_the_profile_rules_refuse_is_malformed(
    mode: str, delays: list[float], reply: dict[str, Any]
) -> None:
    run = _prep(mode, [reply] * 3)
    assert run.prepare().state is PrepState.FAILED
    assert {reason for reason, _, _ in run.failures()} == {"malformed_output"}
    assert run.stored is None


@pytest.mark.fr("FR23")
def test_preparation_dispatches_even_when_the_cap_is_full(mode: str) -> None:
    run = _prep(mode, [GOOD], policy={**POLICY, "max_concurrent_analyzer_calls": 1})
    cap = run.gate._analyzer_cap
    assert cap is not None and cap.try_acquire()
    assert run.prepare().state is PrepState.READY
    assert len(run.prepared) == 1 and cap.in_flight == 1


@pytest.mark.fr("FR35")
def test_each_attempts_cost_is_counted_against_the_ceiling(mode: str, delays: list[float]) -> None:
    run = _prep(mode, ["prose", GOOD], policy={**POLICY, "spending_ceiling": 0.0001})
    run.prepare()
    one = (1000 * 0.15 + 200 * 0.60) / 1_000_000
    assert run.gate._spend.total == pytest.approx(2 * one)
    (warning,) = [e for e in run.handler.seen if isinstance(e, events.SpendingCeilingExceeded)]
    assert warning.request_id.startswith("prep-")


@pytest.mark.fr("FR38")
def test_no_event_or_log_line_holds_sources_or_the_reply(
    mode: str, delays: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG, logger="primagate")
    run = _prep(mode, ["Leaky prose reply", {**GOOD, "purpose": "Secret purpose"}])
    run.prepare()
    shown = json.dumps([e.to_dict() for e in run.handler.seen]) + caplog.text
    for text in (SECRET, "lookup_order", "Leaky prose", "Secret purpose"):
        assert text not in shown
    assert "preparation_failed" in caplog.text and "profile_ready" in caplog.text


@pytest.mark.fr("FR23")
def test_prepare_without_an_analyzer_is_rejected(mode: str) -> None:
    one_tier = {**POLICY, "enabled_tiers": ["MEDIUM"], "ambiguous_context_default": "MEDIUM"}
    run = Prep(mode, [], policy=one_tier, analyzer=None)
    with pytest.raises(errors.BadRequestError, match="analyzer"):
        if mode == "sync":
            run.gate._get_driver().prepare(ProfileSources("x"), "fp")
        else:
            asyncio.run(run.gate._get_async_driver().prepare(ProfileSources("x"), "fp"))
    assert run.prepared == [] and run.others == []


@pytest.mark.fr("FR23")
def test_a_preparation_on_a_closed_instance_stops_before_its_first_step(mode: str) -> None:
    run = _prep(mode, [GOOD])
    sync_driver, async_driver = run.gate._get_driver(), run.gate._get_async_driver()
    run.gate.close()
    # The same error type as every other use of a closed instance.
    assert issubclass(PreparationStopped, errors.PrimaGateError)
    with pytest.raises(errors.PrimaGateError, match="closed") as raised:
        if mode == "sync":
            sync_driver.prepare(run.sources, run.fingerprint)
        else:
            asyncio.run(async_driver.prepare(run.sources, run.fingerprint))
    assert isinstance(raised.value, PreparationStopped)
    assert run.prepared == [] and run.stored is None and run.types == []


@pytest.mark.fr("FR23")
@pytest.mark.mode_independent("drives the sync and the async driver in one body")
def test_a_driver_without_tier_routing_cannot_prepare() -> None:
    from primagate.drivers.async_ import AsyncDriver
    from primagate.drivers.sync import SyncDriver

    sync = SyncDriver(adapters={}, settings={}, transport=None)  # type: ignore[arg-type]
    with pytest.raises(errors.BadRequestError):
        sync.prepare(ProfileSources("x"), "fp")
    later = AsyncDriver(adapters={}, settings={}, transport=None)  # type: ignore[arg-type]
    with pytest.raises(errors.BadRequestError):
        asyncio.run(later.prepare(ProfileSources("x"), "fp"))


class Broken(Exception):
    pass


class FailingStore(InMemoryProfileStore):
    """A store whose write fails, as a full disk or a lost mount would make it."""

    def __init__(self, non_blocking: bool) -> None:
        super().__init__()
        self.non_blocking = non_blocking

    def write(self, profile: Profile) -> None:
        raise Broken("store is down")


@pytest.mark.parametrize("non_blocking", [True, False])
@pytest.mark.fr("FR23")
def test_a_store_error_reaches_the_caller_unchanged(mode: str, non_blocking: bool) -> None:
    run = _prep(mode, [GOOD], profile_store=FailingStore(non_blocking))
    with pytest.raises(Broken, match="store is down"):
        run.prepare()
    assert "preparation_started" in run.types and "profile_ready" not in run.types


@pytest.mark.fr("FR33")
def test_an_unpriced_analyzer_reports_its_cost_as_unknown(mode: str) -> None:
    run = _prep(mode, [GOOD], analyzer="unpriced-local-model")
    assert run.prepare().state is PrepState.READY
    (unknown,) = [e for e in run.handler.seen if isinstance(e, events.CostUnknown)]
    assert unknown.request_id.startswith("prep-")
    assert unknown.model == "openai/unpriced-local-model"
    assert run.types.index("cost_unknown") < run.types.index("profile_ready")


@pytest.mark.fr("FR33")
def test_a_priced_analyzer_reports_no_unknown_cost(mode: str) -> None:
    run = _prep(mode, [GOOD])
    run.prepare()
    assert "cost_unknown" not in run.types


def _context_exceeded() -> httpx2.Response:
    error = {"message": "too long", "type": "invalid_request_error"}
    return httpx2.Response(400, json={"error": {**error, "code": "context_length_exceeded"}})


@pytest.mark.fr("FR24")
def test_a_context_window_error_is_a_final_overflow(mode: str, delays: list[float]) -> None:
    run = _prep(mode, [_context_exceeded])
    assert run.prepare() == Prepared(PrepState.FAILED, None)
    assert len(run.prepared) == 1 and delays == [] and run.stored is None
    assert run.failures() == [("overflow", 1, True)]


@pytest.mark.fr("FR24")
def test_a_context_window_error_after_a_retry_names_its_attempt(
    mode: str, delays: list[float]
) -> None:
    run = _prep(mode, ["prose", _context_exceeded])
    assert run.prepare().state is PrepState.FAILED
    assert run.failures() == [("malformed_output", 1, False), ("overflow", 2, True)]
