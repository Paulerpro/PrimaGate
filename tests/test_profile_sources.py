import asyncio
import dataclasses
import json
from types import MappingProxyType
from typing import Any

import httpx2
import pytest
from conftest import Caller

from primagate import PrimaGate
from primagate.core.profile import PREP_PROMPT_VERSION, input_fingerprint
from primagate.schema import errors, events
from primagate.schema.policy import TIER_ROUTING_ALIAS, TierRouting
from primagate.schema.profile import ProfileSources

TWO: dict[str, Any] = {"SMALL": {"model": "gpt-4o-mini"}, "MEDIUM": {"model": "gpt-4o"}}
POLICY: dict[str, Any] = {
    "enabled_tiers": ["SMALL", "MEDIUM"],
    "confidence_threshold": 0.8,
    "context_token_budget": 1000,
    "interim_binding": "MEDIUM",
    "ambiguous_context_default": "SMALL",
}
SOURCES: dict[str, Any] = {
    "instructions": "Answer questions about orders.",
    "tool_descriptions": ["Look up an order by its number", "Cancel an order"],
    "skill_descriptions": ["Explain the refund policy"],
}
EXPECTED = ProfileSources(
    "Answer questions about orders.",
    ("Look up an order by its number", "Cancel an order"),
    ("Explain the refund policy",),
)


def _gate(
    sources: Any = SOURCES,
    *,
    analyzer: str | None = "gpt-4o-mini",
    bindings: Any = TWO,
    policy: Any = POLICY,
    **kwargs: Any,
) -> PrimaGate:
    return PrimaGate(
        tier_bindings=bindings,
        policy=policy,
        analyzer=analyzer,
        profile_sources=sources,
        **kwargs,
    )


def _routing(gate: PrimaGate) -> TierRouting:
    routing = gate._tier_routing
    assert routing is not None
    return routing


def _fingerprint(sources: Any = SOURCES, **kwargs: Any) -> str:
    fingerprint = _routing(_gate(sources, **kwargs)).fingerprint
    assert fingerprint is not None
    return fingerprint


def test_the_prep_prompt_version_is_named() -> None:
    assert PREP_PROMPT_VERSION == "prep-1"


def test_declared_sources_are_stored_and_fingerprinted_with_the_analyzer() -> None:
    routing = _routing(_gate())
    assert routing.sources == EXPECTED
    assert routing.fingerprint == input_fingerprint(EXPECTED, PREP_PROMPT_VERSION, "gpt-4o-mini")


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ({"instructions": "Be brief."}, ProfileSources(instructions="Be brief.")),
        ({"tool_descriptions": ["Look up"]}, ProfileSources(tool_descriptions=("Look up",))),
        ({"skill_descriptions": ("Refunds",)}, ProfileSources(skill_descriptions=("Refunds",))),
    ],
)
def test_every_key_is_optional(given: dict[str, Any], expected: ProfileSources) -> None:
    routing = _routing(_gate(given))
    assert routing.sources == expected
    assert routing.fingerprint == input_fingerprint(expected, PREP_PROMPT_VERSION, "gpt-4o-mini")


def test_editing_one_tool_description_changes_the_fingerprint() -> None:
    edited = {**SOURCES, "tool_descriptions": ["Look up an order by its id", "Cancel an order"]}
    assert _fingerprint(edited) != _fingerprint()


@pytest.mark.parametrize("field", ["instructions", "skill_descriptions"])
def test_editing_any_other_source_changes_the_fingerprint(field: str) -> None:
    value: Any = "Something else." if field == "instructions" else ["Something else"]
    assert _fingerprint({**SOURCES, field: value}) != _fingerprint()


def test_the_same_descriptions_in_another_order_keep_the_fingerprint() -> None:
    reordered = {
        **SOURCES,
        "tool_descriptions": list(reversed(SOURCES["tool_descriptions"])),
    }
    assert _fingerprint(reordered) == _fingerprint()


PRICED: dict[str, Any] = {
    "SMALL": {"model": "gpt-4o-mini", "pricing": {"input": 0.1, "output": 0.2}},
    "MEDIUM": {"model": "gpt-4o"},
}
LIMITED: dict[str, Any] = {
    "SMALL": {"model": "gpt-4o-mini", "limits": {"context_window": 1000}},
    "MEDIUM": {"model": "gpt-4o", "settings": {"temperature": 0.2}},
}


@pytest.mark.parametrize(
    "change",
    [
        {"price_overrides": {"gpt-4o": {"input": 1.0, "output": 2.0}}},
        {"bindings": PRICED},
        {"bindings": LIMITED},
        {"policy": {**POLICY, "confidence_threshold": 0.5, "spending_ceiling": 3.0}},
        {"cache_max_entries": 0},
    ],
)
def test_prices_bindings_the_policy_and_the_cache_leave_the_fingerprint_alone(
    change: dict[str, Any],
) -> None:
    assert _fingerprint(**change) == _fingerprint()


def test_another_analyzer_changes_the_fingerprint() -> None:
    assert _fingerprint(analyzer="gpt-4o") != _fingerprint()


def test_padding_around_the_analyzer_name_does_not_change_the_fingerprint() -> None:
    assert _fingerprint(analyzer="  gpt-4o-mini ") == _fingerprint()


@pytest.mark.parametrize(
    "sources",
    [
        None,
        {},
        {"instructions": "", "tool_descriptions": [], "skill_descriptions": ()},
        {"instructions": "   "},
    ],
)
def test_no_sources_or_only_empty_ones_leave_routing_as_before(sources: Any) -> None:
    routing = _routing(_gate(sources))
    assert routing.sources is None and routing.fingerprint is None


@pytest.mark.parametrize("sources", [None, {}, {"tool_descriptions": []}])
def test_empty_sources_need_no_analyzer(sources: Any) -> None:
    assert PrimaGate(profile_sources=sources)._tier_routing is None


ONE: dict[str, Any] = {
    "enabled_tiers": ["MEDIUM"],
    "confidence_threshold": 0.8,
    "context_token_budget": 1000,
    "interim_binding": "MEDIUM",
}


def test_empty_sources_with_a_one_tier_policy_are_accepted_as_none() -> None:
    routing = _routing(_gate({"instructions": ""}, policy=ONE, analyzer=None))
    assert routing.sources is None and routing.fingerprint is None


def test_blank_instructions_beside_a_description_fingerprint_as_if_omitted() -> None:
    tools = {"tool_descriptions": ["Look up an order"]}
    routing = _routing(_gate({**tools, "instructions": "  "}))
    assert routing.sources is not None and routing.sources.instructions == ""
    assert routing.fingerprint == _fingerprint(tools)


class _Text(str):
    pass


def test_a_read_only_mapping_and_a_text_subclass_are_accepted() -> None:
    given = MappingProxyType({**SOURCES, "instructions": _Text(SOURCES["instructions"])})
    routing = _routing(_gate(given))
    assert routing.sources == EXPECTED
    assert routing.fingerprint == _fingerprint()


def test_the_repr_never_shows_source_text() -> None:
    shown = repr(_routing(_gate()))
    for source in (
        SOURCES["instructions"],
        *SOURCES["tool_descriptions"],
        *SOURCES["skill_descriptions"],
    ):
        assert source not in shown


@pytest.mark.parametrize(
    "setup",
    [
        {"bindings": None, "policy": None, "analyzer": None},
        {"policy": ONE, "analyzer": None},
        # Only one Tier means the Analyzer is never asked, so it would never read a Profile.
        {"policy": ONE, "analyzer": "gpt-4o-mini"},
    ],
)
def test_sources_without_an_analyzer_to_read_them_are_refused(setup: dict[str, Any]) -> None:
    with pytest.raises(errors.BadRequestError, match="profile_sources"):
        _gate(**setup)


SECRET = "sk-secret-value"


@pytest.mark.parametrize(
    ("sources", "field"),
    [
        (SECRET, "profile_sources"),
        ([SECRET], "profile_sources"),
        ({"instructions": 3}, "instructions"),
        ({"instructions": None}, "instructions"),
        ({"instructions": [SECRET]}, "instructions"),
        ({"tool_descriptions": SECRET}, "tool_descriptions"),
        ({"tool_descriptions": SECRET.encode()}, "tool_descriptions"),
        ({"tool_descriptions": bytearray(SECRET.encode())}, "tool_descriptions"),
        ({"tool_descriptions": None}, "tool_descriptions"),
        ({"tool_descriptions": {SECRET: 1}}, "tool_descriptions"),
        ({"tool_descriptions": [SECRET, ""]}, "tool_descriptions"),
        ({"tool_descriptions": [SECRET, "  "]}, "tool_descriptions"),
        ({"skill_descriptions": [SECRET, 3]}, "skill_descriptions"),
        ({"skill_descriptions": iter([SECRET])}, "skill_descriptions"),
        (
            {"tool_descriptions": [{"type": "function", "function": {"name": SECRET}}]},
            "tool_descriptions",
        ),
        ({SECRET: ["x"]}, "profile_sources"),
        ({3: ["x"]}, "profile_sources"),
    ],
)
def test_a_bad_shape_is_refused_naming_the_field_but_never_its_content(
    sources: Any, field: str
) -> None:
    with pytest.raises(errors.BadRequestError, match=field) as caught:
        _gate(sources)
    assert SECRET not in str(caught.value)


def test_changing_the_callers_lists_after_setup_changes_nothing() -> None:
    tools = ["Look up an order by its number", "Cancel an order"]
    gate = _gate({**SOURCES, "tool_descriptions": tools})
    before = _routing(gate)
    tools.append("Delete every order")
    tools[0] = "Edited"
    after = _routing(gate)
    assert after.sources == EXPECTED and isinstance(after.sources.tool_descriptions, tuple)
    assert after.fingerprint == before.fingerprint == _fingerprint()


def test_the_sources_and_fingerprint_cannot_be_reassigned() -> None:
    routing = _routing(_gate())
    with pytest.raises(dataclasses.FrozenInstanceError):
        routing.fingerprint = "other"  # type: ignore[misc]
    assert routing.sources is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        routing.sources.instructions = "other"  # type: ignore[misc]


def test_both_drivers_are_handed_the_same_routing() -> None:
    gate = _gate()
    try:
        sync_routing = gate._get_driver()._tier_routing
        async_routing = gate._get_async_driver()._tier_routing
        assert sync_routing is async_routing is gate._tier_routing
    finally:
        asyncio.run(gate.aclose())


def _reply(text: str) -> httpx2.Response:
    body = {
        "id": "c",
        "created": 1,
        "model": "gpt-4o",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    return httpx2.Response(200, json=body, headers={"date": "Thu, 01 Jan 1970 00:00:01 GMT"})


def test_a_routed_call_still_uses_the_empty_profile_and_sends_no_source(call: Caller) -> None:
    sent: list[str] = []
    analyzer_prompts: list[str] = []
    seen: list[events.Event] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        text = request.content.decode()
        system = str(json.loads(text)["messages"][0].get("content", ""))
        if f"Prompt version: {PREP_PROMPT_VERSION}" in system:
            # The background preparation is the one request allowed to carry the sources.
            return httpx2.Response(401, json={"error": {"message": "x", "type": "auth"}})
        sent.append(text)
        if "Prompt version: analyzer" in system:
            analyzer_prompts.append(system)
            return _reply("MEDIUM 0.9")
        return _reply("ok")

    gate = _gate(
        api_keys={"openai": "sk-o"},
        transport=httpx2.MockTransport(handle),
        event_handler=seen.append,
    )
    pinned = gate._tier_routing
    call(gate, model=TIER_ROUTING_ALIAS, messages=[{"role": "user", "content": "where is it"}])
    runner = gate._runner
    assert runner is not None
    for thread in list(runner._threads):
        thread.join(10)

    assert gate._tier_routing is pinned
    (prompt,) = analyzer_prompts
    assert "Not known about the app" in prompt and "Purpose:" not in prompt
    for source in (
        SOURCES["instructions"],
        *SOURCES["tool_descriptions"],
        *SOURCES["skill_descriptions"],
    ):
        assert all(source not in body for body in sent)
    (decision,) = [e for e in seen if isinstance(e, events.RoutingDecision)]
    # Stays unknown until a prepared Profile is served.
    assert decision.input_fingerprint == "unknown"
