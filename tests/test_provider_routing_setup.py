from typing import Any

import pytest

from primagate import PrimaGate
from primagate.schema import errors
from primagate.schema.wire import Endpoint

URL = "https://b.test/v1"
SECRET = "sk-second-account-9"


def _routing(value: Any) -> dict[tuple[str, str], tuple[Endpoint, ...]]:
    return PrimaGate(provider_routing=value)._provider_routing


def test_endpoints_are_stored_in_declared_order() -> None:
    stored = _routing(
        {
            "gpt-4o": [{"base_url": URL}, {"base_url": "https://c.test/v1", "api_key": SECRET}],
            "claude-sonnet-5-5": [{"base_url": "https://d.test"}],
        }
    )
    assert stored[("openai", "gpt-4o")] == (
        Endpoint(URL),
        Endpoint("https://c.test/v1", SECRET),
    )
    assert stored[("anthropic", "claude-sonnet-5-5")] == (Endpoint("https://d.test"),)


@pytest.mark.parametrize("value", [None, {}])
def test_routing_is_off_without_a_declaration(value: Any) -> None:
    assert _routing(value) == {}
    assert PrimaGate()._provider_routing == {}


def test_a_provider_prefix_names_the_same_model() -> None:
    assert ("openai", "gpt-4o") in _routing({"openai/gpt-4o": [{"base_url": URL}]})
    with pytest.raises(errors.BadRequestError, match="gpt-4o"):
        _routing({"gpt-4o": [{"base_url": URL}], "openai/gpt-4o": [{"base_url": URL}]})


def test_routing_on_with_no_endpoint_names_the_model() -> None:
    with pytest.raises(errors.BadRequestError, match="gpt-4o"):
        _routing({"gpt-4o": []})


@pytest.mark.parametrize(
    "endpoint",
    [
        {},
        {"base_url": ""},
        {"base_url": 5},
        {"base_url": URL, "api_key": 5},
        {"base_url": URL, "timeout": 3},
        URL,
        None,
    ],
)
def test_a_malformed_endpoint_is_rejected(endpoint: Any) -> None:
    with pytest.raises(errors.BadRequestError, match="gpt-4o"):
        _routing({"gpt-4o": [endpoint]})


@pytest.mark.parametrize(
    "value",
    [[("gpt-4o", [])], "gpt-4o", {"gpt-4o": URL}, {"gpt-4o": None}, {5: [{"base_url": URL}]}],
)
def test_the_declaration_must_be_a_map_of_endpoint_lists(value: Any) -> None:
    with pytest.raises(errors.BadRequestError):
        _routing(value)


def test_an_unknown_provider_prefix_is_rejected() -> None:
    with pytest.raises(errors.BadRequestError, match="nope"):
        _routing({"nope/x": [{"base_url": URL}]})


def test_the_stored_declaration_cannot_be_changed_afterwards() -> None:
    stored = _routing({"gpt-4o": [{"base_url": URL}]})
    with pytest.raises(TypeError):
        stored[("openai", "other")] = ()  # type: ignore[index]


def test_the_key_stays_out_of_repr_and_setup_errors() -> None:
    gate = PrimaGate(provider_routing={"gpt-4o": [{"base_url": URL, "api_key": SECRET}]})
    assert SECRET not in repr(gate) and SECRET not in repr(gate._provider_routing)
    bad = {"gpt-4o": [{"base_url": URL, "api_key": SECRET, "timeout": 3}]}
    with pytest.raises(errors.BadRequestError) as raised:
        _routing(bad)
    assert SECRET not in str(raised.value)
    wrong_type = {"gpt-4o": [{"base_url": 5, "api_key": SECRET}]}
    with pytest.raises(errors.BadRequestError) as raised:
        _routing(wrong_type)
    assert SECRET not in str(raised.value)
