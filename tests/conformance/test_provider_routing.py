from typing import Any

import httpx2
import pytest
from conftest import Caller

from primagate import PrimaGate
from primagate.schema import errors

MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5-5"}
MSGS = [{"role": "user", "content": "hi"}]
KEYS = {"openai": "sk-o", "anthropic": "sk-a"}
SECOND = [{"base_url": "https://b.test/v1", "api_key": "sk-second"}]


def _gate(**kwargs: Any) -> tuple[PrimaGate, list[httpx2.Request]]:
    sent: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        body = {"error": {"message": "overloaded", "type": "server_error"}}
        return httpx2.Response(503, json=body)

    gate = PrimaGate(api_keys=KEYS, transport=httpx2.MockTransport(handle), **kwargs)
    return gate, sent


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.parametrize("routing", [None, {}], ids=["omitted", "empty"])
def test_with_routing_off_a_failing_call_makes_exactly_one_request(
    call: Caller, provider: str, routing: Any
) -> None:
    gate, sent = _gate(provider_routing=routing)
    with pytest.raises(errors.ServiceUnavailableError) as raised:
        call(gate, model=MODELS[provider], messages=MSGS)
    assert len(sent) == 1
    assert len(raised.value.attempt_chain) == 1
    assert len(raised.value.attempt_chain[0].dispatches) == 1


@pytest.mark.parametrize("provider", MODELS)
def test_a_declaration_alone_does_not_add_requests(call: Caller, provider: str) -> None:
    gate, sent = _gate(provider_routing={MODELS[provider]: SECOND})
    with pytest.raises(errors.ServiceUnavailableError):
        call(gate, model=MODELS[provider], messages=MSGS)
    assert len(sent) == 1
    assert sent[0].url.host != "b.test"


@pytest.mark.parametrize("provider", MODELS)
def test_setup_rejects_routing_on_without_an_endpoint(provider: str) -> None:
    with pytest.raises(errors.BadRequestError, match=MODELS[provider]):
        PrimaGate(api_keys=KEYS, provider_routing={MODELS[provider]: []})
