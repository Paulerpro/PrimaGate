from typing import Any

import httpx2
import pytest
from conftest import Caller

from primagate import PrimaGate
from primagate.schema import errors

MODELS = {"openai": "gpt-4o", "anthropic": "claude-sonnet-5-5"}
MSGS = [{"role": "user", "content": "hi"}]
KEYS = {"openai": "sk-o", "anthropic": "sk-a"}


def _failing(**kwargs: Any) -> tuple[PrimaGate, list[httpx2.Request]]:
    sent: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        body = {"error": {"message": "overloaded", "type": "server_error"}}
        return httpx2.Response(503, json=body)

    return PrimaGate(api_keys=KEYS, transport=httpx2.MockTransport(handle), **kwargs), sent


@pytest.mark.parametrize("provider", MODELS)
@pytest.mark.parametrize(
    "declared",
    [
        {},
        {"fallbacks": None, "default_fallbacks": None},
        {"fallbacks": {}, "default_fallbacks": []},
    ],
    ids=["omitted", "none", "empty"],
)
@pytest.mark.fr("FR8")
def test_no_fallback_list_is_the_same_as_an_empty_one(
    call: Caller, provider: str, declared: dict[str, Any]
) -> None:
    gate, sent = _failing(**declared)
    with pytest.raises(errors.ServiceUnavailableError) as raised:
        call(gate, model=MODELS[provider], messages=MSGS)
    assert len(sent) == 1
    assert len(raised.value.attempt_chain) == 1


@pytest.mark.fr("FR7")
@pytest.mark.mode_independent("setup validation runs when the instance is made, before any call")
def test_setup_rejects_a_bad_declaration_before_any_call() -> None:
    with pytest.raises(errors.BadRequestError, match="gpt-4o"):
        PrimaGate(api_keys=KEYS, fallbacks={"gpt-4o": "claude-sonnet-5-5"})
