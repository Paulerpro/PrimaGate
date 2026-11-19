import asyncio

import httpx2
import pytest


def _handler(calls: list[httpx2.Request]) -> httpx2.MockTransport:
    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(200, json={"ok": True})

    return httpx2.MockTransport(handle)


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_recorded_mode_needs_no_network(mode: str) -> None:
    calls: list[httpx2.Request] = []
    if mode == "sync":
        with httpx2.Client(transport=_handler(calls)) as client:
            resp = client.get("https://provider.invalid/v1/ping")
    else:

        async def fetch() -> httpx2.Response:
            async with httpx2.AsyncClient(transport=_handler(calls)) as client:
                return await client.get("https://provider.invalid/v1/ping")

        resp = asyncio.run(fetch())
    assert resp.json() == {"ok": True}
    # A vacuous pass would leave this empty.
    assert len(calls) == 1


@pytest.mark.fr("FR1")
def test_first_call_recorded_through_primagate(call) -> None:  # type: ignore[no-untyped-def]
    import sys

    from primagate import PrimaGate

    calls: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "created": 1,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hi"},
                    }
                ],
            },
        )

    gate = PrimaGate(api_keys={"openai": "sk-recorded"}, transport=httpx2.MockTransport(handle))
    resp = call(gate, model="gpt-4o", messages=[{"role": "user", "content": "hello"}])
    assert resp.choices[0].message.content == "hi"
    assert len(calls) == 1
    assert "openai" not in sys.modules
