import asyncio

import httpx2
import pytest

from primagate.schema.wire import ProviderRequest, RawOutcome
from primagate.transport.async_ import AsyncTransport
from primagate.transport.sync import SyncTransport

REQ = ProviderRequest(
    method="POST",
    url="https://x.test/v1/chat/completions",
    headers={"Authorization": "Bearer k"},
    json_body={"a": 1},
)


@pytest.fixture(params=["sync", "async"])
def send(request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    def run(transport: httpx2.MockTransport) -> RawOutcome:
        if request.param == "sync":
            sync = SyncTransport(timeout=5.0, transport=transport)
            try:
                return sync.send(REQ)
            finally:
                sync.close()

        async def go() -> RawOutcome:
            asynchronous = AsyncTransport(timeout=5.0, transport=transport)
            try:
                return await asynchronous.send(REQ)
            finally:
                await asynchronous.aclose()

        return asyncio.run(go())

    return run


def test_returns_raw_status_headers_body(send) -> None:  # type: ignore[no-untyped-def]
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(429, headers={"retry-after": "3"}, content=b"slow")

    out = send(httpx2.MockTransport(handle))
    assert (out.status, out.body, out.failure) == (429, b"slow", None)
    assert out.headers["retry-after"] == "3"
    assert len(seen) == 1, "transport must not retry"
    assert seen[0].headers["authorization"] == "Bearer k"


@pytest.mark.parametrize(
    ("exc", "kind"),
    [(httpx2.ReadTimeout("t"), "timeout"), (httpx2.ConnectError("c"), "connection")],
)
def test_failures_are_reported_not_mapped(send, exc: Exception, kind: str) -> None:  # type: ignore[no-untyped-def]
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise exc

    out = send(httpx2.MockTransport(handle))
    assert out.failure == kind
    assert out.status is None


def test_other_httpx_errors_are_reported_not_raised(send) -> None:  # type: ignore[no-untyped-def]
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.DecodingError("bad encoding")

    out = send(httpx2.MockTransport(handle))
    assert out.failure == "connection"


async def test_async_transport_keeps_one_client_per_loop() -> None:
    transport = AsyncTransport(
        timeout=5.0, transport=httpx2.MockTransport(lambda r: httpx2.Response(200))
    )
    await asyncio.gather(*(transport.send(REQ) for _ in range(4)))
    assert len(transport._clients) == 1
    first = next(iter(transport._clients.values()))

    def other_loop() -> httpx2.AsyncClient:
        async def go() -> httpx2.AsyncClient:
            await transport.send(REQ)
            return transport._clients[asyncio.get_running_loop()]

        return asyncio.run(go())

    second = await asyncio.to_thread(other_loop)
    assert second is not first
    await transport.aclose()
    assert first.is_closed
    assert len(transport._clients) == 0


async def test_async_transport_drops_clients_of_finished_loops() -> None:
    transport = AsyncTransport(
        timeout=5.0, transport=httpx2.MockTransport(lambda r: httpx2.Response(200))
    )
    kept: list[object] = []

    def other_loop() -> None:
        async def go() -> None:
            await transport.send(REQ)
            kept.append(asyncio.get_running_loop())  # keeps the loop object alive

        asyncio.run(go())

    await asyncio.to_thread(other_loop)
    assert len(transport._clients) == 1
    await transport.send(REQ)
    assert len(transport._clients) == 1
    assert asyncio.get_running_loop() in transport._clients
