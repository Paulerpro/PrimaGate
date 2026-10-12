import httpx2


def _handler(calls: list[httpx2.Request]) -> httpx2.MockTransport:
    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(200, json={"ok": True})

    return httpx2.MockTransport(handle)


def test_sync_recorded_mode_needs_no_network() -> None:
    calls: list[httpx2.Request] = []
    with httpx2.Client(transport=_handler(calls)) as client:
        resp = client.get("https://provider.invalid/v1/ping")
    assert resp.json() == {"ok": True}
    # A vacuous pass would leave this empty.
    assert len(calls) == 1


async def test_async_recorded_mode_needs_no_network() -> None:
    calls: list[httpx2.Request] = []
    async with httpx2.AsyncClient(transport=_handler(calls)) as client:
        resp = await client.get("https://provider.invalid/v1/ping")
    assert resp.json() == {"ok": True}
    assert len(calls) == 1
