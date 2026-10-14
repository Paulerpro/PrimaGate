from __future__ import annotations

import asyncio
import threading
import weakref

import httpx2

from primagate.schema.wire import ProviderRequest, RawOutcome


class AsyncTransport:
    """Sends requests and reports what happened. It never maps errors or retries."""

    def __init__(
        self, *, timeout: float, transport: httpx2.AsyncBaseTransport | None = None
    ) -> None:
        self._timeout = timeout
        self._transport = transport
        # An AsyncClient is bound to the loop that first used it, so each loop gets its own.
        # Weak keys let a finished loop take its client with it.
        self._clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx2.AsyncClient] = (
            weakref.WeakKeyDictionary()
        )
        self._lock = threading.Lock()

    def _client(self) -> httpx2.AsyncClient:
        loop = asyncio.get_running_loop()
        with self._lock:
            # A client's pooled connections can keep its loop alive, so weak keys alone may
            # never release it; drop entries whose loop has finished.
            for stale in [lp for lp in self._clients if lp.is_closed()]:
                del self._clients[stale]
            client = self._clients.get(loop)
            if client is None or client.is_closed:
                client = httpx2.AsyncClient(timeout=self._timeout, transport=self._transport)
                self._clients[loop] = client
            return client

    async def send(self, request: ProviderRequest) -> RawOutcome:
        try:
            resp = await self._client().request(
                request.method, request.url, headers=request.headers, json=request.json_body
            )
        except httpx2.TimeoutException:
            return RawOutcome(status=None, headers={}, body=b"", failure="timeout")
        except httpx2.HTTPError:
            return RawOutcome(status=None, headers={}, body=b"", failure="connection")
        return RawOutcome(status=resp.status_code, headers=dict(resp.headers), body=resp.content)

    async def aclose(self) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            mine = self._clients.pop(loop, None)
            # Clients on other loops cannot be awaited closed from here.
            self._clients.clear()
        if mine is not None:
            await mine.aclose()
