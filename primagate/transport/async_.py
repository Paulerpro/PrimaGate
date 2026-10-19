from __future__ import annotations

import asyncio
import threading
import weakref
from collections.abc import AsyncGenerator

import httpx2

from primagate.schema.wire import ProviderRequest, RawEvent, RawOutcome, StreamFailure


class AsyncEventStream:
    """An opened event stream. Failures after opening are reported as `StreamFailure` kinds."""

    def __init__(self, response: httpx2.Response) -> None:
        self._response = response
        self.headers = dict(response.headers)

    async def events(self) -> AsyncGenerator[RawEvent]:
        try:
            async for event in httpx2.EventSource(self._response):
                yield RawEvent(event.event, event.data)
        except httpx2.TimeoutException:
            raise StreamFailure("timeout") from None
        except httpx2.HTTPError:
            raise StreamFailure("connection") from None

    async def aclose(self) -> None:
        await self._response.aclose()


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

    async def open_stream(self, request: ProviderRequest) -> RawOutcome | AsyncEventStream:
        """Open a streaming response; a non-success status comes back whole as an outcome."""
        client = self._client()
        try:
            built = client.build_request(
                request.method, request.url, headers=request.headers, json=request.json_body
            )
            resp = await client.send(built, stream=True)
        except httpx2.TimeoutException:
            return RawOutcome(status=None, headers={}, body=b"", failure="timeout")
        except httpx2.HTTPError:
            return RawOutcome(status=None, headers={}, body=b"", failure="connection")
        if 200 <= resp.status_code < 300:
            return AsyncEventStream(resp)
        try:
            body = await resp.aread()
        except httpx2.TimeoutException:
            return RawOutcome(status=None, headers={}, body=b"", failure="timeout")
        except httpx2.HTTPError:
            return RawOutcome(status=None, headers={}, body=b"", failure="connection")
        finally:
            await resp.aclose()
        return RawOutcome(status=resp.status_code, headers=dict(resp.headers), body=body)

    async def aclose(self) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            mine = self._clients.pop(loop, None)
            # Clients on other loops cannot be awaited closed from here.
            self._clients.clear()
        if mine is not None:
            await mine.aclose()
