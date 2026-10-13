from __future__ import annotations

import httpx2

from primagate.schema.wire import ProviderRequest, RawOutcome


class SyncTransport:
    """Sends requests and reports what happened. It never maps errors or retries."""

    def __init__(self, *, timeout: float, transport: httpx2.BaseTransport | None = None) -> None:
        self._client = httpx2.Client(timeout=timeout, transport=transport)

    def send(self, request: ProviderRequest) -> RawOutcome:
        try:
            resp = self._client.request(
                request.method, request.url, headers=request.headers, json=request.json_body
            )
        except httpx2.TimeoutException:
            return RawOutcome(status=None, headers={}, body=b"", failure="timeout")
        except httpx2.HTTPError:
            return RawOutcome(status=None, headers={}, body=b"", failure="connection")
        return RawOutcome(status=resp.status_code, headers=dict(resp.headers), body=resp.content)

    def close(self) -> None:
        self._client.close()
