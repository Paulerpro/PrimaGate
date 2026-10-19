from __future__ import annotations

from collections.abc import Iterator

import httpx2

from primagate.schema.wire import ProviderRequest, RawEvent, RawOutcome, StreamFailure


class SyncEventStream:
    """An opened event stream. Failures after opening are reported as `StreamFailure` kinds."""

    def __init__(self, response: httpx2.Response) -> None:
        self._response = response
        self.headers = dict(response.headers)

    def events(self) -> Iterator[RawEvent]:
        try:
            for event in httpx2.EventSource(self._response):
                yield RawEvent(event.event, event.data)
        except httpx2.TimeoutException:
            raise StreamFailure("timeout") from None
        except httpx2.HTTPError:
            raise StreamFailure("connection") from None

    def close(self) -> None:
        self._response.close()


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

    def open_stream(self, request: ProviderRequest) -> RawOutcome | SyncEventStream:
        """Open a streaming response; a non-success status comes back whole as an outcome."""
        try:
            built = self._client.build_request(
                request.method, request.url, headers=request.headers, json=request.json_body
            )
            resp = self._client.send(built, stream=True)
        except httpx2.TimeoutException:
            return RawOutcome(status=None, headers={}, body=b"", failure="timeout")
        except httpx2.HTTPError:
            return RawOutcome(status=None, headers={}, body=b"", failure="connection")
        if 200 <= resp.status_code < 300:
            return SyncEventStream(resp)
        try:
            body = resp.read()
        except httpx2.TimeoutException:
            return RawOutcome(status=None, headers={}, body=b"", failure="timeout")
        except httpx2.HTTPError:
            return RawOutcome(status=None, headers={}, body=b"", failure="connection")
        finally:
            resp.close()
        return RawOutcome(status=resp.status_code, headers=dict(resp.headers), body=body)

    def close(self) -> None:
        self._client.close()
