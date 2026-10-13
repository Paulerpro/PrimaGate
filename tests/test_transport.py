import httpx2
import pytest

from primagate.schema.wire import ProviderRequest, RawOutcome
from primagate.transport.sync import SyncTransport

REQ = ProviderRequest(
    method="POST",
    url="https://x.test/v1/chat/completions",
    headers={"Authorization": "Bearer k"},
    json_body={"a": 1},
)


def _send(transport: httpx2.MockTransport) -> RawOutcome:
    sync = SyncTransport(timeout=5.0, transport=transport)
    try:
        return sync.send(REQ)
    finally:
        sync.close()


def test_returns_raw_status_headers_body() -> None:
    seen: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(429, headers={"retry-after": "3"}, content=b"slow")

    out = _send(httpx2.MockTransport(handle))
    assert (out.status, out.body, out.failure) == (429, b"slow", None)
    assert out.headers["retry-after"] == "3"
    assert len(seen) == 1, "transport must not retry"
    assert seen[0].headers["authorization"] == "Bearer k"


@pytest.mark.parametrize(
    ("exc", "kind"),
    [(httpx2.ReadTimeout("t"), "timeout"), (httpx2.ConnectError("c"), "connection")],
)
def test_failures_are_reported_not_mapped(exc: Exception, kind: str) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise exc

    out = _send(httpx2.MockTransport(handle))
    assert out.failure == kind
    assert out.status is None


def test_other_httpx_errors_are_reported_not_raised() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.DecodingError("bad encoding")

    out = _send(httpx2.MockTransport(handle))
    assert out.failure == "connection"
