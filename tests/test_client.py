import json
import logging
import threading

import httpx2
import pytest

import primagate
import primagate.client as client_module
from primagate import PrimaGate
from primagate.schema import errors

KEY = "sk-test-key-123456"
OK = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hello"},
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
MSGS = [{"role": "user", "content": "hi"}]


def _gate(
    status: int = 200, body: object = OK, key: str | None = KEY
) -> tuple[PrimaGate, list[httpx2.Request]]:
    calls: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(status, json=body)

    return PrimaGate(api_key=key, transport=httpx2.MockTransport(handle)), calls


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def test_first_call_returns_unified_response() -> None:
    gate, calls = _gate()
    resp = gate.completion(model="gpt-4o", messages=MSGS, temperature=0.3)
    assert resp.choices[0].message.content == "hello"
    assert len(calls) == 1
    sent = json.loads(calls[0].content)
    assert sent == {"model": "gpt-4o", "messages": MSGS, "temperature": 0.3}
    assert calls[0].headers["authorization"] == f"Bearer {KEY}"


def test_prefixed_model_is_stripped_before_sending() -> None:
    gate, calls = _gate()
    gate.completion(model="openai/gpt-4o", messages=MSGS)
    assert json.loads(calls[0].content)["model"] == "gpt-4o"


def test_env_key_is_used_when_no_init_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    gate, calls = _gate(key=None)
    gate.completion(model="gpt-4o", messages=MSGS)
    assert calls[0].headers["authorization"] == "Bearer sk-from-env"


def test_init_key_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    gate, calls = _gate()
    gate.completion(model="gpt-4o", messages=MSGS)
    assert calls[0].headers["authorization"] == f"Bearer {KEY}"


def test_missing_key_raises_before_any_request() -> None:
    gate, calls = _gate(key=None)
    with pytest.raises(errors.AuthenticationError) as info:
        gate.completion(model="gpt-4o", messages=MSGS)
    assert info.value.llm_provider == "openai"
    assert calls == []


@pytest.mark.parametrize(
    ("status", "cls"),
    [
        (400, errors.BadRequestError),
        (401, errors.AuthenticationError),
        (404, errors.NotFoundError),
        (429, errors.RateLimitError),
        (503, errors.ServiceUnavailableError),
    ],
)
def test_http_errors_are_mapped_once_without_retry(
    status: int, cls: type[errors.PrimaGateError]
) -> None:
    gate, calls = _gate(status, {"error": {"message": "nope", "code": None}})
    with pytest.raises(cls) as info:
        gate.completion(model="gpt-4o", messages=MSGS)
    assert info.value.status_code == status
    assert len(calls) == 1


def test_connection_failure_maps_to_api_connection_error() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down")

    gate = PrimaGate(api_key=KEY, transport=httpx2.MockTransport(handle))
    with pytest.raises(errors.APIConnectionError):
        gate.completion(model="gpt-4o", messages=MSGS)


def test_key_never_appears_in_errors_repr_or_logs(caplog: pytest.LogCaptureFixture) -> None:
    echoed = {"error": {"message": f"Incorrect API key provided: {KEY}", "code": "invalid"}}
    gate, _ = _gate(401, echoed)
    with caplog.at_level(logging.DEBUG), pytest.raises(errors.AuthenticationError) as info:
        gate.completion(model="gpt-4o", messages=MSGS)
    err = info.value
    assert KEY not in str(err)
    assert KEY not in repr(err)
    assert KEY not in err.message
    assert KEY not in "".join(map(str, err.args))
    assert KEY not in repr(gate)
    assert KEY not in str(gate)
    assert all(KEY not in r.getMessage() for r in caplog.records)


def test_closed_instance_raises() -> None:
    gate, calls = _gate()
    gate.close()
    with pytest.raises(errors.PrimaGateError, match="closed"):
        gate.completion(model="gpt-4o", messages=MSGS)
    assert calls == []


def test_instances_own_their_state() -> None:
    a, calls_a = _gate()
    b, calls_b = _gate()
    a.completion(model="gpt-4o", messages=MSGS)
    assert len(calls_a) == 1
    assert calls_b == []
    a.close()
    b.completion(model="gpt-4o", messages=MSGS)


def test_default_instance_is_created_once_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "_default", None)
    created: list[PrimaGate] = []
    original = PrimaGate.__init__

    def counting_init(self: PrimaGate, *args: object, **kwargs: object) -> None:
        created.append(self)
        original(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(PrimaGate, "__init__", counting_init)
    barrier = threading.Barrier(8)
    seen: list[PrimaGate] = []

    def worker() -> None:
        barrier.wait()
        seen.append(client_module.get_default())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(created) == 1
    assert len({id(g) for g in seen}) == 1


def test_module_level_completion_forwards_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    gate, calls = _gate()
    monkeypatch.setattr(client_module, "_default", gate)
    resp = primagate.completion(model="gpt-4o", messages=MSGS)
    assert resp.id == "chatcmpl-1"
    assert len(calls) == 1


def test_every_optional_param_reaches_the_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    params = {"temperature": 0.5, "top_p": 0.9, "max_tokens": 7, "stop": ["x"], "n": 2, "seed": 3}
    gate, calls = _gate()
    gate.completion(model="gpt-4o", messages=MSGS, **params)  # type: ignore[arg-type]
    assert json.loads(calls[0].content) == {"model": "gpt-4o", "messages": MSGS, **params}

    gate2, calls2 = _gate()
    monkeypatch.setattr(client_module, "_default", gate2)
    primagate.completion(model="gpt-4o", messages=MSGS, **params)  # type: ignore[arg-type]
    assert json.loads(calls2[0].content) == {"model": "gpt-4o", "messages": MSGS, **params}


def test_close_closes_the_underlying_client() -> None:
    gate, _ = _gate()
    gate.completion(model="gpt-4o", messages=MSGS)
    gate.close()
    assert gate._driver is not None
    assert gate._driver._transport._client.is_closed


@pytest.mark.parametrize("bad", ["hello", [{"content": "no role"}], [{"role": ""}]])
def test_malformed_messages_raise_bad_request(bad: object) -> None:
    gate, calls = _gate()
    with pytest.raises(errors.BadRequestError):
        gate.completion(model="gpt-4o", messages=bad)  # type: ignore[arg-type]
    assert calls == []


def test_message_instances_are_accepted() -> None:
    from primagate.schema.types import Message

    gate, calls = _gate()
    gate.completion(model="gpt-4o", messages=[Message(role="user", content="hi")])
    assert json.loads(calls[0].content)["messages"] == MSGS
