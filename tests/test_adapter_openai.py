import ast
import json
from pathlib import Path

import pytest

import primagate.adapters.openai as openai_adapter
from primagate.adapters.openai import OpenAIAdapter
from primagate.schema import errors
from primagate.schema.types import ChatCompletionRequest, Message
from primagate.schema.wire import RawOutcome


def _outcome(status: int, body: object, *, headers: dict[str, str] | None = None) -> RawOutcome:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return RawOutcome(status=status, headers=headers or {}, body=raw)


def test_build_request() -> None:
    adapter = OpenAIAdapter()
    req = ChatCompletionRequest(
        model="gpt-4o", messages=[Message(role="user", content="hi")], temperature=0.1
    )
    out = adapter.build_request(req, api_key="sk-secret", base_url="https://x.test/v1/")
    assert out.method == "POST"
    assert out.url == "https://x.test/v1/chat/completions"
    assert out.headers["Authorization"] == "Bearer sk-secret"
    assert out.json_body == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.1,
    }
    assert "sk-secret" not in repr(out)


def test_parse_response_ignores_unknown_fields() -> None:
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 5,
        "model": "gpt-4o",
        "system_fingerprint": "fp",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "logprobs": None,
                "message": {"role": "assistant", "content": "hello", "refusal": None},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3, "x": {}},
    }
    resp = OpenAIAdapter().parse_response(_outcome(200, body))
    assert resp.choices[0].message.content == "hello"
    assert resp.usage is not None
    assert resp.usage.total_tokens == 3


def test_parse_response_without_usage() -> None:
    body = {"id": "c", "created": 1, "model": "m", "choices": []}
    assert OpenAIAdapter().parse_response(_outcome(200, body)).usage is None


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"choices": 5}'])
def test_parse_response_invalid_body_is_api_error(body: bytes) -> None:
    with pytest.raises(errors.APIError):
        OpenAIAdapter().parse_response(_outcome(200, body))


def _err(status: int, code: str | None = None, msg: str = "boom") -> RawOutcome:
    return _outcome(status, {"error": {"message": msg, "code": code, "type": "t"}})


@pytest.mark.parametrize(
    ("outcome", "cls"),
    [
        (_err(400), errors.BadRequestError),
        (_err(400, "context_length_exceeded"), errors.ContextWindowExceededError),
        (_err(400, "content_policy_violation"), errors.ContentPolicyViolationError),
        (_err(401), errors.AuthenticationError),
        (_err(403), errors.PermissionDeniedError),
        (_err(404), errors.NotFoundError),
        (_err(408), errors.Timeout),
        (_err(429), errors.RateLimitError),
        (_err(500), errors.ServiceUnavailableError),
        (_err(503), errors.ServiceUnavailableError),
        (_outcome(502, b"<html>bad gateway</html>"), errors.ServiceUnavailableError),
        (_err(418), errors.APIError),
        (RawOutcome(status=None, headers={}, body=b"", failure="timeout"), errors.Timeout),
        (
            RawOutcome(status=None, headers={}, body=b"", failure="connection"),
            errors.APIConnectionError,
        ),
    ],
)
def test_map_error(outcome: RawOutcome, cls: type[errors.PrimaGateError]) -> None:
    err = OpenAIAdapter().map_error(outcome)
    assert type(err) is cls
    assert err.llm_provider == "openai"
    assert err.status_code == outcome.status


def test_map_error_uses_provider_message() -> None:
    assert OpenAIAdapter().map_error(_err(429, msg="slow down")).message == "slow down"


def test_capabilities_and_stream_stub() -> None:
    adapter = OpenAIAdapter()
    assert adapter.capabilities().streaming is False
    with pytest.raises(NotImplementedError):
        adapter.new_stream_parser()


def test_adapter_module_performs_no_io() -> None:
    tree = ast.parse(Path(openai_adapter.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"httpx2", "socket", "urllib", "requests", "os", "subprocess"}


@pytest.mark.parametrize("status", [500, 502, 504])
def test_all_5xx_map_to_service_unavailable(status: int) -> None:
    assert type(OpenAIAdapter().map_error(_err(status))) is errors.ServiceUnavailableError


@pytest.mark.parametrize("code", [["a"], {"k": 1}, 5])
def test_non_string_error_code_does_not_crash(code: object) -> None:
    out = _outcome(400, {"error": {"message": "bad", "code": code}})
    assert type(OpenAIAdapter().map_error(out)) is errors.BadRequestError


@pytest.mark.parametrize("message", [None, ""])
def test_empty_provider_message_falls_back_to_body(message: object) -> None:
    out = _outcome(400, {"error": {"message": message, "code": None}})
    assert OpenAIAdapter().map_error(out).message not in ("None", "")
