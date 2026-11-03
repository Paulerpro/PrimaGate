import asyncio
import gc
import json
import logging
import threading

import httpx2
import pytest
from conftest import Caller

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

    keys = {"openai": key} if key else None
    return PrimaGate(api_keys=keys, transport=httpx2.MockTransport(handle)), calls


@pytest.fixture(autouse=True)
def _no_env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


def test_first_call_returns_unified_response(call: Caller) -> None:
    gate, calls = _gate()
    resp = call(gate, model="gpt-4o", messages=MSGS, temperature=0.3)
    assert resp.choices[0].message.content == "hello"
    assert len(calls) == 1
    sent = json.loads(calls[0].content)
    assert sent == {"model": "gpt-4o", "messages": MSGS, "temperature": 0.3}
    assert calls[0].headers["authorization"] == f"Bearer {KEY}"


def test_prefixed_model_is_stripped_before_sending(call: Caller) -> None:
    gate, calls = _gate()
    call(gate, model="openai/gpt-4o", messages=MSGS)
    assert json.loads(calls[0].content)["model"] == "gpt-4o"


def test_env_key_is_used_when_no_init_key(monkeypatch: pytest.MonkeyPatch, call: Caller) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    gate, calls = _gate(key=None)
    call(gate, model="gpt-4o", messages=MSGS)
    assert calls[0].headers["authorization"] == "Bearer sk-from-env"


def test_init_key_wins_over_env(monkeypatch: pytest.MonkeyPatch, call: Caller) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    gate, calls = _gate()
    call(gate, model="gpt-4o", messages=MSGS)
    assert calls[0].headers["authorization"] == f"Bearer {KEY}"


def test_missing_key_raises_before_any_request(call: Caller) -> None:
    gate, calls = _gate(key=None)
    with pytest.raises(errors.AuthenticationError) as info:
        call(gate, model="gpt-4o", messages=MSGS)
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
    status: int, cls: type[errors.PrimaGateError], call: Caller
) -> None:
    gate, calls = _gate(status, {"error": {"message": "nope", "code": None}})
    with pytest.raises(cls) as info:
        call(gate, model="gpt-4o", messages=MSGS)
    assert info.value.status_code == status
    assert len(calls) == 1


def test_connection_failure_maps_to_api_connection_error(call: Caller) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down")

    gate = PrimaGate(api_keys={"openai": KEY}, transport=httpx2.MockTransport(handle))
    with pytest.raises(errors.APIConnectionError):
        call(gate, model="gpt-4o", messages=MSGS)


def test_key_never_appears_in_errors_repr_or_logs(
    caplog: pytest.LogCaptureFixture, call: Caller
) -> None:
    echoed = {"error": {"message": f"Incorrect API key provided: {KEY}", "code": "invalid"}}
    gate, _ = _gate(401, echoed)
    with caplog.at_level(logging.DEBUG), pytest.raises(errors.AuthenticationError) as info:
        call(gate, model="gpt-4o", messages=MSGS)
    err = info.value
    assert KEY not in str(err)
    assert KEY not in repr(err)
    assert KEY not in err.message
    assert KEY not in "".join(map(str, err.args))
    assert KEY not in repr(gate)
    assert KEY not in str(gate)
    assert all(KEY not in r.getMessage() for r in caplog.records)


def test_closed_instance_raises(call: Caller) -> None:
    gate, calls = _gate()
    gate.close()
    with pytest.raises(errors.PrimaGateError, match="closed"):
        call(gate, model="gpt-4o", messages=MSGS)
    assert calls == []


def test_instances_own_their_state(call: Caller) -> None:
    a, calls_a = _gate()
    b, calls_b = _gate()
    call(a, model="gpt-4o", messages=MSGS)
    assert len(calls_a) == 1
    assert calls_b == []
    a.close()
    call(b, model="gpt-4o", messages=MSGS)


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


def test_module_level_completion_forwards_to_default(
    monkeypatch: pytest.MonkeyPatch, call: Caller
) -> None:
    gate, calls = _gate()
    monkeypatch.setattr(client_module, "_default", gate)
    resp = call.module(model="gpt-4o", messages=MSGS)
    assert resp.id == "chatcmpl-1"
    assert len(calls) == 1


def test_every_optional_param_reaches_the_provider(
    monkeypatch: pytest.MonkeyPatch, call: Caller
) -> None:
    params = {"temperature": 0.5, "top_p": 0.9, "max_tokens": 7, "stop": ["x"], "n": 2, "seed": 3}
    gate, calls = _gate()
    call(gate, model="gpt-4o", messages=MSGS, **params)  # type: ignore[arg-type]
    assert json.loads(calls[0].content) == {"model": "gpt-4o", "messages": MSGS, **params}

    gate2, calls2 = _gate()
    monkeypatch.setattr(client_module, "_default", gate2)
    call.module(model="gpt-4o", messages=MSGS, **params)  # type: ignore[arg-type]
    assert json.loads(calls2[0].content) == {"model": "gpt-4o", "messages": MSGS, **params}


def test_close_closes_the_underlying_client() -> None:
    gate, _ = _gate()
    gate.completion(model="gpt-4o", messages=MSGS)
    gate.close()
    assert gate._driver is not None
    assert gate._driver._transport._client.is_closed


@pytest.mark.parametrize("bad", ["hello", [{"content": "no role"}], [{"role": ""}]])
def test_malformed_messages_raise_bad_request(bad: object, call: Caller) -> None:
    gate, calls = _gate()
    with pytest.raises(errors.BadRequestError):
        call(gate, model="gpt-4o", messages=bad)  # type: ignore[arg-type]
    assert calls == []


def test_message_instances_are_accepted(call: Caller) -> None:
    from primagate.schema.types import Message

    gate, calls = _gate()
    call(gate, model="gpt-4o", messages=[Message(role="user", content="hi")])
    assert json.loads(calls[0].content)["messages"] == MSGS


async def test_acompletion_matches_completion() -> None:
    gate, calls = _gate()
    other, other_calls = _gate()
    sync_resp = gate.completion(model="gpt-4o", messages=MSGS)
    async_resp = await other.acompletion(model="gpt-4o", messages=MSGS)
    assert async_resp == sync_resp
    assert len(calls) == 1 and len(other_calls) == 1


async def test_aclose_closes_the_loop_client_and_blocks_later_calls() -> None:
    gate, calls = _gate()
    await gate.acompletion(model="gpt-4o", messages=MSGS)
    await gate.aclose()
    with pytest.raises(errors.PrimaGateError, match="closed"):
        await gate.acompletion(model="gpt-4o", messages=MSGS)
    with pytest.raises(errors.PrimaGateError, match="closed"):
        gate.completion(model="gpt-4o", messages=MSGS)
    assert len(calls) == 1


async def test_close_blocks_acompletion() -> None:
    gate, calls = _gate()
    gate.close()
    with pytest.raises(errors.PrimaGateError, match="closed"):
        await gate.acompletion(model="gpt-4o", messages=MSGS)
    assert calls == []


def test_each_event_loop_gets_its_own_client_and_sync_has_another() -> None:
    gate, _ = _gate()
    gate.completion(model="gpt-4o", messages=MSGS)
    for _ in range(2):
        asyncio.run(gate.acompletion(model="gpt-4o", messages=MSGS))
    sync_client = gate._driver._transport._client  # type: ignore[union-attr]
    gc.collect()
    # Finished loops release their clients, so none survive the runs.
    assert len(gate._async_driver._transport._clients) == 0  # type: ignore[union-attr]
    assert isinstance(sync_client, httpx2.Client)


async def test_one_client_serves_concurrent_tasks_on_a_loop() -> None:
    gate, calls = _gate()
    await asyncio.gather(
        *(
            gate.acompletion(model="gpt-4o", messages=[{"role": "user", "content": f"hi {i}"}])
            for i in range(5)
        )
    )
    assert len(calls) == 5
    clients = list(gate._async_driver._transport._clients.values())  # type: ignore[union-attr]
    assert len(clients) == 1


def test_sync_and_async_use_separate_drivers() -> None:
    gate, _ = _gate()
    gate.completion(model="gpt-4o", messages=MSGS)
    asyncio.run(gate.acompletion(model="gpt-4o", messages=MSGS))
    assert gate._driver is not gate._async_driver


async def test_aclose_also_closes_the_sync_client_and_the_loop_client() -> None:
    gate, _ = _gate()
    gate.completion(model="gpt-4o", messages=MSGS)
    await gate.acompletion(model="gpt-4o", messages=MSGS)
    sync_client = gate._driver._transport._client  # type: ignore[union-attr]
    await gate.aclose()
    assert sync_client.is_closed
    assert len(gate._async_driver._transport._clients) == 0  # type: ignore[union-attr]


async def test_driver_is_reused_after_the_key_disappears(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    gate, calls = _gate(key=None)
    await gate.acompletion(model="gpt-4o", messages=MSGS)
    monkeypatch.delenv("OPENAI_API_KEY")
    await gate.acompletion(model="gpt-4o", messages=[{"role": "user", "content": "again"}])
    assert len(calls) == 2


ANTHROPIC_OK = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
AKEY = "sk-ant-test-key-123456"


def _both(
    status: int = 200, body: object = ANTHROPIC_OK, **kwargs: object
) -> tuple[PrimaGate, list[httpx2.Request]]:
    calls: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(status, json=body)

    gate = PrimaGate(transport=httpx2.MockTransport(handle), **kwargs)  # type: ignore[arg-type]
    return gate, calls


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "anthropic/claude-sonnet-5-5"])
def test_claude_models_go_to_anthropic_with_their_own_key(model: str, call: Caller) -> None:
    gate, calls = _both(api_keys={"openai": KEY, "anthropic": AKEY})
    resp = call(gate, model=model, messages=MSGS)
    assert resp.choices[0].message.content == "hello"
    request = calls[0]
    assert request.url == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == AKEY
    assert KEY not in str(request.headers)
    assert json.loads(request.content)["model"] == "claude-sonnet-5-5"


def test_anthropic_key_is_never_sent_to_openai(call: Caller) -> None:
    gate, calls = _both(body=OK, api_keys={"openai": KEY, "anthropic": AKEY})
    call(gate, model="gpt-4o", messages=MSGS)
    assert AKEY not in str(calls[0].headers)
    assert calls[0].headers["authorization"] == f"Bearer {KEY}"


def test_anthropic_key_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, call: Caller
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", AKEY)
    gate, calls = _both()
    call(gate, model="claude-sonnet-5-5", messages=MSGS)
    assert calls[0].headers["x-api-key"] == AKEY


def test_init_key_wins_over_anthropic_env(monkeypatch: pytest.MonkeyPatch, call: Caller) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    gate, calls = _both(api_keys={"anthropic": AKEY})
    call(gate, model="claude-sonnet-5-5", messages=MSGS)
    assert calls[0].headers["x-api-key"] == AKEY


def test_missing_anthropic_key_names_anthropic_and_sends_nothing(call: Caller) -> None:
    gate, calls = _both(api_keys={"openai": KEY})
    with pytest.raises(errors.AuthenticationError) as info:
        call(gate, model="claude-sonnet-5-5", messages=MSGS)
    assert info.value.llm_provider == "anthropic"
    assert "ANTHROPIC_API_KEY" in info.value.message
    assert calls == []


def test_missing_key_for_one_provider_does_not_block_the_other(call: Caller) -> None:
    gate, calls = _gate()
    call(gate, model="gpt-4o", messages=MSGS)
    with pytest.raises(errors.AuthenticationError):
        call(gate, model="claude-sonnet-5-5", messages=MSGS)
    assert len(calls) == 1


def test_base_urls_override_per_provider(call: Caller) -> None:
    gate, calls = _both(
        api_keys={"anthropic": AKEY}, base_urls={"anthropic": "https://proxy.test/anthropic/v1"}
    )
    call(gate, model="claude-sonnet-5-5", messages=MSGS)
    assert str(calls[0].url) == "https://proxy.test/anthropic/v1/messages"


def test_unknown_provider_names_in_config_maps_raise_at_init() -> None:
    with pytest.raises(errors.BadRequestError, match="anthopic"):
        PrimaGate(api_keys={"anthopic": AKEY})
    with pytest.raises(errors.BadRequestError, match="nope"):
        PrimaGate(base_urls={"nope": "https://x.test"})


def test_unknown_provider_prefix_fails_before_any_network_call(call: Caller) -> None:
    gate, calls = _both(api_keys={"anthropic": AKEY})
    with pytest.raises(errors.BadRequestError, match="nope"):
        call(gate, model="nope/some-model", messages=MSGS)
    assert calls == []


def test_anthropic_context_window_and_openai_context_window_share_a_class(call: Caller) -> None:
    openai_gate, _ = _gate(
        400, {"error": {"message": "too long", "code": "context_length_exceeded"}}
    )
    anthropic_gate, _ = _both(
        400,
        {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "prompt is too long"},
        },
        api_keys={"anthropic": AKEY},
    )
    with pytest.raises(errors.ContextWindowExceededError):
        call(openai_gate, model="gpt-4o", messages=MSGS)
    with pytest.raises(errors.ContextWindowExceededError):
        call(anthropic_gate, model="claude-sonnet-5-5", messages=MSGS)


def test_anthropic_key_never_appears_in_errors_repr_or_logs(
    caplog: pytest.LogCaptureFixture, call: Caller
) -> None:
    echoed = {
        "type": "error",
        "error": {"type": "authentication_error", "message": f"invalid x-api-key {AKEY}"},
    }
    gate, _ = _both(401, echoed, api_keys={"anthropic": AKEY})
    with caplog.at_level(logging.DEBUG), pytest.raises(errors.AuthenticationError) as info:
        call(gate, model="claude-sonnet-5-5", messages=MSGS)
    err = info.value
    for text in (str(err), repr(err), err.message, "".join(map(str, err.args)), repr(gate)):
        assert AKEY not in text
    assert all(AKEY not in r.getMessage() for r in caplog.records)


def test_client_module_carries_no_openai_literals() -> None:
    from pathlib import Path

    source = Path(client_module.__file__).read_text(encoding="utf-8")
    assert "openai" not in source.lower()


TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}


def test_tool_definitions_and_dict_tool_messages_reach_the_wire(call: Caller) -> None:
    gate, calls = _gate()
    call(
        gate,
        model="gpt-4o",
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "42"},
        ],
        tools=[TOOL],
    )
    sent = json.loads(calls[0].content)
    assert sent["tools"] == [TOOL]
    assert sent["messages"][1]["content"] is None
    assert sent["messages"][1]["tool_calls"][0]["id"] == "c1"
    assert sent["messages"][2] == {"role": "tool", "content": "42", "tool_call_id": "c1"}


@pytest.mark.parametrize(
    "tools",
    [
        [{"type": "other", "function": {"name": "f"}}],
        [{"type": "function"}],
        [{"type": "function", "function": {"description": "no name"}}],
        ["f"],
        "f",
    ],
)
def test_bad_tool_definition_raises_before_any_request(call: Caller, tools: object) -> None:
    gate, calls = _gate()
    with pytest.raises(errors.BadRequestError):
        call(gate, model="gpt-4o", messages=MSGS, tools=tools)
    assert calls == []


def test_tool_message_without_an_id_raises_before_any_request(call: Caller) -> None:
    gate, calls = _gate()
    with pytest.raises(errors.BadRequestError, match="tool_call_id"):
        call(gate, model="gpt-4o", messages=[{"role": "tool", "content": "42"}])
    assert calls == []


def test_misspelled_keyword_is_rejected(call: Caller) -> None:
    gate, calls = _gate()
    with pytest.raises(TypeError, match="temprature"):
        call(gate, model="gpt-4o", messages=MSGS, temprature=0.1)
    assert calls == []


def test_missing_model_is_rejected(call: Caller) -> None:
    gate, calls = _gate()
    with pytest.raises(TypeError, match="model"):
        call(gate, messages=MSGS)
    assert calls == []


def test_empty_tools_list_sends_no_tools_key(call: Caller) -> None:
    gate, calls = _gate()
    call(gate, model="gpt-4o", messages=MSGS, tools=[])
    assert "tools" not in json.loads(calls[0].content)


@pytest.mark.parametrize(
    "tool_call",
    [
        {"id": "c1", "function": {"name": "f", "arguments": {"a": 1}}},
        {"id": "c1", "function": {"name": "f", "arguments": None}},
        {"id": 1, "function": {"name": "f", "arguments": "{}"}},
        {"function": {"name": "f", "arguments": "{}"}},
    ],
)
def test_malformed_tool_call_raises_before_any_request(call: Caller, tool_call: dict) -> None:  # type: ignore[type-arg]
    gate, calls = _gate()
    with pytest.raises(errors.BadRequestError):
        call(
            gate,
            model="gpt-4o",
            messages=[{"role": "assistant", "content": None, "tool_calls": [tool_call]}],
        )
    assert calls == []


def test_tool_name_must_be_a_string(call: Caller) -> None:
    gate, calls = _gate()
    with pytest.raises(errors.BadRequestError):
        call(
            gate,
            model="gpt-4o",
            messages=MSGS,
            tools=[{"type": "function", "function": {"name": 5}}],
        )
    assert calls == []


SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
JSON_SCHEMA_FORMAT = {"type": "json_schema", "json_schema": {"name": "place", "schema": SCHEMA}}


class _Place:
    """Stands in for a pydantic model; the SDK must only call model_json_schema()."""

    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        return SCHEMA


def _sent_format(calls: list[httpx2.Request]) -> object:
    return json.loads(calls[0].content).get("response_format")


def test_json_schema_response_format_is_sent_as_given(call: Caller) -> None:
    gate, calls = _gate()
    call(gate, model="gpt-4o", messages=MSGS, response_format=JSON_SCHEMA_FORMAT)
    assert _sent_format(calls) == JSON_SCHEMA_FORMAT


def test_class_response_format_reads_its_schema(call: Caller) -> None:
    gate, calls = _gate()
    call(gate, model="gpt-4o", messages=MSGS, response_format=_Place)
    assert _sent_format(calls) == {
        "type": "json_schema",
        "json_schema": {"name": "_Place", "schema": SCHEMA},
    }


def test_text_response_format_is_treated_as_unset(call: Caller) -> None:
    gate, calls = _gate()
    call(gate, model="gpt-4o", messages=MSGS, response_format={"type": "text"})
    assert _sent_format(calls) is None


def test_json_object_response_format_is_sent_to_openai(call: Caller) -> None:
    gate, calls = _gate()
    call(gate, model="gpt-4o", messages=MSGS, response_format={"type": "json_object"})
    assert _sent_format(calls) == {"type": "json_object"}


@pytest.mark.parametrize(
    "bad",
    [
        "json",
        {"type": "xml"},
        {"type": "json_schema"},
        {"type": "json_schema", "json_schema": {"name": "p"}},
        {"type": "json_schema", "json_schema": {"schema": SCHEMA}},
        {"type": "json_schema", "json_schema": {"name": "p", "schema": "x"}},
    ],
)
def test_bad_response_format_raises_before_any_request(call: Caller, bad: object) -> None:
    gate, calls = _gate()
    with pytest.raises(errors.BadRequestError):
        call(gate, model="gpt-4o", messages=MSGS, response_format=bad)
    assert calls == []


def test_pydantic_is_never_imported() -> None:
    import subprocess
    import sys

    code = "import primagate, sys; sys.exit('pydantic' in sys.modules)"
    assert subprocess.run([sys.executable, "-I", "-c", code]).returncode == 0


def test_stream_false_is_a_plain_call(call: Caller) -> None:
    gate, calls = _gate()
    result = call(gate, model="gpt-4o", messages=MSGS, stream=False)
    assert result.choices[0].message.content == "hello"
    assert "stream" not in json.loads(calls[0].content)
