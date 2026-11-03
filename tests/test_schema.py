from primagate.schema.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChunkChoice,
    Delta,
    FunctionCall,
    Message,
    ToolCall,
    Usage,
)


def _response() -> ChatCompletion:
    return ChatCompletion(
        id="chatcmpl-1",
        created=1,
        model="gpt-4o",
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content="hi"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3),
    )


def test_request_attribute_access_and_to_dict() -> None:
    req = ChatCompletionRequest(
        model="gpt-4o",
        messages=[Message(role="user", content="hello")],
        temperature=0.2,
    )
    assert req.messages[0].content == "hello"
    assert req.to_dict() == {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
    }


def test_response_attribute_access() -> None:
    resp = _response()
    assert resp.choices[0].message.content == "hi"
    assert resp.usage is not None
    assert resp.usage.total_tokens == 3


def test_to_dict_nests_recursively_and_matches_model_dump() -> None:
    resp = _response()
    data = resp.to_dict()
    assert data["choices"][0]["message"] == {"role": "assistant", "content": "hi"}
    assert data["usage"]["total_tokens"] == 3
    assert resp.model_dump() == data
    assert isinstance(data["choices"][0], dict)


def test_chunk_round_trip() -> None:
    chunk = ChatCompletionChunk(
        id="c1",
        created=1,
        model="gpt-4o",
        choices=[ChunkChoice(index=0, delta=Delta(content="x"), finish_reason=None)],
    )
    assert chunk.choices[0].delta.content == "x"
    assert chunk.model_dump() == chunk.to_dict()
    assert chunk.to_dict()["choices"][0]["delta"] == {"content": "x"}


def test_assistant_tool_call_message_keeps_null_content() -> None:
    call = ToolCall(id="call_1", function=FunctionCall(name="f", arguments="{}"))
    assert Message(role="assistant", tool_calls=[call]).to_dict() == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
        ],
    }


def test_message_without_content_or_calls_still_omits_content() -> None:
    assert Message(role="user").to_dict() == {"role": "user"}


def test_tool_result_message_carries_its_call_id() -> None:
    msg = Message(role="tool", tool_call_id="call_1", content="42")
    assert msg.to_dict() == {"role": "tool", "content": "42", "tool_call_id": "call_1"}


def test_tool_result_without_content_still_sends_a_null_content() -> None:
    assert Message(role="tool", tool_call_id="c1").to_dict() == {
        "role": "tool",
        "content": None,
        "tool_call_id": "c1",
    }


def test_metadata_is_omitted_until_the_core_fills_it() -> None:
    from primagate.schema.types import ChatCompletion, Choice, Message, Metadata

    choice = Choice(index=0, message=Message(role="assistant", content="x"))
    bare = ChatCompletion(id="c", created=1, model="m", choices=[choice])
    assert "metadata" not in bare.to_dict()
    named = ChatCompletion(
        id="c", created=1, model="m", choices=[choice], metadata=Metadata(dropped_params=["tools"])
    )
    assert named.to_dict()["metadata"] == {
        "dropped_params": ["tools"],
        "attempt_chain": [],
        "unknown_model": False,
    }


def test_a_skipped_row_serializes_its_reason_and_a_normal_row_does_not() -> None:
    from primagate.schema.types import Attempt, DispatchRecord

    real = Attempt("openai/gpt-4o", [DispatchRecord("openai", 0)])
    skipped = Attempt("anthropic/claude", [], skipped="no tools")
    assert "skipped" not in real.to_dict()
    assert skipped.to_dict() == {
        "model": "anthropic/claude",
        "dispatches": [],
        "skipped": "no tools",
    }
