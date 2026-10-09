from primagate.schema.types import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionRequest,
    Choice,
    ChunkChoice,
    Delta,
    Message,
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
