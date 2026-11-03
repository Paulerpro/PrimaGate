import ast
from pathlib import Path

import pytest

import primagate
from primagate.core import steps
from primagate.core.call import advance, start
from primagate.core.counting import completion_text, heuristic_count, prompt_text
from primagate.schema import events
from primagate.schema.types import (
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    FunctionCall,
    Message,
    ToolCall,
    Usage,
)
from primagate.schema.wire import Capabilities

CAPS = {"openai": Capabilities()}


@pytest.mark.parametrize(
    ("text", "tokens"), [("", 0), ("a", 1), ("abcd", 1), ("abcde", 2), ("x" * 400, 100)]
)
def test_the_heuristic_is_one_token_per_four_characters_rounded_up(text: str, tokens: int) -> None:
    assert heuristic_count(text) == tokens


def test_prompt_text_gathers_text_parts_and_tool_calls_and_skips_images() -> None:
    call = ToolCall("1", FunctionCall("lookup", '{"q": "x"}'))
    request = ChatCompletionRequest(
        model="gpt-4o",
        messages=[
            Message(role="system", content="be brief"),
            Message(
                role="user",
                content=[  # type: ignore[arg-type]
                    {"type": "text", "text": "what is this"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            ),
            Message(role="assistant", content=None, tool_calls=[call]),
            Message(role="tool", content="42", tool_call_id="1"),
        ],
    )
    text = prompt_text(request)
    for piece in ("be brief", "what is this", "lookup", '{"q": "x"}', "42"):
        assert piece in text
    assert "AAAA" not in text and "image_url" not in text


def test_completion_text_gathers_every_choice_and_tool_call() -> None:
    call = ToolCall("1", FunctionCall("lookup", "{}"))
    response = ChatCompletion(
        id="c",
        created=1,
        model="m",
        choices=[
            Choice(0, Message(role="assistant", content="first")),
            Choice(1, Message(role="assistant", content=None, tool_calls=[call])),
        ],
    )
    text = completion_text(response)
    assert "first" in text and "lookup" in text


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="gpt-4o", messages=[Message(role="user", content="hello")])


def _reply(usage: Usage | None) -> ChatCompletion:
    message = Message(role="assistant", content="world!")
    return ChatCompletion(
        id="c", created=1, model="gpt-4o", choices=[Choice(0, message)], usage=usage
    )


def _walk(response: ChatCompletion) -> tuple[list[steps.Action], list[events.Event], steps.Action]:
    """Drive core through one successful dispatch, answering CountTokens with fixed counts."""
    seen: list[steps.Action] = []
    announced: list[events.Event] = []
    action = start(_request(), CAPS)
    outcome: steps.Outcome = steps.IdOutcome("req-1")
    while not isinstance(action, steps.Return | steps.Raise):
        action = advance(action, outcome)
        seen.append(action)
        if isinstance(action, steps.Emit):
            announced.extend(action.events)
            outcome = steps.EmitOutcome()
        elif isinstance(action, steps.CountTokens):
            outcome = steps.CountOutcome(5, 7, True)
        elif isinstance(action, steps.CacheRead):
            outcome = steps.CacheReadOutcome(None)
        elif isinstance(action, steps.CacheWrite):
            outcome = steps.CacheWriteOutcome()
        else:
            outcome = steps.DispatchOutcome(response=response)
    return seen, announced, action


def test_a_response_with_usage_is_never_counted() -> None:
    seen, _, last = _walk(_reply(Usage(3, 4, 7)))
    assert not any(isinstance(a, steps.CountTokens) for a in seen)
    assert isinstance(last, steps.Return) and last.response.metadata is not None
    assert last.response.metadata.token_counts is not None
    assert last.response.metadata.token_counts.heuristic is False


def test_a_response_without_usage_asks_the_driver_to_count_and_is_priced_from_it() -> None:
    seen, announced, last = _walk(_reply(None))
    asks = [a for a in seen if isinstance(a, steps.CountTokens)]
    assert len(asks) == 1
    assert (asks[0].model, asks[0].prompt, asks[0].completion) == ("gpt-4o", "hello", "world!")
    assert isinstance(last, steps.Return) and last.response.metadata is not None
    counts, cost = last.response.metadata.token_counts, last.response.metadata.cost
    assert counts is not None and (counts.prompt_tokens, counts.completion_tokens) == (5, 7)
    assert counts.total_tokens == 12 and counts.heuristic is True
    assert cost is not None and cost.total == pytest.approx((5 * 2.5 + 7 * 10) / 1e6)
    assert last.response.usage is None
    assert [e.event_type for e in announced][-1] == "call_succeeded"


def test_count_tokens_is_in_the_closed_action_set() -> None:
    assert steps.CountTokens in steps.ACTION_TYPES


def _module_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
    return found


def test_tiktoken_is_never_imported_at_module_level() -> None:
    root = Path(primagate.__file__).parent
    offenders = [
        str(path.relative_to(root))
        for path in root.rglob("*.py")
        if "tiktoken" in _module_level_imports(path)
    ]
    assert not offenders, offenders
    source = (root / "drivers" / "tokens.py").read_text(encoding="utf-8")
    assert "import tiktoken" in source, "the lazy import must exist where the counter uses it"


def test_tiktoken_is_declared_only_as_an_optional_extra() -> None:
    import tomllib

    config = tomllib.loads(
        (Path(primagate.__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert not any("tiktoken" in d for d in config["project"]["dependencies"])
    (spec,) = config["project"]["optional-dependencies"]["tokens"]
    assert spec.startswith("tiktoken>=") and ",<" in spec
