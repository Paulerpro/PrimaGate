import copy
import dataclasses
import inspect
import re
from typing import Any

import pytest

import primagate
from primagate.core import context as core_context
from primagate.core.context import MAX_EXCHANGES, OverBudget, RoutingContext, bounded_context
from primagate.core.counting import heuristic_count, messages_text
from primagate.schema.types import ChatCompletionRequest, FunctionCall, Message, ToolCall

BIG = 10_000


def _turn(label: str, size: int = 40) -> list[Message]:
    """One exchange: a question and an answer, each `size` characters."""
    return [
        Message(role="user", content=(f"q{label}" + "x" * size)[:size]),
        Message(role="assistant", content=(f"a{label}" + "y" * size)[:size]),
    ]


def _request(*groups: list[Message]) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="primagate/auto", messages=[m for group in groups for m in group]
    )


def _tokens(messages: list[Message]) -> int:
    return heuristic_count(messages_text(messages))


def _built(request: ChatCompletionRequest, budget: int = BIG) -> RoutingContext:
    result = bounded_context(request, budget)
    assert isinstance(result, RoutingContext)
    return result


def _texts(context: RoutingContext) -> list[str]:
    return [str(m.content) for m in context.messages]


def test_five_earlier_exchanges_keep_the_latest_three_and_the_current_one() -> None:
    turns = [_turn(str(i)) for i in range(6)]  # five earlier and a current one
    context = _built(_request(*turns))
    assert MAX_EXCHANGES == 3 and context.earlier_kept == 3
    assert _texts(context) == [str(m.content) for t in turns[2:] for m in t]


def test_fewer_than_three_earlier_exchanges_are_all_kept() -> None:
    turns = [_turn(str(i)) for i in range(3)]
    context = _built(_request(*turns))
    assert context.earlier_kept == 2 and len(context.messages) == 6


def test_the_budget_can_bind_before_the_count() -> None:
    turns = [_turn(str(i)) for i in range(5)]
    fits_two_earlier = _tokens(turns[4]) + 2 * _tokens(turns[3])
    context = _built(_request(*turns), fits_two_earlier)
    assert context.earlier_kept == 2
    assert _texts(context) == [str(m.content) for t in turns[2:] for m in t]
    assert context.tokens <= fits_two_earlier


def test_a_budget_that_fits_only_the_current_exchange_keeps_nothing_earlier() -> None:
    turns = [_turn(str(i)) for i in range(4)]
    current = _tokens(turns[-1])
    context = _built(_request(*turns), current)
    assert context.earlier_kept == 0 and context.tokens == current
    assert _texts(context) == [str(m.content) for m in turns[-1]]


def test_an_exchange_is_dropped_whole_never_cut() -> None:
    turns = [_turn("0", 400), _turn("1", 40)]
    current = _tokens(turns[1])
    # Room for part of the earlier exchange, but not all of it.
    context = _built(_request(*turns), current + _tokens(turns[0]) - 1)
    assert context.earlier_kept == 0
    assert _texts(context) == [str(m.content) for m in turns[1]]


def test_the_total_equal_to_the_budget_is_kept() -> None:
    turns = [_turn("0"), _turn("1")]
    exact = sum(_tokens(t) for t in turns)
    assert _built(_request(*turns), exact).earlier_kept == 1
    assert _built(_request(*turns), exact - 1).earlier_kept == 0


def test_a_tool_loop_stays_together_as_the_current_exchange() -> None:
    call = ToolCall(id="c1", type="function", function=FunctionCall("lookup", '{"id": 1}'))
    loop = [
        Message(role="user", content="find order 1"),
        Message(role="assistant", content=None, tool_calls=[call]),
        Message(role="tool", content='{"status": "shipped"}', tool_call_id="c1"),
    ]
    earlier = _turn("0")
    context = _built(_request(earlier, loop))
    assert context.earlier_kept == 1 and context.messages[-3:] == tuple(loop)
    only_current = _built(_request(loop), _tokens(loop))
    assert len(only_current.messages) == 3 and only_current.earlier_kept == 0


def test_the_current_exchange_alone_over_the_budget_says_so() -> None:
    request = _request(_turn("0"), _turn("1", 400))
    current = _tokens(request.messages[2:])
    result = bounded_context(request, current - 1)
    assert isinstance(result, OverBudget)
    assert result.tokens == current and result.budget == current - 1


def test_a_current_exchange_is_never_cut_to_fit() -> None:
    request = _request(_turn("0", 400))
    result = bounded_context(request, 5)
    assert isinstance(result, OverBudget)
    assert len(request.messages) == 2 and len(str(request.messages[0].content)) == 400


def test_the_request_is_left_exactly_as_given_and_the_view_is_a_copy() -> None:
    request = _request(*[_turn(str(i)) for i in range(5)])
    before = copy.deepcopy(request)
    context = _built(request)
    assert request == before
    assert not any(v is m for v in context.messages for m in request.messages)
    context.messages[0].content = "changed in the view"
    context.messages[-1].tool_calls = [
        ToolCall(id="x", type="function", function=FunctionCall("f", "{}"))
    ]
    assert request == before


def test_tool_calls_inside_the_view_are_copies_too() -> None:
    call = ToolCall(id="c1", type="function", function=FunctionCall("lookup", "{}"))
    request = _request(
        [Message(role="user", content="hi"), Message(role="assistant", tool_calls=[call])]
    )
    context = _built(request)
    assert context.messages[1].tool_calls is not request.messages[1].tool_calls
    context.messages[1].tool_calls[0].function.name = "other"  # type: ignore[index]
    assert call.function.name == "lookup"


def test_tool_arguments_and_tool_results_count_against_the_budget() -> None:
    call = ToolCall(id="c1", type="function", function=FunctionCall("lookup", "a" * 800))
    loop = [
        Message(role="user", content="find it"),
        Message(role="assistant", content=None, tool_calls=[call]),
        Message(role="tool", content="r" * 800, tool_call_id="c1"),
    ]
    assert _tokens(loop) >= 400
    assert _built(_request(loop)).tokens == _tokens(loop)
    assert isinstance(bounded_context(_request(loop), 399), OverBudget)


def test_a_request_with_no_messages_gives_an_empty_view() -> None:
    context = _built(ChatCompletionRequest(model="primagate/auto", messages=[]))
    assert context.messages == () and context.tokens == 0


def test_a_trailing_system_message_is_left_out_and_the_last_user_turn_stays_current() -> None:
    request = _request(_turn("0"), _turn("1"), [Message(role="system", content="note")])
    context = _built(request)
    assert context.earlier_kept == 1 and _texts(context)[-1] == str(_turn("1")[1].content)


def test_empty_content_and_a_tool_result_with_no_call_before_it_are_handled() -> None:
    orphan = [Message(role="tool", content="late result", tool_call_id="gone")]
    nothing = Message(role="assistant", content=None)
    context = _built(_request(orphan, [Message(role="user", content="next"), nothing]))
    assert context.earlier_kept == 1 and context.messages[0].role == "tool"


def test_one_message_is_its_own_view() -> None:
    context = _built(_request([Message(role="user", content="hello")]))
    assert _texts(context) == ["hello"] and context.earlier_kept == 0


def test_a_leading_assistant_turn_is_its_own_earliest_exchange() -> None:
    lead = [Message(role="assistant", content="Welcome back")]
    request = _request(lead, _turn("1"))
    context = _built(request)
    assert context.earlier_kept == 1 and _texts(context)[0] == "Welcome back"
    assert _built(request, _tokens(request.messages[1:])).earlier_kept == 0


@pytest.mark.parametrize("role", ["system", "developer"])
def test_system_and_developer_messages_never_reach_the_analyzer_or_the_count(role: str) -> None:
    prompt = Message(role=role, content="s" * 4000)
    request = _request([prompt], _turn("0"), [Message(role=role, content="mid")], _turn("1"))
    context = _built(request, _tokens(_turn("0")) + _tokens(_turn("1")))
    assert all(m.role not in ("system", "developer") for m in context.messages)
    assert context.earlier_kept == 1 and len(context.messages) == 4


def test_only_system_messages_give_an_empty_view_that_is_not_over_budget() -> None:
    context = _built(_request([Message(role="system", content="be nice")]), 1)
    assert context.messages == () and context.tokens == 0 and context.earlier_kept == 0


@pytest.mark.parametrize("budget", [0, -1, 1.5, True, None, "100"])
def test_a_bad_budget_is_refused(budget: Any) -> None:
    with pytest.raises(ValueError):
        bounded_context(_request(_turn("0")), budget)


def test_an_image_part_adds_no_tokens() -> None:
    image = {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}
    with_image = Message(role="user", content=[{"type": "text", "text": "x" * 40}, image])  # type: ignore[arg-type]
    plain = Message(role="user", content="x" * 40)
    assert _built(_request([with_image])).tokens == _built(_request([plain])).tokens


def test_the_estimate_counts_each_exchange_and_sums_them() -> None:
    turns = [_turn("0"), _turn("1"), _turn("2")]
    assert _built(_request(*turns)).tokens == sum(_tokens(t) for t in turns)


def test_the_result_types_are_frozen() -> None:
    context = _built(_request(_turn("0")))
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.tokens = 1  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        OverBudget(1, 1).tokens = 2  # type: ignore[misc]


def test_no_model_is_called_and_nothing_is_wired_or_exported() -> None:
    source = inspect.getsource(core_context)
    assert not re.search(r"\b(dispatch|drivers|transport|adapters)\b", source)
    for name in ("RoutingContext", "OverBudget", "bounded_context"):
        assert not hasattr(primagate, name)
