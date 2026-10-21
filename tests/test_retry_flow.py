import ast
from pathlib import Path

import pytest

import primagate.core
from primagate.core import steps
from primagate.core.call import first_action, next_action
from primagate.schema import errors
from primagate.schema.types import (
    ChatCompletion,
    ChatCompletionRequest,
    Choice,
    DispatchRecord,
    ErrorInfo,
    Message,
)
from primagate.schema.wire import Capabilities, Endpoint, RetryPolicy

B = Endpoint("https://b.test/v1")
C = Endpoint("https://c.test/v1")
POLICY = RetryPolicy(max_retries=2, backoff_base=0.5, backoff_max=1.5)
CAPS = {"openai": Capabilities()}
ROUTING = {("openai", "gpt-4o"): (B, C)}


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="gpt-4o", messages=[Message(role="user", content="hi")])


def _start(routing: dict | None = ROUTING) -> steps.Dispatch:  # type: ignore[type-arg]
    action = first_action(_request(), CAPS, routing=routing, retry=POLICY)
    assert isinstance(action, steps.Dispatch)
    return action


def _fail(action: steps.Action, error: errors.PrimaGateError) -> steps.Action:
    return next_action(action, steps.DispatchOutcome(error=error))


def _ok() -> ChatCompletion:
    message = Message(role="assistant", content="yo")
    return ChatCompletion(
        id="c", created=1, model="gpt-4o", choices=[Choice(index=0, message=message)]
    )


def test_a_listed_model_carries_its_route() -> None:
    action = _start()
    assert action.route is not None and action.route.redundant == (B, C)
    assert action.endpoint is None and action.position == 0 and action.retries == 0


def test_an_unlisted_model_dispatches_directly() -> None:
    assert _start(routing={}).route is None
    assert _start(routing=None).route is None


def test_routing_off_never_retries_or_advances() -> None:
    action = _start(routing={})
    err = errors.ServiceUnavailableError("down", status_code=503)
    result = _fail(action, err)
    assert isinstance(result, steps.Raise) and result.error is err


def test_a_retry_error_asks_for_randomness_before_it_sleeps() -> None:
    result = _fail(_start(), errors.RateLimitError("slow", status_code=429))
    assert isinstance(result, steps.Random)
    assert result.resume.retries == 1 and result.resume.position == 0
    sleep = next_action(result, steps.RandomOutcome(1.0))
    assert isinstance(sleep, steps.Sleep) and sleep.resume is result.resume
    assert next_action(sleep, steps.SleepOutcome()) is result.resume


@pytest.mark.parametrize(
    ("retry", "r", "expected"),
    [(1, 1.0, 0.5), (2, 1.0, 1.0), (3, 1.0, 1.5), (4, 1.0, 1.5), (1, 0.0, 0.25), (2, 0.5, 0.75)],
)
def test_delay_doubles_to_the_cap_with_equal_jitter(retry: int, r: float, expected: float) -> None:
    resume = steps.Dispatch("openai", _request(), route=steps.Route((B,), POLICY), retries=retry)
    sleep = next_action(steps.Random(resume), steps.RandomOutcome(r))
    assert isinstance(sleep, steps.Sleep) and sleep.seconds == pytest.approx(expected)


def test_spent_retries_move_to_the_next_endpoint_without_a_delay() -> None:
    action = _start()
    for _ in range(POLICY.max_retries):
        random = _fail(action, errors.Timeout("t"))
        assert isinstance(random, steps.Random)
        sleep = next_action(random, steps.RandomOutcome(1.0))
        action = next_action(sleep, steps.SleepOutcome())  # type: ignore[assignment]
    advanced = _fail(action, errors.Timeout("last"))
    assert isinstance(advanced, steps.Dispatch)
    assert advanced.endpoint == B and advanced.position == 1 and advanced.retries == 0
    assert advanced.history == (DispatchRecord("openai", 2, ErrorInfo("Timeout", "last", None)),)


@pytest.mark.parametrize(
    "error",
    [errors.AuthenticationError("no", status_code=401), errors.NotFoundError("gone")],
)
def test_an_advance_error_skips_retries(error: errors.PrimaGateError) -> None:
    advanced = _fail(_start(), error)
    assert isinstance(advanced, steps.Dispatch) and advanced.endpoint == B
    assert advanced.history[0].retries == 0


@pytest.mark.parametrize("error", [errors.BadRequestError("bad"), errors.ImageFetchError("img")])
def test_a_raise_error_stops_the_flow(error: errors.PrimaGateError) -> None:
    result = _fail(_start(), error)
    assert isinstance(result, steps.Raise) and result.error is error
    assert len(error.attempt_chain) == 1


def test_the_last_endpoint_raises_its_error_with_every_row() -> None:
    action = _start()
    action = _fail(action, errors.AuthenticationError("a"))  # type: ignore[assignment]
    action = _fail(action, errors.AuthenticationError("b"))  # type: ignore[assignment]
    last = errors.AuthenticationError("c")
    result = _fail(action, last)
    assert isinstance(result, steps.Raise) and result.error is last
    rows = last.attempt_chain
    assert len(rows) == 1
    assert [d.endpoint for d in rows[0].dispatches] == ["openai", B.base_url, C.base_url]
    assert [d.error.message for d in rows[0].dispatches if d.error] == ["a", "b", "c"]


def test_a_success_after_a_retry_records_the_retries() -> None:
    action = _fail(_start(), errors.ServiceUnavailableError("down", status_code=503))
    assert isinstance(action, steps.Random)
    sleep = next_action(action, steps.RandomOutcome(1.0))
    retry = next_action(sleep, steps.SleepOutcome())
    assert isinstance(retry, steps.Dispatch)
    done = next_action(retry, steps.DispatchOutcome(response=_ok()))
    assert isinstance(done, steps.Return) and done.response.metadata is not None
    (attempt,) = done.response.metadata.attempt_chain
    assert attempt.dispatches == [DispatchRecord("openai", 1)]


def test_the_core_has_no_clock_or_random_source() -> None:
    forbidden = {"time", "random", "asyncio", "secrets"}
    for path in Path(primagate.core.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {(node.module or "").split(".")[0]}
            else:
                continue
            assert not names & forbidden, f"{path.name} imports {names & forbidden}"
