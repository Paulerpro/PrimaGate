import dataclasses
import importlib
import json
import socket
import sys
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx2
import pytest

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
LIVE_KEY = "sk-live-must-never-print"


@pytest.fixture(scope="module")
def demo() -> ModuleType:
    # `examples/` is a folder of scripts, not an installed package, so it goes on the path by hand.
    sys.path.insert(0, str(EXAMPLES))
    try:
        return importlib.import_module("forced_failure")
    finally:
        sys.path.remove(str(EXAMPLES))


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any attempt to reach the network fails the test instead of leaving the machine."""

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the demo tried to use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.mark.usefixtures("offline")
def test_the_recorded_demo_ends_every_scenario_as_expected(
    demo: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    assert demo.main([]) == 0
    out = capsys.readouterr().out
    assert "6/6 scenarios as expected (100%)" in out
    assert out.count("PASS") == 6 and "FAIL" not in out
    # Each layer shows in the printed Attempt Chains, the failed endpoint by its address.
    assert (
        "openai/gpt-4o-mini  https://openai-primary.invalid/v1: 2 retries, APIConnectionError; "
        "https://api.openai.com/v1: 0 retries, ok"
    ) in out
    assert "skipped: MEDIUM is not in enabled_tiers" in out
    assert "raised AttemptsExhaustedError" in out
    assert "raised APIConnectionError while iterating" in out


@pytest.mark.usefixtures("offline")
def test_the_recorded_output_holds_no_key_and_no_request(
    demo: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    demo.main([])
    out = capsys.readouterr().out
    assert demo.RECORDED_KEY not in out
    assert demo.QUESTION[0]["content"] not in out
    assert '"messages"' not in out


MISSES = [
    pytest.param(
        1,
        {"model": "openai/not-the-answering-model"},
        "expected model openai/not-the-answering-model, got openai/gpt-4.1-mini",
        id="model",
    ),
    pytest.param(
        1,
        {"chain": ("openai/gpt-4.1-mini  https://api.openai.com/v1: 0 retries, ok",)},
        "expected chain ['openai/gpt-4.1-mini  https://api.openai.com/v1: 0 retries, ok'], got",
        id="chain",
    ),
    pytest.param(
        1,
        {"sent": ("openai/gpt-4.1-mini",)},
        "expected to send ['openai/gpt-4.1-mini'], "
        "sent ['openai/gpt-4o-mini', 'openai/gpt-4.1-mini']",
        id="sent",
    ),
    pytest.param(
        1,
        {"outcome": "AttemptsExhaustedError"},
        "expected outcome AttemptsExhaustedError, got answered",
        id="outcome",
    ),
    pytest.param(
        5, {"while_iterating": False}, "expected the outcome from the call", id="while_iterating"
    ),
]


@pytest.mark.usefixtures("offline")
@pytest.mark.parametrize(("index", "change", "problem"), MISSES)
def test_a_scenario_that_misses_its_expectation_fails_the_run(
    demo: ModuleType,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    index: int,
    change: dict[str, Any],
    problem: str,
) -> None:
    real = demo.scenarios

    def broken(live: bool, key: str) -> list[Any]:
        found = list(real(live, key))
        wrong = dataclasses.replace(found[index].expected, **change)
        found[index] = dataclasses.replace(found[index], expected=wrong)
        return found

    monkeypatch.setattr(demo, "scenarios", broken)
    assert demo.main([]) == 1
    out = capsys.readouterr().out
    assert out.count("PASS") == 5 and out.count("FAIL") == 1
    assert problem in out
    assert "5/6 scenarios as expected (83%)" in out


def test_live_without_a_key_stops_before_any_call(
    demo: ModuleType, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def no_gate(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a gate was built without a key")

    monkeypatch.setattr(demo, "PrimaGate", no_gate)
    assert demo.main(["--live"]) == 2
    captured = capsys.readouterr()
    assert "OPENAI_API_KEY" in captured.err and captured.out == ""


@pytest.fixture
def runs(demo: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Stands in for the run, recording how main started it."""
    started: list[tuple[Any, ...]] = []

    def fake_run(*args: Any) -> int:
        started.append(args)
        return 0

    monkeypatch.setattr(demo, "run", fake_run)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return started


def test_live_takes_the_key_from_the_option_or_the_environment(
    demo: ModuleType, runs: list[tuple[Any, ...]], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert demo.main(["--live", "--api-key", "k"]) == 0
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert demo.main(["--live"]) == 0
    assert runs == [(True, "k"), (True, "k")]


@pytest.mark.parametrize("source", ["option", "environment"])
def test_a_blank_key_is_a_missing_key(
    demo: ModuleType,
    runs: list[tuple[Any, ...]],
    monkeypatch: pytest.MonkeyPatch,
    source: str,
) -> None:
    if source == "environment":
        monkeypatch.setenv("OPENAI_API_KEY", "   ")
        argv = ["--live"]
    else:
        argv = ["--live", "--api-key", "   "]
    assert demo.main(argv) == 2
    assert runs == []


def test_a_key_without_live_is_refused(demo: ModuleType, runs: list[tuple[Any, ...]]) -> None:
    with pytest.raises(SystemExit) as stopped:
        demo.main(["--api-key", "k"])
    assert stopped.value.code == 2 and runs == []


class StandIn:
    """Answers as OpenAI would for live mode's settings: names that do not exist are not found."""

    def __init__(self, demo: ModuleType) -> None:
        self.demo = demo
        self.unreachable = 0
        self.refused = 0
        self.answered: list[dict[str, Any]] = []
        self.transport = httpx2.MockTransport(self._handle)

    def _handle(self, request: httpx2.Request) -> httpx2.Response:
        if request.url.host.endswith(".invalid"):
            self.unreachable += 1
            raise httpx2.ConnectError("name does not resolve")
        body = json.loads(request.content)
        model = body["model"]
        if model.startswith("primagate-missing-"):
            self.refused += 1
            error = {"message": "no such model", "type": "invalid_request_error"}
            return httpx2.Response(404, json={"error": {**error, "code": "model_not_found"}})
        self.answered.append(body)
        answered: httpx2.Response = self.demo.answer(model, bool(body.get("stream")))
        return answered


@pytest.mark.usefixtures("offline")
def test_live_settings_end_as_expected_and_never_print_the_key(
    demo: ModuleType, capsys: pytest.CaptureFixture[str], delays: list[float]
) -> None:
    stand_in = StandIn(demo)
    assert demo.run(True, LIVE_KEY, stand_in.transport) == 0
    out = capsys.readouterr().out
    assert "5/5 scenarios as expected (100%)" in out
    assert "SKIPPED (recorded only)" in out
    assert out.count("PASS") == 5 and "FAIL" not in out
    assert LIVE_KEY not in out
    # The numbers examples/README.md states for a live run.
    assert stand_in.refused + len(stand_in.answered) == 14 and len(stand_in.answered) == 4
    assert stand_in.unreachable == 3
    assert all(body["max_tokens"] == 30 for body in stand_in.answered)
    # Live mode keeps the library's own backoff between retries.
    assert delays == [0.5, 1.0]


def test_the_demo_is_not_packaged() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["primagate"]
