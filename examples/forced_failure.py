"""Watch a request survive forced failures, one layer of the reliability chain at a time.

PrimaGate's chain runs in a fixed order: Provider Routing and retries, then Tier Failover, then
the Fallback List. Each scenario below forces a failure somewhere in it, prints what happened and
the Attempt Chain, and checks the outcome against what should happen.

    uv run python examples/forced_failure.py           # recorded: no network, no key
    uv run python examples/forced_failure.py --live    # real OpenAI calls, needs OPENAI_API_KEY

Only the public `primagate` API is used. Recorded mode replays provider replies through
httpx2's MockTransport, the same seam PrimaGate's own tests use.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx2

from primagate import (
    TIER_ROUTING_ALIAS,
    AttemptsExhaustedError,
    DispatchStarted,
    Event,
    PrimaGate,
    PrimaGateError,
)

QUESTION = [{"role": "user", "content": "In one short sentence, what is the capital of France?"}]
# Enough for a one-sentence answer, and it caps what a live run can cost.
MAX_TOKENS = 30
OPENAI = "https://api.openai.com/v1"
# `.invalid` never resolves (RFC 6761), so a live run fails here before any byte, key included,
# is sent anywhere.
UNREACHABLE = "https://openai-primary.invalid/v1"
RECORDED_KEY = "recorded-demo-key"
# PrimaGate refuses a zero backoff; a millisecond keeps the recorded run instant. Live mode keeps
# the library's default, so a real rate limit is not retried at once.
RECORDED_BACKOFF: dict[str, Any] = {"backoff_base": 0.001, "backoff_max": 0.001}

ROLES = ("small", "medium", "powerful", "fallback", "analyzer")
# Recorded mode: real model names, made to fail by the script.
RECORDED_CAST = {
    "small": "gpt-4o-mini",
    "medium": "gpt-4.1-mini",
    "powerful": "gpt-4o",
    "fallback": "gpt-4.1",
    "analyzer": "gpt-4.1-nano",
}
# Live mode: the one layer meant to answer gets a real, cheap model; every other name does not
# exist, so OpenAI refuses it for free.
LIVE_MODEL = "gpt-4o-mini"
RECORDED_ANSWER = "The capital of France is Paris."
NOT_ENABLED = "MEDIUM is not in enabled_tiers"


# --- what a scenario is ---


@dataclass(frozen=True)
class Script:
    """How the recorded provider behaves for one scenario."""

    # The Analyzer's answer: which Tier, with what confidence.
    verdict: str = "SMALL 0.9"
    # Models that answer 503 Service Unavailable.
    down: frozenset[str] = frozenset()
    # Hosts that cannot be reached at all.
    unreachable: frozenset[str] = frozenset()
    # Models whose stream breaks after its first chunk.
    drops: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Expected:
    # "answered", or the class name of the error the caller sees.
    outcome: str
    # The model that answered, or was streaming when the error came.
    model: str | None
    # Each Attempt Chain row as printed: every endpoint with its retries and how it ended, or the
    # reason the row was skipped.
    chain: tuple[str, ...]
    # Every model sent, in order, a model repeated only when sent again after another one.
    sent: tuple[str, ...]
    while_iterating: bool = False


@dataclass(frozen=True)
class Scenario:
    name: str
    setup: str
    request: str
    expected: Expected
    gate: dict[str, Any]
    call: dict[str, Any]
    script: Script = Script()
    recorded_only: bool = False


@dataclass
class Got:
    outcome: str
    model: str | None
    chain: tuple[str, ...]
    sent: tuple[str, ...]
    while_iterating: bool = False
    reply: str | None = None
    last_error: str | None = None
    chunks: int = 0


# --- the six scenarios ---


def _cast(live: bool, up: str | None) -> dict[str, str]:
    if not live:
        return dict(RECORDED_CAST)
    return {role: LIVE_MODEL if role == up else f"primagate-missing-{role}" for role in ROLES}


def _routed(
    cast: dict[str, str],
    enabled: tuple[str, ...] = ("SMALL", "MEDIUM", "POWERFUL"),
    fallbacks: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "tier_bindings": {
            "SMALL": {"model": cast["small"]},
            "MEDIUM": {"model": cast["medium"]},
            "POWERFUL": {"model": cast["powerful"]},
        },
        "policy": {
            "enabled_tiers": list(enabled),
            "confidence_threshold": 0.8,
            "context_token_budget": 1000,
            # Live mode cannot script the Analyzer, so its model does not exist and routing falls
            # back here: either way SMALL is the resolved Tier.
            "interim_binding": "SMALL",
            "ambiguous_context_default": "SMALL",
        },
        "analyzer": cast["analyzer"],
        "default_fallbacks": list(fallbacks) or None,
    }


def _name(model: str) -> str:
    return f"openai/{model}"


def _row(model: str, *dispatches: tuple[str, int, str]) -> str:
    """A sent row: each dispatch as (endpoint, retries, error type or "ok")."""
    return f"{_name(model)}  " + "; ".join(f"{e}: {r} retries, {o}" for e, r, o in dispatches)


def _skipped(model: str, reason: str) -> str:
    return f"{_name(model)}  skipped: {reason}"


def _down(cast: dict[str, str], *roles: str) -> frozenset[str]:
    return frozenset(cast[role] for role in roles)


def scenarios(live: bool, key: str) -> list[Scenario]:
    # How a failing model is refused: a missing name live, an outage when recorded.
    fail = "NotFoundError" if live else "ServiceUnavailableError"
    endpoint = LIVE_MODEL if live else RECORDED_CAST["small"]
    s2 = _cast(live, "medium")
    s3 = _cast(live, "fallback")
    s4 = _cast(live, None)
    s5 = _cast(live, "medium")
    s6 = _cast(live, "small")
    routed = {"model": TIER_ROUTING_ALIAS}
    streamed = {"model": TIER_ROUTING_ALIAS, "stream": True}
    return [
        Scenario(
            name="Endpoint down",
            setup="the primary endpoint cannot be reached; a redundant endpoint is declared",
            request=f"plain call to {endpoint}",
            expected=Expected(
                "answered",
                _name(endpoint),
                (_row(endpoint, (UNREACHABLE, 2, "APIConnectionError"), (OPENAI, 0, "ok")),),
                (_name(endpoint),),
            ),
            gate={
                "base_urls": {"openai": UNREACHABLE},
                "provider_routing": {endpoint: [{"base_url": OPENAI, "api_key": key}]},
            },
            call={"model": endpoint},
            script=Script(unreachable=frozenset({httpx2.URL(UNREACHABLE).host})),
        ),
        Scenario(
            name="Resolved Tier down",
            setup="routing resolves SMALL, whose model fails; MEDIUM is enabled",
            request="Tier Routing (primagate/auto)",
            expected=Expected(
                "answered",
                _name(s2["medium"]),
                (_row(s2["small"], (OPENAI, 0, fail)), _row(s2["medium"], (OPENAI, 0, "ok"))),
                (_name(s2["small"]), _name(s2["medium"])),
            ),
            gate=_routed(s2),
            call=routed,
            script=Script(down=_down(s2, "small")),
        ),
        Scenario(
            name="MEDIUM unavailable",
            setup="SMALL fails, MEDIUM is not in enabled_tiers, a Fallback List is declared",
            request="Tier Routing (primagate/auto)",
            expected=Expected(
                "answered",
                _name(s3["fallback"]),
                (
                    _row(s3["small"], (OPENAI, 0, fail)),
                    _skipped(s3["medium"], NOT_ENABLED),
                    _row(s3["fallback"], (OPENAI, 0, "ok")),
                ),
                (_name(s3["small"]), _name(s3["fallback"])),
            ),
            gate=_routed(s3, enabled=("SMALL", "POWERFUL"), fallbacks=(s3["fallback"],)),
            call=routed,
            script=Script(down=_down(s3, "small")),
        ),
        Scenario(
            name="Everything down",
            setup="SMALL, MEDIUM and the one Fallback List entry all fail",
            request="Tier Routing (primagate/auto)",
            expected=Expected(
                "AttemptsExhaustedError",
                None,
                tuple(_row(s4[r], (OPENAI, 0, fail)) for r in ("small", "medium", "fallback")),
                tuple(_name(s4[r]) for r in ("small", "medium", "fallback")),
            ),
            gate=_routed(s4, fallbacks=(s4["fallback"],)),
            call=routed,
            script=Script(down=_down(s4, "small", "medium", "fallback")),
        ),
        Scenario(
            name="Stream fails before the first chunk",
            setup="SMALL fails before sending anything; MEDIUM is enabled",
            request="Tier Routing (primagate/auto), streamed",
            expected=Expected(
                "answered",
                _name(s5["medium"]),
                (_row(s5["small"], (OPENAI, 0, fail)), _row(s5["medium"], (OPENAI, 0, "ok"))),
                (_name(s5["small"]), _name(s5["medium"])),
            ),
            gate=_routed(s5),
            call=streamed,
            script=Script(down=_down(s5, "small")),
        ),
        Scenario(
            name="Stream fails after the first chunk",
            setup="SMALL's stream breaks after one chunk; a Fallback List is declared",
            request="Tier Routing (primagate/auto), streamed",
            expected=Expected(
                "APIConnectionError",
                _name(s6["small"]),
                (_row(s6["small"], (OPENAI, 0, "APIConnectionError")),),
                (_name(s6["small"]),),
                while_iterating=True,
            ),
            gate=_routed(s6, fallbacks=(s6["fallback"],)),
            call=streamed,
            script=Script(drops=_down(s6, "small")),
            # A real provider cannot be made to drop a stream mid-way by configuration alone.
            recorded_only=True,
        ),
    ]


# --- the recorded provider ---


def _sse(data: dict[str, Any] | str) -> bytes:
    return f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n".encode()


def _chunk(model: str, delta: dict[str, Any], finish: str | None = None) -> bytes:
    choice = {"index": 0, "delta": delta, "finish_reason": finish}
    return _sse({"id": "rec", "created": 0, "model": model, "choices": [choice]})


def _first_chunk(model: str) -> bytes:
    return _chunk(model, {"role": "assistant", "content": RECORDED_ANSWER})


def _rest_of_stream(model: str) -> bytes:
    usage = {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
    tail = {"id": "rec", "created": 0, "model": model, "choices": [], "usage": usage}
    return _chunk(model, {}, "stop") + _sse(tail) + _sse("[DONE]")


SSE = {"content-type": "text/event-stream"}


def answer(model: str, stream: bool, text: str = RECORDED_ANSWER) -> httpx2.Response:
    """An OpenAI answer, whole or streamed."""
    if stream:
        return httpx2.Response(
            200, content=_first_chunk(model) + _rest_of_stream(model), headers=SSE
        )
    message = {"role": "assistant", "content": text}
    body = {
        "id": "rec",
        "created": 0,
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }
    return httpx2.Response(200, json=body)


class _Dropping(httpx2.SyncByteStream):
    """Delivers one chunk, then fails the way a dropped connection does."""

    def __init__(self, first: bytes) -> None:
        self._first = first

    def __iter__(self) -> Iterator[bytes]:
        yield self._first
        raise httpx2.ReadError("connection dropped")


def recorded(script: Script, analyzer: str) -> httpx2.MockTransport:
    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.host in script.unreachable:
            raise httpx2.ConnectError("name does not resolve")
        body = json.loads(request.content)
        model, stream = body["model"], bool(body.get("stream"))
        if model == analyzer:
            return answer(model, False, script.verdict)
        if model in script.down:
            error = {"message": "overloaded", "type": "server_error", "code": None}
            return httpx2.Response(503, json={"error": error})
        if model in script.drops and stream:
            return httpx2.Response(200, stream=_Dropping(_first_chunk(model)), headers=SSE)
        return answer(model, stream)

    return httpx2.MockTransport(handle)


# --- running and reporting ---


def _rows(chain: list[Any], primary: str) -> tuple[str, ...]:
    """One printable line per Attempt Chain row."""
    lines: list[str] = []
    for row in chain:
        if row.skipped is not None:
            lines.append(f"{row.model}  skipped: {row.skipped}")
            continue
        # The primary endpoint is recorded by its provider's name; its address says which failed.
        # Error types only: an error's text can echo the request.
        parts = [
            f"{primary if d.endpoint == 'openai' else d.endpoint}: {d.retries} retries, "
            f"{d.error.type if d.error else 'ok'}"
            for d in row.dispatches
        ]
        lines.append(f"{row.model}  " + "; ".join(parts))
    return tuple(lines)


def _last_sent(chain: list[Any]) -> str | None:
    sent = [row.model for row in chain if row.skipped is None]
    return sent[-1] if sent else None


def _first_line(text: str | None) -> str:
    lines = (text or "").strip().splitlines()
    return lines[0][:100] if lines else "(empty reply)"


def run_scenario(
    scenario: Scenario, key: str, transport: httpx2.BaseTransport | None, live: bool
) -> Got:
    sent: list[str] = []

    # Worker dispatches only: the Analyzer's question is not a dispatch event. Retries of one
    # model collapse into one entry, so a model sent again later still shows twice.
    def watch(event: Event) -> None:
        if isinstance(event, DispatchStarted) and sent[-1:] != [event.model]:
            sent.append(event.model)

    gate = PrimaGate(
        api_keys={"openai": key},
        transport=transport,
        event_handler=watch,
        **({} if live else RECORDED_BACKOFF),
        **{k: v for k, v in scenario.gate.items() if v is not None},
    )
    primary = scenario.gate.get("base_urls", {}).get("openai", OPENAI)
    params = {"messages": QUESTION, "max_tokens": MAX_TOKENS, **scenario.call}
    try:
        return _outcome(gate, params, sent, primary)
    except PrimaGateError as error:
        last = error.last_error if isinstance(error, AttemptsExhaustedError) else None
        return Got(
            type(error).__name__,
            None,
            _rows(error.attempt_chain, primary),
            tuple(sent),
            last_error=type(last).__name__ if last is not None else None,
        )
    finally:
        gate.close()


def _outcome(gate: PrimaGate, params: dict[str, Any], sent: list[str], primary: str) -> Got:
    if not params.get("stream"):
        response = gate.completion(**params)
        if response.metadata is None:
            raise RuntimeError("the response carries no metadata")
        chain = response.metadata.attempt_chain
        text = response.choices[0].message.content if response.choices else None
        return Got("answered", _last_sent(chain), _rows(chain, primary), tuple(sent), reply=text)
    stream = gate.completion(**params)
    chunks: list[Any] = []
    try:
        for chunk in stream:
            chunks.append(chunk)
    except PrimaGateError as error:
        return Got(
            type(error).__name__,
            _last_sent(error.attempt_chain),
            _rows(error.attempt_chain, primary),
            tuple(sent),
            while_iterating=True,
            chunks=len(chunks),
        )
    meta = chunks[-1].metadata if chunks else None
    if meta is None:
        raise RuntimeError("the stream ended without metadata on its last chunk")
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    return Got(
        "answered",
        _last_sent(meta.attempt_chain),
        _rows(meta.attempt_chain, primary),
        tuple(sent),
        reply=text,
        chunks=len(chunks),
    )


def check(expected: Expected, got: Got) -> list[str]:
    problems = []
    if got.outcome != expected.outcome:
        problems.append(f"expected outcome {expected.outcome}, got {got.outcome}")
    if got.model != expected.model:
        problems.append(f"expected model {expected.model}, got {got.model}")
    if got.chain != expected.chain:
        problems.append(f"expected chain {list(expected.chain)}, got {list(got.chain)}")
    if got.sent != expected.sent:
        problems.append(f"expected to send {list(expected.sent)}, sent {list(got.sent)}")
    if got.while_iterating != expected.while_iterating:
        where = "while iterating" if expected.while_iterating else "from the call"
        problems.append(f"expected the outcome {where}")
    return problems


def _describe(got: Got) -> str:
    if got.outcome == "answered":
        streamed = f" (streamed, {got.chunks} chunks)" if got.chunks else ""
        return f"answered by {got.model}{streamed}: {_first_line(got.reply)!r}"
    if got.while_iterating:
        return f"raised {got.outcome} while iterating, after {got.chunks} chunk(s) from {got.model}"
    last = f" (last error: {got.last_error})" if got.last_error else ""
    return f"raised {got.outcome}{last}"


def run(live: bool, key: str, transport: httpx2.BaseTransport | None = None) -> int:
    """Run every scenario and print the report; 0 only when each ends as expected."""
    logger = logging.getLogger("primagate")
    previous = logger.level
    # The forced failures log warnings such as `routing_degraded`; the report already shows them.
    logger.setLevel(logging.ERROR)
    try:
        return _run(live, key, transport)
    finally:
        logger.setLevel(previous)


def _run(live: bool, key: str, transport: httpx2.BaseTransport | None) -> int:
    mode = "live: real OpenAI calls" if live else "recorded: scripted replies, no network"
    print(f"PrimaGate forced-failure demo ({mode})\n")
    found = scenarios(live, key)
    passed = ran = 0
    for number, scenario in enumerate(found, 1):
        print(f"[{number}/{len(found)}] {scenario.name}")
        print(f"  setup:   {scenario.setup}")
        print(f"  request: {scenario.request}")
        if live and scenario.recorded_only:
            print("  SKIPPED (recorded only)\n")
            continue
        wire = transport if live else recorded(scenario.script, scenario.gate.get("analyzer", ""))
        try:
            got = run_scenario(scenario, key, wire, live)
        except Exception as error:
            # Anything else, a setup the library now refuses included, is this scenario's
            # failure, not the end of the run. Its type only: its text could echo the request.
            got = Got(type(error).__name__, None, (), ())
        problems = check(scenario.expected, got)
        print(f"  outcome: {_describe(got)}")
        print("  attempt chain:")
        for index, line in enumerate(got.chain, 1):
            print(f"    {index}. {line}")
        ran += 1
        if problems:
            print("  FAIL")
            for problem in problems:
                print(f"    {problem}")
        else:
            passed += 1
            print("  PASS")
        print()
    share = 100 * passed / ran if ran else 0
    print(f"{passed}/{ran} scenarios as expected ({share:.0f}%)")
    skipped = len(found) - ran
    if skipped:
        print(f"{skipped} skipped (recorded only)")
    return 0 if ran and passed == ran else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Force failures and watch PrimaGate recover.")
    parser.add_argument("--live", action="store_true", help="send real requests to OpenAI")
    parser.add_argument("--api-key", default=None, help="else OPENAI_API_KEY is read")
    args = parser.parse_args(argv)
    if not args.live:
        if args.api_key is not None:
            parser.error("--api-key is only used with --live")
        return run(False, RECORDED_KEY)
    key = (args.api_key or os.environ.get("OPENAI_API_KEY", "")).strip()
    if not key:
        # Before any call, so a missing key costs nothing.
        print("No OpenAI key: set OPENAI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2
    return run(True, key)


if __name__ == "__main__":
    sys.exit(main())
