"""What the live tests share: the models a run spends on, a capped way to call, and helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx2
import pytest
from conftest import Caller, StreamCaller

from primagate import (
    BadRequestError,
    DispatchStarted,
    Event,
    PrimaGate,
    RoutingDecision,
)
from primagate.core.models import resolve_model
from primagate.schema.types import ChatCompletion, ChatCompletionChunk

PROVIDERS = ("openai", "anthropic")
KEY_ENV = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
# Every live request asks for at most this many tokens, so a full run has a known ceiling.
MAX_TOKENS = 50
# The cheapest current model on each provider. An environment variable swaps in another one when
# a provider retires these, without a code change.
DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-5-5"}
# Tier Routing only asks the Analyzer when two Tiers bind different models, so MEDIUM gets the
# next cheapest one. It answers only when the Analyzer picks MEDIUM.
DEFAULT_MEDIUM_MODELS = {"openai": "gpt-4.1-mini", "anthropic": "claude-sonnet-5-5"}

# A test that the cap cut short proves nothing either way, so it is skipped with this reason.
INCONCLUSIVE = f"inconclusive: cut at the {MAX_TOKENS}-token cap"

ASK = [{"role": "user", "content": "Reply with the single word: ready"}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Look up the current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]
# Asks for the tool call alone, so the answer starts with it and fits under the cap.
ASK_TOOL = [
    {
        "role": "user",
        "content": "Call the weather tool for Oslo now. Do not write any text before or after it.",
    }
]


class LiveConfigError(Exception):
    """The live run itself is set up wrongly, for example a bad `PRIMAGATE_LIVE_REQUIRE`."""


class MissingKeyError(LiveConfigError):
    """A provider the gate requires has no key, so its tests fail instead of skipping."""


class LiveOutage(Exception):
    """The provider failed in a way the SDK absorbed, such as an Analyzer it could not reach.

    Raised by a test so the report counts it as an outage rather than a failed assertion.
    """


def stop_if_cut(finish_reason: str | None) -> None:
    """Skip a test whose reply the token cap cut off, since it proves nothing either way."""
    if finish_reason == "length":
        pytest.skip(INCONCLUSIVE)


def model_env(provider: str, *, medium: bool = False) -> str:
    return f"PRIMAGATE_LIVE_{provider.upper()}_{'MEDIUM_' if medium else ''}MODEL"


def model_for(provider: str, env: Mapping[str, str] = os.environ, *, medium: bool = False) -> str:
    """The model to call: the default, or the override, which must be the provider's own."""
    name = model_env(provider, medium=medium)
    chosen = env.get(name, "").strip()
    if not chosen:
        return (DEFAULT_MEDIUM_MODELS if medium else DEFAULT_MODELS)[provider]
    # The tests name models without a prefix, so a prefix for this provider is dropped.
    prefix, _, rest = chosen.partition("/")
    bare = rest if prefix == provider and rest else chosen
    try:
        resolved, _ = resolve_model(bare)
    except BadRequestError:
        resolved = None
    if resolved != provider:
        raise LiveConfigError(f"{name} must name a {provider} model, without another prefix")
    return bare


def capped(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The call's arguments with `max_tokens` set and never above the cap."""
    out = dict(kwargs)
    out.setdefault("max_tokens", MAX_TOKENS)
    tokens = out["max_tokens"]
    if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 < tokens <= MAX_TOKENS:
        raise ValueError(f"a live call must ask for 1 to {MAX_TOKENS} tokens")
    return out


class LiveCaller(Caller):
    """The suite's sync-or-async caller, with every request capped and stream errors raised.

    A stream error is raised as it came, not turned into a failed assertion, so the report can
    tell a provider outage from a regression.
    """

    def __call__(self, gate: PrimaGate, **kwargs: Any) -> ChatCompletion:
        return super().__call__(gate, **capped(kwargs))

    def stream(self, gate: PrimaGate, **kwargs: Any) -> list[ChatCompletionChunk]:
        run = StreamCaller(self.mode)(gate, **capped(kwargs))
        if run.call_error is not None:
            raise run.call_error
        if run.read_error is not None:
            raise run.read_error
        return run.chunks


class Recorder:
    def __init__(self) -> None:
        self.seen: list[Event] = []
        # Dispatches the SDK refused after announcing them (an unsupported hint), which sent
        # nothing. A test that expects such a refusal says so with `refuse()`.
        self.refused = 0

    def refuse(self) -> None:
        self.refused += 1

    def __call__(self, event: Event) -> None:
        self.seen.append(event)

    @property
    def types(self) -> list[str]:
        return [e.event_type for e in self.seen]

    def of(self, kind: type) -> list[Any]:
        return [e for e in self.seen if isinstance(e, kind)]

    @property
    def requests(self) -> int:
        """Requests sent: each worker dispatch, and each question the Analyzer was asked."""
        asked = [d for d in self.of(RoutingDecision) if d.analyzer_prompt_version != "unknown"]
        return len(self.of(DispatchStarted)) - self.refused + len(asked)


class Live:
    """One live test's provider, key and model, and the gates it made."""

    def __init__(self, provider: str, key: str, model: str, medium_model: str) -> None:
        self.provider = provider
        self.model = model
        self.medium_model = medium_model
        self.events = Recorder()
        self._key = key
        self._gates: list[PrimaGate] = []

    def __repr__(self) -> str:
        # Shown in assertion output, so the key stays out.
        return f"Live({self.provider}, {self.model})"

    def gate(self, **kwargs: Any) -> PrimaGate:
        kwargs.setdefault("event_handler", self.events)
        made = PrimaGate(api_keys={self.provider: self._key}, **kwargs)
        self._gates.append(made)
        return made

    def close(self) -> None:
        for made in self._gates:
            made.close()


def shape(value: Any) -> Any:
    """Keys and value types only, so a real reply compares with a recorded one on shape."""
    if isinstance(value, dict):
        return {k: shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shape(v) for v in value]
    return type(value).__name__


def _recorded_body(provider: str, model: str) -> dict[str, Any]:
    if provider == "anthropic":
        return {
            "id": "msg_rec",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": "ready"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 2},
        }
    message = {"role": "assistant", "content": "ready"}
    return {
        "id": "chatcmpl-rec",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
    }


def recorded_reply(call: Caller, provider: str, model: str, **kwargs: Any) -> ChatCompletion:
    """The same call answered by a scripted reply, as the recorded suite answers it."""
    body = _recorded_body(provider, model)

    def handle(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json=body, headers={"date": "Thu, 01 Jan 1970 00:00:01 GMT"})

    gate = PrimaGate(
        api_keys={provider: "recorded-key"}, transport=httpx2.MockTransport(handle), **kwargs
    )
    try:
        return call(gate, **capped({"model": model, "messages": ASK}))
    finally:
        gate.close()


def joined(chunks: list[ChatCompletionChunk]) -> str:
    return "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
