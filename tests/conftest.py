import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

import primagate
from primagate import PrimaGate
from primagate.schema.types import ChatCompletion, ChatCompletionChunk


class Caller:
    """Makes one call in either mode so a test body is written once."""

    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __call__(self, gate: PrimaGate, **kwargs: Any) -> ChatCompletion:
        if self.mode == "sync":
            return gate.completion(**kwargs)
        return asyncio.run(gate.acompletion(**kwargs))

    def module(self, **kwargs: Any) -> ChatCompletion:
        if self.mode == "sync":
            return primagate.completion(**kwargs)
        return asyncio.run(primagate.acompletion(**kwargs))


@pytest.fixture(params=["sync", "async"])
def call(request: pytest.FixtureRequest) -> Caller:
    return Caller(request.param)


@dataclass
class StreamRun:
    """What a streamed call did, split by where a failure surfaced."""

    chunks: list[ChatCompletionChunk] = field(default_factory=list)
    call_error: Exception | None = None
    read_error: Exception | None = None
    closed_early: bool = False


class StreamCaller:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def __call__(
        self, gate: PrimaGate, *, stop_after: int | None = None, **kwargs: Any
    ) -> StreamRun:
        if self.mode == "sync":
            return self._sync(gate, stop_after, kwargs)
        return asyncio.run(self._async(gate, stop_after, kwargs))

    def _sync(self, gate: PrimaGate, stop_after: int | None, kwargs: dict[str, Any]) -> StreamRun:
        run = StreamRun()
        try:
            stream = gate.completion(stream=True, **kwargs)
        except Exception as exc:
            run.call_error = exc
            return run
        try:
            for chunk in stream:
                run.chunks.append(chunk)
                if stop_after is not None and len(run.chunks) >= stop_after:
                    stream.close()
                    run.closed_early = True
                    break
        except Exception as exc:
            run.read_error = exc
        return run

    async def _async(
        self, gate: PrimaGate, stop_after: int | None, kwargs: dict[str, Any]
    ) -> StreamRun:
        run = StreamRun()
        try:
            stream = await gate.acompletion(stream=True, **kwargs)
        except Exception as exc:
            run.call_error = exc
            return run
        try:
            async for chunk in stream:
                run.chunks.append(chunk)
                if stop_after is not None and len(run.chunks) >= stop_after:
                    await stream.aclose()
                    run.closed_early = True
                    break
        except Exception as exc:
            run.read_error = exc
        return run


@pytest.fixture(params=["sync", "async"])
def stream_call(request: pytest.FixtureRequest) -> StreamCaller:
    return StreamCaller(request.param)


@pytest.fixture
def delays(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record every sleep instead of waiting, and make jitter a fixed full-scale value."""
    waited: list[float] = []

    async def fake_async(seconds: float) -> None:
        waited.append(seconds)

    monkeypatch.setattr(time, "sleep", waited.append)
    monkeypatch.setattr(asyncio, "sleep", fake_async)
    monkeypatch.setattr(random, "random", lambda: 1.0)
    return waited
