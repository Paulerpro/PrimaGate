import asyncio
from typing import Any

import pytest

import primagate
from primagate import PrimaGate
from primagate.schema.types import ChatCompletion


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
