import asyncio
from collections.abc import Callable
from typing import Any

import httpx2
import pytest

from primagate.core.context import RoutingContext
from primagate.drivers.async_ import AsyncDriver
from primagate.drivers.providers import parse_fallbacks, parse_provider_routing
from primagate.drivers.sync import SyncDriver
from primagate.schema.analyzer import Analysis, AnalyzerFailure, Verdict
from primagate.schema.errors import BadRequestError
from primagate.schema.policy import TIER_ROUTING_ALIAS, Policy
from primagate.schema.profile import ProfileContent
from primagate.schema.tiers import Tier
from primagate.schema.types import Message

POLICY = Policy(
    enabled_tiers=(Tier.SMALL, Tier.MEDIUM),
    confidence_threshold=0.8,
    context_token_budget=1000,
    interim_binding=Tier.MEDIUM,
    ambiguous_context_default=Tier.SMALL,
)
PROFILE = ProfileContent(purpose="Answers order questions.", unknowns=("constraints",))
VIEW = RoutingContext((Message(role="user", content="where is order 12"),), 0, 5)
KEYS = {"openai": "sk-o", "anthropic": "sk-a"}


def _openai(text: str) -> httpx2.Response:
    body = {
        "id": "c",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 4, "total_tokens": 104},
    }
    return httpx2.Response(200, json=body)


def _anthropic(text: str) -> httpx2.Response:
    body = {
        "id": "m",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 4},
    }
    return httpx2.Response(200, json=body)


def _down(request: httpx2.Request) -> httpx2.Response:
    return httpx2.Response(503, json={"error": {"message": "x", "type": "server_error"}})


class Harness:
    """Drives `analyze` in either mode over a mock that records every send."""

    def __init__(
        self,
        mode: str,
        reply: Callable[[httpx2.Request], httpx2.Response],
        keys: dict[str, str] | None = None,
    ) -> None:
        self.mode = mode
        self.sent: list[httpx2.Request] = []

        def handle(request: httpx2.Request) -> httpx2.Response:
            self.sent.append(request)
            return reply(request)

        common: dict[str, Any] = {
            "api_keys": KEYS if keys is None else keys,
            "timeout": 5.0,
            "routing": parse_provider_routing(
                {"gpt-4o-mini": [{"base_url": "https://b.test/v1", "api_key": "sk-b"}]}
            ),
            "fallbacks": parse_fallbacks({"gpt-4o-mini": ["claude-sonnet-5-5"]}, None),
        }
        if mode == "sync":
            self.driver: Any = SyncDriver.with_builtin_adapters(
                http_transport=httpx2.MockTransport(handle), **common
            )
        else:
            self.driver = AsyncDriver.with_builtin_adapters(
                http_transport=httpx2.MockTransport(handle), **common
            )

    def analyze(self, model: str = "gpt-4o-mini", view: RoutingContext = VIEW) -> Analysis:
        if self.mode == "sync":
            result: Analysis = self.driver.analyze(model, PROFILE, view, POLICY)
            return result
        return asyncio.run(self.driver.analyze(model, PROFILE, view, POLICY))


@pytest.fixture(params=["sync", "async"])
def mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.mark.fr("FR19", "FR20")
def test_a_clean_answer_is_read_and_sent_once_to_the_analyzer_alone(mode: str) -> None:
    h = Harness(mode, lambda r: _openai("MEDIUM 0.82"))
    result = h.analyze()
    assert result.outcome == Verdict(Tier.MEDIUM, 0.82, "accepted", False)
    assert [r.url.host for r in h.sent] == ["api.openai.com"]


@pytest.mark.fr("FR19")
def test_the_analyzer_cost_is_its_own_line(mode: str) -> None:
    result = Harness(mode, lambda r: _openai("SMALL 0.9")).analyze()
    assert result.usage is not None and result.usage.cost is not None
    assert (result.usage.provider, result.usage.model) == ("openai", "gpt-4o-mini")
    assert result.usage.token_counts is not None and result.usage.token_counts.total_tokens == 104


@pytest.mark.fr("FR20")
def test_below_the_threshold_it_downgrades_without_failing(mode: str) -> None:
    result = Harness(mode, lambda r: _openai("MEDIUM 0.3")).analyze()
    assert result.outcome == Verdict(Tier.SMALL, 0.3, "ambiguous_context_default", True)


@pytest.mark.parametrize(
    "text", ["I would say MEDIUM 0.9", "medium 0.9", "MEDIUM 90%", "MEDIUM 0.9. Because...", ""]
)
@pytest.mark.fr("FR20")
def test_a_label_in_longer_text_is_never_extracted(mode: str, text: str) -> None:
    result = Harness(mode, lambda r: _openai(text)).analyze()
    assert result.outcome == AnalyzerFailure("malformed_output")
    assert result.usage is not None


@pytest.mark.fr("FR19")
def test_a_failing_provider_is_a_failure_with_one_send_and_no_retry_or_fallback(mode: str) -> None:
    h = Harness(mode, _down)
    result = h.analyze()
    assert result == Analysis(AnalyzerFailure("provider_error"), None)
    assert len(h.sent) == 1 and h.sent[0].url.host == "api.openai.com"


@pytest.mark.fr("FR20")
def test_a_not_enabled_tier_is_a_failure(mode: str) -> None:
    result = Harness(mode, lambda r: _openai("POWERFUL 0.99")).analyze()
    assert result.outcome == AnalyzerFailure("tier_not_enabled")


@pytest.mark.fr("FR19")
def test_the_analyzer_may_be_on_another_provider(mode: str) -> None:
    h = Harness(mode, lambda r: _anthropic("SMALL 0.95"))
    result = h.analyze("claude-sonnet-5-5")
    assert result.outcome == Verdict(Tier.SMALL, 0.95, "accepted", False)
    assert [r.url.host for r in h.sent] == ["api.anthropic.com"]


def test_an_empty_view_sends_nothing(mode: str) -> None:
    h = Harness(mode, lambda r: _openai("SMALL 1"))
    result = h.analyze(view=RoutingContext((), 0, 0))
    assert result == Analysis(AnalyzerFailure("no_conversation"), None) and h.sent == []


@pytest.mark.fr("FR19")
def test_the_alias_cannot_be_the_analyzer(mode: str) -> None:
    h = Harness(mode, lambda r: _openai("SMALL 1"))
    with pytest.raises(BadRequestError):
        h.analyze(TIER_ROUTING_ALIAS)
    assert h.sent == []


@pytest.mark.fr("FR19")
def test_the_request_carries_the_profile_and_conversation_and_no_tools(mode: str) -> None:
    h = Harness(mode, lambda r: _openai("SMALL 1"))
    h.analyze()
    body = h.sent[0].content.decode()
    assert "Answers order questions." in body and "where is order 12" in body
    assert '"tools"' not in body and '"stream"' not in body


@pytest.mark.fr("FR19")
def test_a_missing_key_is_a_failure_not_a_raise(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    h = Harness(mode, lambda r: _openai("SMALL 1"), keys={})
    assert h.analyze() == Analysis(AnalyzerFailure("provider_error"), None) and h.sent == []


@pytest.mark.fr("FR19")
def test_a_dropped_connection_is_a_failure_with_one_attempt(mode: str) -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down")

    h = Harness(mode, refuse)
    assert h.analyze() == Analysis(AnalyzerFailure("provider_error"), None)
    assert len(h.sent) == 1
