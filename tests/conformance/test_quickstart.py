import inspect
import re
from pathlib import Path

import httpx2
import pytest

import primagate
import primagate.client as client_module
from primagate import PrimaGate
from primagate.schema import errors

README = (Path(__file__).parents[2] / "README.md").read_text(encoding="utf-8")
SAMPLES = re.findall(r"```python\n(.*?)```", README, flags=re.DOTALL)
REPLY = "Hello from the recorded response"


def _slug(heading: str) -> str:
    return re.sub(r"[^a-z0-9 -]", "", heading.lower()).strip().replace(" ", "-")


def test_readme_covers_the_quickstart_steps() -> None:
    assert "pip install primagate" in README
    assert "OPENAI_API_KEY" in README
    assert any("acompletion(" in s and "asyncio.run(" in s for s in SAMPLES), "async sample"
    assert any("completion(" in s and "acompletion(" not in s for s in SAMPLES), "sync sample"


def test_readme_shows_provider_switching_and_error_handling() -> None:
    assert any("gpt-" in s and "claude-" in s for s in SAMPLES), "one sample on both providers"
    handling = [s for s in SAMPLES if "UnsupportedParamsError" in s]
    assert handling, "an error-handling sample"
    assert all("drop_params=" in s and "dropped_params" in s for s in handling)


def test_samples_import_only_from_the_package_root() -> None:
    internal = re.compile(
        r"^\s*(from primagate\.|import primagate\.|from primagate import (schema|client))", re.M
    )
    assert not [s for s in SAMPLES if internal.search(s)]
    assert "primagate.schema" not in README


def test_every_error_class_is_exported_from_the_package_root() -> None:
    classes = [
        name
        for name, obj in inspect.getmembers(errors, inspect.isclass)
        if issubclass(obj, errors.PrimaGateError)
    ]
    assert "UnsupportedParamsError" in classes
    for name in classes:
        assert getattr(primagate, name) is getattr(errors, name), name
    # The root exports the errors and the entry points, nothing else yet.
    others = {"PrimaGate", "__version__", "acompletion", "completion"}
    assert set(primagate.__all__) == set(classes) | others


def test_readme_links_resolve_to_headings() -> None:
    anchors = {_slug(h) for h in re.findall(r"^#+\s+(.*)$", README, flags=re.MULTILINE)}
    links = re.findall(r"\]\(#([^)]+)\)", README)
    assert links, "the quickstart should link to its notes"
    assert set(links) <= anchors, set(links) - anchors


def _openai_reply() -> dict[str, object]:
    return {
        "id": "chatcmpl-1",
        "created": 1,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": REPLY},
            }
        ],
    }


def _anthropic_reply() -> dict[str, object]:
    return {
        "id": "msg_1",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": REPLY}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }


def _run(sample: str, monkeypatch: pytest.MonkeyPatch) -> list[httpx2.Request]:
    calls: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        if request.url.host == "api.anthropic.com":
            return httpx2.Response(200, json=_anthropic_reply())
        if request.url.host == "api.openai.com":
            return httpx2.Response(200, json=_openai_reply())
        raise AssertionError(f"sample reached an unexpected host: {request.url.host}")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-recorded")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-recorded")
    # A leftover module-level setting from another test would change what a sample does.
    monkeypatch.setattr(primagate, "drop_params", None)
    gate = PrimaGate(transport=httpx2.MockTransport(handle))
    monkeypatch.setattr(client_module, "_default", gate)
    try:
        exec(compile(sample, "README.md", "exec"), {"__name__": "__main__"})
    finally:
        gate.close()
    return calls


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.strip().splitlines()[0][:40])
def test_readme_samples_run_in_recorded_mode(
    sample: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _run(sample, monkeypatch)
    assert calls, "a sample should make at least one call"
    out = capsys.readouterr().out
    if "UnsupportedParamsError" in sample:
        # The error sample prints what it caught and dropped; the model reply follows the retry.
        assert "Caught:" in out and "Dropped: ['image']" in out
    assert REPLY in out


def test_samples_together_reach_both_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    hosts: set[str] = set()
    for sample in SAMPLES:
        hosts |= {c.url.host for c in _run(sample, monkeypatch)}
    assert hosts == {"api.openai.com", "api.anthropic.com"}


def test_the_error_sample_prints_the_dropped_feature(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = next(s for s in SAMPLES if "UnsupportedParamsError" in s)
    calls = _run(sample, monkeypatch)
    assert len(calls) == 1, "the first attempt must fail before any request"
    assert calls[0].url.host == "api.anthropic.com"
    assert "image_url" not in calls[0].content.decode()
    assert "Dropped: ['image']" in capsys.readouterr().out


def test_readme_documents_failure_handling() -> None:
    for term in ("AttemptsExhaustedError", "attempt_chain", "last_error", "skipped"):
        assert term in README, term
    sample = next(s for s in SAMPLES if "AttemptsExhaustedError" in s)
    # `d.error.type` in the sample is a dispatch row's error, not an attribute of the exception.
    sample_attributes = set(re.findall(r"(?<![\w.])error\.(\w+)", sample))
    instance = errors.AttemptsExhaustedError(
        "m", attempt_chain=[], last_error=errors.PrimaGateError("x")
    )
    assert sample_attributes and all(hasattr(instance, a) for a in sample_attributes)


def test_the_readme_groups_match_how_errors_are_classified() -> None:
    from primagate.core.classify import Fate, classify

    block = re.search(r"^- `Timeout`.*?(?=\n\n)", README, re.S | re.M)
    assert block
    groups = [g for g in block.group(0).removeprefix("- ").split("\n- ") if g.strip()]
    assert len(groups) == 3
    for group, fate in zip(groups, (Fate.RETRY, Fate.ADVANCE, Fate.RAISE), strict=True):
        names = re.findall(r"`(\w+(?:Error|Timeout))`|`(Timeout)`", group)
        assert names
        for name in {n for pair in names for n in pair if n}:
            assert classify(getattr(errors, name)("x")) is fate, name
