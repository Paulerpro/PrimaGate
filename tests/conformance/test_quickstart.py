import re
from pathlib import Path

import httpx2
import pytest

import primagate.client as client_module
from primagate import PrimaGate

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


def test_readme_links_resolve_to_headings() -> None:
    anchors = {_slug(h) for h in re.findall(r"^#+\s+(.*)$", README, flags=re.MULTILINE)}
    links = re.findall(r"\]\(#([^)]+)\)", README)
    assert links, "the quickstart should link to its notes"
    assert set(links) <= anchors, set(links) - anchors


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.strip().splitlines()[0][:40])
def test_readme_samples_run_in_recorded_mode(
    sample: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        calls.append(request)
        return httpx2.Response(
            200,
            json={
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
            },
        )

    monkeypatch.setenv("OPENAI_API_KEY", "sk-recorded")
    monkeypatch.setattr(
        client_module, "_default", PrimaGate(transport=httpx2.MockTransport(handle))
    )
    exec(compile(sample, "README.md", "exec"), {"__name__": "__main__"})
    assert len(calls) == 1
    assert REPLY in capsys.readouterr().out
