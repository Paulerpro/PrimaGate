"""The live gate's plugin, tested offline: key handling, failure classification and the report.

Nothing here reaches a provider. The end-to-end tests run a throwaway suite of fake live tests in
a separate pytest process, with the real plugin files copied in.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import httpx2
import pytest
from live import conftest as plugin
from live import support

import primagate
from primagate import (
    APIConnectionError,
    APIError,
    AttemptsExhaustedError,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    PermissionDeniedError,
    PrimaGateError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    UnsupportedParamsError,
)

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
NOW = datetime(2026, 11, 19, 6, 0, tzinfo=UTC)
# Stand-ins for what must never reach the report: a key, the prompt and the provider's reply.
FAKE_KEY = "sk-live-report-FAKE-KEY-0123456789"
PROMPT = "PROMPT-TEXT-never-reported"
REPLY = "REPLY-TEXT-never-reported"
ENV_NAMES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "PRIMAGATE_LIVE_REQUIRE",
    "PRIMAGATE_LIVE_REPORT",
    "PYTEST_ADDOPTS",
)


def _exhausted(last: PrimaGateError) -> AttemptsExhaustedError:
    return AttemptsExhaustedError("all failed", attempt_chain=[], last_error=last)


# --- classification ---


@pytest.mark.parametrize(
    "error",
    [
        Timeout("slow"),
        RateLimitError("429", status_code=429),
        ServiceUnavailableError("503", status_code=503),
        APIConnectionError("reset"),
        APIError("500", status_code=500),
        APIError("529", status_code=529),
        APIError("stream ended before its last event"),
        support.LiveOutage("the Analyzer's provider could not answer"),
        _exhausted(ServiceUnavailableError("503", status_code=503)),
    ],
    ids=lambda e: type(e).__name__ + str(getattr(e, "status_code", "") or ""),
)
def test_a_transient_provider_error_is_an_outage(error: Exception) -> None:
    assert plugin.classify(error) == "outage"


@pytest.mark.parametrize(
    "error",
    [
        AssertionError("shape differs"),
        KeyError("choices"),
        TypeError("bad"),
        TimeoutError("not a PrimaGate error"),
        BadRequestError("400", status_code=400),
        UnsupportedParamsError("tools"),
        APIError("418", status_code=418),
        _exhausted(BadRequestError("400", status_code=400)),
    ],
    ids=lambda e: type(e).__name__ + str(getattr(e, "status_code", "") or ""),
)
def test_anything_else_is_an_sdk_regression(error: Exception) -> None:
    assert plugin.classify(error) == "regression"


@pytest.mark.parametrize(
    "error",
    [
        AuthenticationError("401", status_code=401),
        PermissionDeniedError("403", status_code=403),
        NotFoundError("404: no such model", status_code=404),
        _exhausted(NotFoundError("404", status_code=404)),
        plugin.MissingKeyError("ANTHROPIC_API_KEY not set"),
        plugin.LiveConfigError("bad setting"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_a_key_or_setup_problem_is_a_regression_marked_configuration(error: Exception) -> None:
    assert plugin.classify(error) == "regression (configuration)"
    assert plugin.classify(error).startswith("regression")


def test_the_error_type_is_named_without_its_message() -> None:
    secret = ServiceUnavailableError(f"echo {FAKE_KEY} {PROMPT}")
    assert plugin.error_type(secret) == "ServiceUnavailableError"
    wrapped = _exhausted(APIError(f"{REPLY}", status_code=502))
    assert plugin.error_type(wrapped) == "AttemptsExhaustedError (APIError)"


# --- keys ---


def test_a_key_is_read_from_its_variable_and_trimmed() -> None:
    assert plugin.key_for("openai", {"OPENAI_API_KEY": f"  {FAKE_KEY}\n"}) == FAKE_KEY
    assert plugin.key_for("anthropic", {"ANTHROPIC_API_KEY": FAKE_KEY}) == FAKE_KEY


@pytest.mark.parametrize("value", [None, "", "   "])
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_a_missing_key_skips_with_the_reason(provider: str, value: str | None) -> None:
    env = {} if value is None else {f"{provider.upper()}_API_KEY": value}
    with pytest.raises(pytest.skip.Exception, match=rf"^{provider.upper()}_API_KEY not set$"):
        plugin.key_for(provider, env)


def test_a_required_provider_without_a_key_fails_and_names_the_key() -> None:
    env = {"PRIMAGATE_LIVE_REQUIRE": "openai,anthropic", "OPENAI_API_KEY": FAKE_KEY}
    assert plugin.key_for("openai", env) == FAKE_KEY
    with pytest.raises(plugin.MissingKeyError, match="ANTHROPIC_API_KEY not set"):
        plugin.key_for("anthropic", env)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, set()),
        ("", set()),
        ("openai", {"openai"}),
        (" OpenAI , anthropic ", {"openai", "anthropic"}),
        ("openai,,anthropic,", {"openai", "anthropic"}),
    ],
)
def test_the_required_list_is_read_loosely(value: str | None, expected: set[str]) -> None:
    assert plugin.required_providers(value) == expected


def test_an_unknown_required_provider_is_refused_even_with_every_key_set() -> None:
    env = {"PRIMAGATE_LIVE_REQUIRE": "opneai", "OPENAI_API_KEY": FAKE_KEY}
    with pytest.raises(plugin.LiveConfigError, match="opneai"):
        plugin.key_for("openai", env)


# --- results and the report ---


def _result(outcome: str, provider: str = "openai", **kwargs: str) -> object:
    name = f"tests/live/test_x.py::test_{outcome}[{provider}]"
    return plugin.Result(name, provider, outcome, **kwargs)


def test_recording_keeps_the_worst_outcome_of_a_test() -> None:
    results: dict[str, plugin.Result] = {}
    name, error = "tests/live/test_x.py::test_a[openai]", ServiceUnavailableError("x")
    plugin.record(results, name, "openai", "setup", "passed", None)
    assert results == {}
    plugin.record(results, name, "openai", "call", "passed", None)
    assert results[name].outcome == "passed"
    plugin.record(results, name, "openai", "teardown", "failed", error)
    assert (results[name].outcome, results[name].classification) == ("failed", "outage")
    plugin.record(results, name, "openai", "teardown", "failed", AuthenticationError("401"))
    assert results[name].classification == "regression (configuration)"
    plugin.record(results, name, "openai", "teardown", "failed", AssertionError())
    assert results[name].classification == "regression"
    plugin.record(results, name, "openai", "teardown", "failed", error)
    plugin.record(results, name, "openai", "teardown", "skipped", None)
    assert (results[name].classification, results[name].error_type) == (
        "regression",
        "AssertionError",
    )


def test_a_skip_records_only_a_known_reason() -> None:
    results: dict[str, plugin.Result] = {}
    plugin.record(results, "a", "anthropic", "setup", "skipped", _skip("ANTHROPIC_API_KEY not set"))
    plugin.record(results, "b", "openai", "setup", "skipped", _skip(f"odd {PROMPT}"))
    plugin.record(results, "c", "openai", "call", "skipped", _skip(support.INCONCLUSIVE))
    assert results["c"].skip_reason == "inconclusive: cut at the 50-token cap"
    assert results["a"].skip_reason == "ANTHROPIC_API_KEY not set"
    assert results["b"].skip_reason == "other"


def _skip(message: str) -> BaseException:
    try:
        pytest.skip(message)
    except pytest.skip.Exception as skipped:
        return skipped
    raise AssertionError("pytest.skip did not raise")


def _report(results: list[object], env: dict[str, str] | None = None) -> str:
    return plugin.render(results, env or {}, NOW)


def _counts(report: str, provider: str) -> list[str]:
    for row in report.splitlines():
        cells = [c.strip() for c in row.strip("|").split("|")]
        if cells[0] == provider:
            return cells
    raise AssertionError(f"no row for {provider}")


def test_the_report_counts_each_provider_and_lists_each_failure() -> None:
    results = [
        _result("passed"),
        _result("failed", classification="outage", error_type="ServiceUnavailableError"),
        _result("failed", classification="regression", error_type="AssertionError"),
        _result(
            "failed",
            classification="regression (configuration)",
            error_type="AuthenticationError",
        ),
        _result("skipped", "anthropic", skip_reason="ANTHROPIC_API_KEY not set"),
    ]
    report = _report(results)
    # Requests sent, passed, outage, configuration, regression, skipped.
    assert _counts(report, "openai")[2:] == ["0", "1", "1", "1", "1", "0"]
    assert _counts(report, "anthropic")[2:] == ["0", "0", "0", "0", "0", "1"]
    assert "| tests/live/test_x.py::test_failed[openai] | openai | outage |" in report
    assert "| regression (configuration) | AuthenticationError |" in report
    assert "ANTHROPIC_API_KEY not set: 1" in report


@pytest.mark.parametrize(
    ("classifications", "verdict"),
    [
        (["outage"], "FAIL: 1 provider outage and no SDK regression"),
        (["regression", "regression"], "FAIL: 2 SDK regressions."),
        (["regression", "outage", "outage"], "FAIL: 1 SDK regression and 2 provider outages."),
        (["regression (configuration)"], "FAIL: 1 configuration problem."),
        (
            ["regression", "regression (configuration)", "outage"],
            "FAIL: 1 SDK regression, 1 configuration problem and 1 provider outage.",
        ),
        (["regression (configuration)", "outage"], "FAIL: 1 configuration problem and 1 provider"),
    ],
)
def test_the_verdict_names_what_kind_of_failure_it_was(
    classifications: list[str], verdict: str
) -> None:
    results = [_result("failed", classification=c, error_type="X") for c in classifications]
    line = plugin.verdict(results)
    assert line.startswith(verdict)
    assert "\n" not in line
    assert f"Verdict: {line}" in _report(results)


def test_a_clean_run_never_claims_a_provider_that_was_skipped() -> None:
    both = [_result("passed"), _result("passed", "anthropic")]
    assert plugin.verdict(both) == "PASS: openai and anthropic verified."
    one = [_result("passed"), _result("skipped", "anthropic", skip_reason="x")]
    assert plugin.verdict(one) == "PASS: openai verified; anthropic not verified (skipped)."
    none = [_result("skipped"), _result("skipped", "anthropic")]
    assert plugin.verdict(none).startswith("NOT VERIFIED")


def test_the_report_names_missing_required_keys_but_never_a_key() -> None:
    env = {"PRIMAGATE_LIVE_REQUIRE": "openai,anthropic", "OPENAI_API_KEY": FAKE_KEY}
    report = _report([_result("passed")], env)
    assert "Missing required keys: ANTHROPIC_API_KEY" in report
    assert "Required providers: openai, anthropic" in report
    assert FAKE_KEY not in report


def test_the_report_records_the_requests_each_provider_was_sent() -> None:
    report = plugin.render([_result("passed")], {}, NOW, {"openai": 38, "anthropic": 42})
    assert _counts(report, "openai")[2] == "38" and _counts(report, "anthropic")[2] == "42"


def test_a_report_path_that_is_a_folder_gets_the_report_in_it(tmp_path: Path) -> None:
    def chosen(value: str) -> Path:
        return plugin.report_path({"PRIMAGATE_LIVE_REPORT": value}, tmp_path)

    # Not yet existing: a trailing separator, or a name without a suffix, is a folder.
    assert chosen(str(tmp_path / "new") + "/") == tmp_path / "new" / "live-report.md"
    assert chosen(str(tmp_path / "new2") + "\\") == tmp_path / "new2" / "live-report.md"
    assert chosen("reports") == tmp_path / "reports" / "live-report.md"
    assert chosen("reports/r.md") == tmp_path / "reports" / "r.md"
    assert chosen(str(tmp_path / "x.d" / "r")) == tmp_path / "x.d" / "r" / "live-report.md"


def test_the_session_creates_a_report_folder_that_does_not_exist(tmp_path: Path) -> None:
    results = {"a": plugin.Result("a", "openai", "passed")}
    path = plugin.report_path({"PRIMAGATE_LIVE_REPORT": str(tmp_path / "made")}, tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plugin.render(results.values(), {}, NOW), encoding="utf-8")
    assert (tmp_path / "made" / "live-report.md").is_file()


# --- the caps, the caller and the model overrides ---


def test_a_request_is_capped_at_fifty_tokens() -> None:
    assert support.capped({})["max_tokens"] == 50
    assert support.capped({"max_tokens": 50})["max_tokens"] == 50
    assert support.capped({"max_tokens": 1, "model": "m"}) == {"max_tokens": 1, "model": "m"}
    for bad in (51, 0, -1, "50", 50.0, True, None):
        with pytest.raises(ValueError, match="1 to 50"):
            support.capped({"max_tokens": bad})


def _gate(status: int = 200, sent: list[int] | None = None) -> primagate.PrimaGate:
    def handle(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if sent is not None:
            sent.append(body["max_tokens"])
        if status != 200:
            return httpx2.Response(status, json={"error": {"message": "x", "type": "server_error"}})
        return httpx2.Response(
            200,
            json={
                "id": "c",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ready"},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    return primagate.PrimaGate(
        api_keys={"openai": "k"}, transport=httpx2.MockTransport(handle), max_retries=0
    )


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_live_caller_sends_the_capped_max_tokens_in_both_modes(mode: str) -> None:
    sent: list[int] = []
    caller = support.LiveCaller(mode)
    caller(_gate(sent=sent), model="gpt-4o-mini", messages=support.ASK)
    caller(_gate(sent=sent), model="gpt-4o-mini", messages=support.ASK, max_tokens=7)
    assert sent == [50, 7]
    with pytest.raises(ValueError, match="1 to 50"):
        caller(_gate(sent=sent), model="gpt-4o-mini", messages=support.ASK, max_tokens=51)
    assert sent == [50, 7]


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_a_stream_error_comes_out_as_the_provider_error_not_an_assertion(mode: str) -> None:
    caller = support.LiveCaller(mode)
    with pytest.raises(ServiceUnavailableError):
        caller.stream(_gate(status=503), model="gpt-4o-mini", messages=support.ASK)


def test_a_model_override_is_honoured_and_checked() -> None:
    assert support.model_for("openai", {}) == "gpt-4o-mini"
    assert support.model_for("anthropic", {}, medium=True) == "claude-sonnet-5-5"
    env = {"PRIMAGATE_LIVE_OPENAI_MODEL": " gpt-4.1-nano ", "PRIMAGATE_LIVE_ANTHROPIC_MODEL": " "}
    assert support.model_for("openai", env) == "gpt-4.1-nano"
    assert support.model_for("anthropic", env) == "claude-haiku-5-5"
    own = {"PRIMAGATE_LIVE_OPENAI_MEDIUM_MODEL": "openai/gpt-4.1"}
    assert support.model_for("openai", own, medium=True) == "gpt-4.1"
    for provider, name in (
        ("openai", "claude-haiku-5-5"),
        ("openai", "anthropic/claude-haiku-5-5"),
        ("anthropic", "gpt-4o"),
        ("anthropic", "openai/gpt-4o"),
        ("openai", "nope/gpt-4o"),
    ):
        bad = {f"PRIMAGATE_LIVE_{provider.upper()}_MODEL": name}
        with pytest.raises(support.LiveConfigError, match=f"PRIMAGATE_LIVE_{provider.upper()}"):
            support.model_for(provider, bad)
    assert plugin.classify(support.LiveConfigError("x")) == "regression (configuration)"


def test_the_report_path_defaults_to_the_working_directory(tmp_path: Path) -> None:
    assert plugin.report_path({}, tmp_path) == tmp_path / "live-report.md"
    chosen = tmp_path / "out" / "r.md"
    assert plugin.report_path({"PRIMAGATE_LIVE_REPORT": str(chosen)}, tmp_path) == chosen
    folder = tmp_path / "dir"
    folder.mkdir()
    assert plugin.report_path({"PRIMAGATE_LIVE_REPORT": str(folder)}, tmp_path) == (
        folder / "live-report.md"
    )


# --- end to end, in a separate pytest process ---

FAKE_SUITE = f"""
    import os

    import pytest

    from primagate import AuthenticationError, ServiceUnavailableError

    pytestmark = pytest.mark.live
    PROMPT = {PROMPT!r}
    REPLY = {REPLY!r}


    def test_passes(live):
        assert live.provider in ("openai", "anthropic")


    def test_outage(live):
        raise ServiceUnavailableError(f"overloaded: {{PROMPT}} {{os.environ['OPENAI_API_KEY']}}")


    def test_regression(live):
        assert REPLY == PROMPT, os.environ["OPENAI_API_KEY"]


    def test_auth(live):
        raise AuthenticationError(f"bad key {{os.environ['OPENAI_API_KEY']}}", status_code=401)
"""


@pytest.fixture
def fake_project(tmp_path: Path) -> Path:
    """A copy of the live plugin with fake tests in place of the real ones."""
    project = tmp_path / "project"
    live = project / "live"
    live.mkdir(parents=True)
    (project / "conftest.py").write_text((TESTS / "conftest.py").read_text("utf-8"), "utf-8")
    for name in ("__init__.py", "conftest.py", "support.py"):
        (live / name).write_text((TESTS / "live" / name).read_text("utf-8"), "utf-8")
    (live / "test_fake.py").write_text(textwrap.dedent(FAKE_SUITE), "utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text("utf-8")
    pytest_section = pyproject.split("[tool.pytest.ini_options]", 1)[1].split("\n[", 1)[0]
    pytest_section = pytest_section.replace('testpaths = ["tests"]', 'testpaths = ["live"]')
    (project / "pyproject.toml").write_text("[tool.pytest.ini_options]" + pytest_section, "utf-8")
    return project


def _run(project: Path, *args: str, **env: str) -> tuple[int, str, str | None]:
    clean = {
        k: v
        for k, v in os.environ.items()
        if k not in ENV_NAMES and not k.startswith("PRIMAGATE_LIVE_")
    }
    report = project / "out" / "live-report.md"
    clean.update(env, PRIMAGATE_LIVE_REPORT=str(report))
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-rs", *args],
        cwd=project,
        env=clean,
        capture_output=True,
        text=True,
        timeout=120,
    )
    text = report.read_text("utf-8") if report.exists() else None
    return done.returncode, done.stdout + done.stderr, text


def test_the_default_run_deselects_live_tests_and_writes_no_report(fake_project: Path) -> None:
    code, output, report = _run(fake_project, OPENAI_API_KEY=FAKE_KEY)
    assert code == pytest.ExitCode.NO_TESTS_COLLECTED, output
    assert "8 deselected" in output
    assert report is None


def test_live_without_keys_skips_every_test_with_the_reason(fake_project: Path) -> None:
    code, output, report = _run(fake_project, "-m", "live")
    assert code == pytest.ExitCode.OK, output
    assert "8 skipped" in output
    assert output.count("OPENAI_API_KEY not set") and output.count("ANTHROPIC_API_KEY not set")
    assert report is not None
    assert _counts(report, "openai")[2:] == ["0", "0", "0", "0", "0", "4"]
    assert "Verdict: NOT VERIFIED" in report


def test_one_key_runs_that_provider_and_classifies_each_failure(fake_project: Path) -> None:
    code, output, report = _run(fake_project, "-m", "live", OPENAI_API_KEY=FAKE_KEY)
    assert code == pytest.ExitCode.TESTS_FAILED, output
    assert report is not None
    assert _counts(report, "openai")[2:] == ["0", "1", "1", "1", "1", "0"]
    assert _counts(report, "anthropic")[2:] == ["0", "0", "0", "0", "0", "4"]
    rows = {line.split("|")[1].strip(): line for line in report.splitlines() if "::" in line}
    assert "| outage | ServiceUnavailableError |" in rows["live/test_fake.py::test_outage[openai]"]
    assert "| regression | AssertionError |" in rows["live/test_fake.py::test_regression[openai]"]
    assert (
        "| regression (configuration) | AuthenticationError |"
        in (rows["live/test_fake.py::test_auth[openai]"])
    )
    verdict = "FAIL: 1 SDK regression, 1 configuration problem and 1 provider outage."
    assert f"Verdict: {verdict}" in report
    # The run's own output may show the failures; the report holds no key and no text.
    for secret in (FAKE_KEY, PROMPT, REPLY, "overloaded", "bad key"):
        assert secret not in report


def test_the_gate_fails_a_provider_whose_key_is_missing(fake_project: Path) -> None:
    code, output, report = _run(
        fake_project,
        "-m",
        "live",
        "-k",
        "passes",
        OPENAI_API_KEY=FAKE_KEY,
        PRIMAGATE_LIVE_REQUIRE="openai,anthropic",
    )
    assert code == pytest.ExitCode.TESTS_FAILED, output
    assert "ANTHROPIC_API_KEY not set" in output and "skipped" not in output.splitlines()[-1]
    assert report is not None
    assert "Missing required keys: ANTHROPIC_API_KEY" in report
    assert _counts(report, "anthropic")[2:] == ["0", "0", "0", "1", "0", "0"]
    assert "| regression (configuration) | MissingKeyError |" in report
    assert "Verdict: FAIL: 1 configuration problem." in report
    assert FAKE_KEY not in report


def test_the_real_live_suite_skips_cleanly_without_any_key(tmp_path: Path) -> None:
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.endswith("_API_KEY") and not k.startswith("PRIMAGATE_LIVE_")
    }
    report = tmp_path / "out" / "live-report.md"
    clean["PRIMAGATE_LIVE_REPORT"] = str(report)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/live",
            "-m",
            "live",
            "-p",
            "no:cacheprovider",
            "-rs",
        ],
        cwd=ROOT,
        env=clean,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = done.stdout + done.stderr
    assert done.returncode == pytest.ExitCode.OK, output
    assert " passed" not in output.splitlines()[-1] and " failed" not in output.splitlines()[-1]
    skips = re.findall(r"^SKIPPED \[(\d+)\] .*: (\w+_API_KEY not set)$", output, re.MULTILINE)
    assert skips and sum(int(n) for n, _ in skips) == int(
        re.search(r"(\d+) skipped", output).group(1)  # type: ignore[union-attr]
    )
    text = report.read_text("utf-8")
    assert "Verdict: NOT VERIFIED" in text and "Failed (configuration)" in text


# --- the workflow, the docs and the ignore rule ---


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _load_matrix_script() -> ModuleType:
    # The script lives outside the package; its dataclasses need it registered while it loads.
    name = "conformance_matrix"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "conformance_matrix.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_live_workflow_runs_on_a_schedule_by_hand_and_for_the_release() -> None:
    workflow = _text(".github/workflows/live.yml")
    for needed in (
        'cron: "0 6 * * 1"',
        "workflow_dispatch:",
        "workflow_call:",
        "runs-on: ubuntu-latest",
        'python-version: "3.11"',
        "uv sync --locked",
        "run: uv run pytest -m live",
        "OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}",
        "ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}",
        "default: openai,anthropic",
        "PRIMAGATE_LIVE_REQUIRE: ${{ inputs.require }}",
        "group: primagate-live-conformance",
        "cancel-in-progress: false",
        "retention-days: 30",
        "if: always()",
        "path: live-report.md",
    ):
        assert needed in workflow, needed
    assert "pull_request" not in workflow
    uses = re.findall(r"uses: (\S+)", workflow)
    assert uses and all(re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", u) for u in uses), uses


def test_pull_request_ci_never_selects_the_live_tests() -> None:
    ci = _text(".github/workflows/ci.yml")
    assert "run: uv run pytest\n" in ci and "-m live" not in ci and "API_KEY" not in ci
    assert "addopts = \"-m 'not live'\"" in _text("pyproject.toml")


def test_the_docs_cover_running_cost_keys_report_and_recorded_only_frs() -> None:
    docs = _text("docs/live-conformance.md")
    for needed in (
        "uv run pytest -m live",
        "## Cost",
        "## Where the keys live",
        "live-gate execution",
        "### Set the spend limits",
        "## Read the report",
        "### Recorded only",
        "Anthropic:** pending API credit",
        "PRIMAGATE_LIVE_REQUIRE",
        "PRIMAGATE_LIVE_REPORT",
    ):
        assert needed in docs, needed
    matrix = _load_matrix_script()
    recorded = docs.split("### Recorded only", 1)[1].split("\n## ", 1)[0].lower()
    for number, reason in matrix.RECORDED_ONLY.items():
        assert f"| fr{number} | {reason.lower()}." in recorded, number
    assert "[docs/live-conformance.md](docs/live-conformance.md)" in _text("CONTRIBUTING.md")


def test_the_report_is_never_committed() -> None:
    assert "\nlive-report.md\n" in _text(".gitignore")
