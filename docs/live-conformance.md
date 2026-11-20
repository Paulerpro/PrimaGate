# Live conformance

The Conformance Suite in `tests/conformance/` runs against scripted replies, so it needs no
network and no keys. That cannot show that a real provider still answers the way the recordings
say. The live suite in `tests/live/` closes that gap. It makes real calls to OpenAI and Anthropic
and checks the same contracts the recorded tests check, in both sync and async mode.

Live tests carry the `live` marker. The default run deselects them, so `uv run pytest` and
pull-request CI never spend money or need a key.

## Run it locally

Set the key of each provider you want to test in your shell, then run the live suite:

```bash
export OPENAI_API_KEY=...        # PowerShell: $env:OPENAI_API_KEY = "..."
export ANTHROPIC_API_KEY=...
uv run pytest -m live
```

Keys are read only from these two variables. A provider whose key is not set is skipped with the
reason `OPENAI_API_KEY not set` or `ANTHROPIC_API_KEY not set`. With only an OpenAI key, the OpenAI
tests run and the Anthropic tests are skipped. The run writes `live-report.md` to the working
directory. That file is gitignored.

If you pass your own `-m` expression to a normal run, keep `not live` in it, or the live tests are
selected too.

| Variable | What it does |
|---|---|
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` | The provider keys. Never printed, in the report or anywhere else. |
| `PRIMAGATE_LIVE_REQUIRE` | Comma-separated providers that must run, for example `openai,anthropic`. A required provider with no key fails its tests instead of skipping them. An unknown name fails the run. |
| `PRIMAGATE_LIVE_REPORT` | Where to write the report. A path ending in a separator, an existing folder, or a new name without a suffix is a folder, which is created and gets `live-report.md`. Defaults to `live-report.md` in the working directory. |
| `PRIMAGATE_LIVE_OPENAI_MODEL`, `PRIMAGATE_LIVE_ANTHROPIC_MODEL` | Replace the default model when a provider retires it. It must be that provider's model: a prefix or a name for another provider fails the run as a configuration problem before any call. |
| `PRIMAGATE_LIVE_OPENAI_MEDIUM_MODEL`, `PRIMAGATE_LIVE_ANTHROPIC_MEDIUM_MODEL` | Replace the MEDIUM Tier model of the Tier Routing tests. |

## What runs live

| FRs | What is checked against a real reply |
|---|---|
| FR1, FR13 | A plain call has the same unified shape as a recorded reply, made in one direct dispatch with no retry. |
| FR2 | A tool round trip, structured output against a JSON schema, a text stream, and a streamed tool call. |
| FR4, FR5, FR6 | A feature the model lacks is refused before anything is sent, and dropped and recorded when named. On Anthropic this uses a real gap (images) and the hint parameters `n` and `seed`. |
| FR31, FR32 | A repeat is answered from the cache; a call with `cache=False` and a stream are sent. |
| FR33, FR34 | Token counts come from the provider's usage, the counter is not asked, and the cost follows the price. |
| FR35 | A ceiling set below what a real call costs reports `spending_ceiling_exceeded` after the call succeeds. |
| FR37, FR38 | Events reach the handler and the log with one request id, hold no conversation by default, and hold both sides when payload retention is on. |
| FR20, FR21, FR22, FR36 | Tier Routing with a real Analyzer: its answer is read strictly under the Policy and the decision events are logged before the dispatch; when the Analyzer cannot answer, the call degrades to the Interim Binding. |

Live tests assert contracts (shape, presence, types, recorded drops) rather than exact text,
because real replies vary. A tool or JSON reply that the 50-token cap cuts off (`finish_reason`
`length`) proves nothing either way, so that test is skipped as
`inconclusive: cut at the 50-token cap` rather than failed.

### Recorded only

Every FR that FR41 requires has recorded tests. A required FR is recorded only when its
behaviour cannot be staged against a real provider: it needs the provider to fail, time out or be
cancelled at a chosen moment, which a real service does only by chance, or it is settled at setup
before any request is sent, so a live call would add nothing. This is not a failure.

| FR | Why it is recorded only |
|---|---|
| FR7 | Fallbacks run only after the primary fails, and a real provider cannot fail on cue. |
| FR8 | Running without a Fallback List is setup behaviour with nothing provider-specific. |
| FR9 | The retries and redundant endpoints of each entry need forced 5xx and timeout failures. |
| FR10 | A fallback is skipped only after the primary fails, which cannot be forced live. |
| FR11 | The aggregate error needs every Attempt to fail, which cannot be forced live. |
| FR12 | A retry needs a forced 5xx, 429 or timeout, which a real provider gives only by chance. |
| FR14 | Redundant endpoints are validated at setup, before any request is sent. |
| FR16 | Tier Bindings are validated at setup, before any request is sent. |
| FR25 | Eligibility is decided from declared capabilities before any request is sent. |
| FR27 | Cancelling at an exact point of a real call is not reproducible. |
| FR30 | Tier Failover needs the resolved Tier's model to fail with a retry-able error on cue. |

The [conformance matrix](conformance-matrix.md) has a Live column that lists the live tests for
each FR, or the reason above. The coverage audit fails if a required FR has neither, or if an FR
that gained a live test is still listed as recorded only.

## Cost

Every live request uses the cheapest current model of its provider, asks for at most 50 output
tokens, and sends a prompt of one short sentence. The suite refuses a call that asks for more than
50 tokens, and the Analyzer asks for 16.

| Provider | Model | Requests per full run |
|---|---|---|
| OpenAI | `gpt-4o-mini`, and `gpt-4.1-mini` only if the Analyzer picks MEDIUM | 38 |
| Anthropic | `claude-haiku-5-5`, and `claude-sonnet-5-5` only if the Analyzer picks MEDIUM | 42 |

A full run with both keys makes 80 requests; the report's "Requests sent" column counts them
(worker dispatches and Analyzer questions), so these numbers can be checked after a real run. A
test skipped as inconclusive sends fewer. Four of the requests ask an Analyzer model that does not
exist, which the provider refuses without charge. Most prompts are well under 100 input tokens; a
call with a tool definition, or an Analyzer question, is a few hundred.

Expected spend per full run, at the prices on 2026-11-19:

- OpenAI: under US$0.01.
- Anthropic: under about US$0.05. `claude-haiku-5-5` is deliberately not in the bundled price
  map, because Anthropic prices it in two tiers by prompt size, which a flat price cannot express.
  Its responses therefore report an unknown cost and a `cost_unknown` event (the live tests expect
  that, and price it with `price_overrides` where they check a cost). This estimate assumes at
  most US$1 per million input tokens and US$5 per million output tokens.

A full run costs well under US$0.10, and the weekly schedule under US$0.50 a month. The spend
limits below cap it whatever happens.

## Where the keys live

This decision closes the architecture's deferred item "live-gate execution".

- **GitHub Actions secrets.** The keys are the repository secrets `OPENAI_API_KEY` and
  `ANTHROPIC_API_KEY`. The scheduled run, and the release gate once Story 8.6 adds it, use them
  unattended.
- **Spend-limited.** Each key belongs to a project or workspace with a hard spend limit set on the
  provider's dashboard (see below), so a leaked key or a runaway run has a fixed ceiling.
- **Locally too.** The owner can run `uv run pytest -m live` locally with the same kind of key.
  Until GitHub Actions is unlocked for the repository, that local run is the only live run.

### Set the spend limits

Use keys made for this purpose only, never a personal or production key.

- **OpenAI.** In the OpenAI platform dashboard, create a project for PrimaGate live runs, set a
  monthly budget on that project, and create the API key inside the project. A few dollars a month
  is plenty.
- **Anthropic.** In the Anthropic Console, create a workspace for PrimaGate live runs, set its
  spend limit, and create the API key in that workspace.
- **GitHub.** Add both keys under the repository's Settings, Secrets and variables, Actions, as
  `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

## The workflow

`.github/workflows/live.yml` runs the live suite on ubuntu with Python 3.11:

- weekly, on Monday at 06:00 UTC;
- by hand, from the Actions tab (`workflow_dispatch`);
- from another workflow (`workflow_call`). `.github/workflows/release.yml` calls it as the gate of
  a release: it runs first and publishing happens only if it passes (see
  [releasing.md](releasing.md)). That workflow has not run yet. The caller passes the keys with
  `secrets: inherit`.

Scheduled and manual runs require no provider, so a missing secret shows as skipped in the report
and the run stays green on the providers that have a key. A `workflow_call` requires every provider
by default (its `require` input, `openai,anthropic`), so a missing secret fails the gate instead
of skipping. A release can never pass merely because a secret was missing.

Only one live run goes at a time (a `concurrency` group), so two runs never spend in parallel. The
report is uploaded as the `live-report` artifact, kept for 30 days, whether the run passed or
not. It is not part of pull-request CI.

## Read the report

`live-report.md` starts with a one-line verdict:

- `PASS: openai and anthropic verified.` Every live test that ran passed. A provider that was only
  skipped is listed as not verified, never as passed.
- `FAIL: 1 provider outage and no SDK regression; rerun once the provider recovers.`
- `FAIL: 2 SDK regressions.` or `FAIL: 1 SDK regression and 2 provider outages.`
- `FAIL: 1 configuration problem.` A key or model needs fixing; nothing points at the SDK.
- `NOT VERIFIED: no live test ran; every one was skipped.` No key was set.

Then come per-provider counts (models, requests sent, passed, failed as an outage, as a
configuration problem or as a regression, and skipped), each failure with its test, classification
and error type, and the skip reasons.

Each failed test is classified from the type of the error it raised, never from its message:

| Classification | Raised error | What to do |
|---|---|---|
| outage | `Timeout`, `RateLimitError`, `ServiceUnavailableError`, `APIConnectionError`, `APIError` with status 500 or above, `APIError` with no status (a reply or stream that broke off), or an Analyzer the provider could not answer that routing absorbed | The provider could not answer. Check its status page and rerun. |
| regression (configuration) | `AuthenticationError`, `PermissionDeniedError`, `NotFoundError` (the model does not exist for this key), a required key that is not set, a bad `PRIMAGATE_LIVE_REQUIRE`, or a model override for another provider | Fix the key, its limits or the secret. For `NotFoundError`, the model was retired or is not enabled for the key: pick another with the `PRIMAGATE_LIVE_*_MODEL` overrides. |
| regression | Anything else: a failed assertion, `BadRequestError`, a shape that no longer matches | The SDK no longer matches the provider. Investigate before releasing. |

An `AttemptsExhaustedError` is classified by the error that ended its chain. When a test fails
twice (in the call and in teardown), the report keeps the worse classification: regression over
configuration over outage. Every kind of failure fails the run; the report only says which kind it
was.

Two failures look like something else:

- A `RateLimitError` (429) is counted as an outage, but one that persists across reruns may be an
  exhausted spend limit or budget. Check the provider's dashboard before waiting it out.
- Anthropic answers a 400 about credit balance when the account has no credit. It is reported as a
  regression with `BadRequestError`, but it is configuration: check the Anthropic Console's billing
  before investigating the SDK.

The report holds test ids, providers, models, counts and error type names. It holds no key, no
prompt, no reply and no error message, because a provider's error text can quote the request, so it
can be shared as it is. The pytest output of a failed run can show the failing assertion, so read
that before you share it.

## Status

- **OpenAI:** not yet run with this suite. It is recorded as passed once the owner runs
  `uv run pytest -m live` with a key and the report says so. Until GitHub Actions is unlocked, that
  local run is the only live run.
- **Anthropic:** pending API credit. The Anthropic adapter has never run against the real service,
  and the live suite has never run on Anthropic. It is not claimed as passed. Until an Anthropic
  key is added, scheduled runs show the Anthropic tests as skipped, and the release gate, which
  requires both providers, fails. That is intended.

The plugin and the report are tested offline in `tests/test_live_report.py`, and the live tests
skip cleanly without keys.
