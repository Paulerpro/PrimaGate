# Choosing an Analyzer

Tier Routing asks one model, the Analyzer, which Tier each request needs. You choose that model
and you set `confidence_threshold`, which has no built-in default. This guide says what to look
for in an Analyzer and how to get evidence for both choices.

The comparison commands below are contributor tooling. They need a clone of the repository and
`uv run`; they are not part of the `pip install primagate` package.

## What the Analyzer does and what it sends where

For each routed call the Analyzer reads the Profile of your app and a bounded slice of the recent
conversation, and answers with one Tier word and a confidence, like `MEDIUM 0.82`. Both the
Profile (and, while it is being prepared, your Profile sources) and that slice go to whichever
Provider hosts the Analyzer model. Read
[What the Analyzer reads](../README.md#17-what-the-analyzer-reads) before you choose one, since it
is the privacy disclosure for this step. The worker request is never changed for the Analyzer, and
the Analyzer's cost is reported on its own line.

## The five criteria

`uv run python evals/harness.py --compare-analyzers a,b` measures each candidate on the same tasks
and reports these. No Analyzer is named best: they trade off, so weigh them for your own use.

- **Cost per routing call.** The Analyzer's cost divided by the Analyzer requests it made, with its
  share of the routed cost. Tier Routing only saves money if this stays small next to what the
  cheaper Tiers save. A request that failed has no cost, so the figure is a lower bound, and it is
  `n/a` when no Analyzer cost was seen at all (not zero).
- **Added latency.** Average and 95th percentile of the time the Analyzer adds before the worker
  starts, over every request it made, including unusable and failed ones. The 95th percentile over
  about 100 calls is indicative only.
- **Valid-output rate.** The share of calls where the Analyzer's answer was usable. A
  `malformed_output`, `provider_error` or `tier_not_enabled` degrade is invalid, and each reason
  is counted. Every invalid answer sends the call to your Interim Binding, so a low rate means you
  pay for the Analyzer and still do not get routing.
- **Tier-to-difficulty correlation.** The share of tasks sent to the Tier their difficulty calls
  for, the share sent to that Tier or a more capable one, and a rank correlation between difficulty
  and Tier (`n/a` when the Analyzer chose one Tier for everything). The difficulty labels are an
  expectation, not a measurement. Next to them read the usefulness and cost of the routed run,
  which shows a cheap Analyzer that routes badly.
- **Calibration.** Whether a higher confidence went with a higher pass rate in that candidate's own
  calls, from the same buckets and plain statement the main report uses. It is never assumed.
  Jev is an early-access model that has not been validated. The report labels any candidate whose
  name contains `jev` as "early access, unvalidated" (the label goes by name, nothing more), and
  its calibration is reported from its data like any other candidate's.

## Setting `confidence_threshold`

Each candidate gets a table over thresholds 0.5 to 0.9. The base is every call that was answered,
so a call the Analyzer failed on counts as not kept. The table shows the share of those calls at or
above the threshold (they would keep their chosen Tier), their pass rate, and the pass rate of the
rest (they would take `ambiguous_context_default`). Rows under the threshold the run used are
marked approximate, because calls just below it already ran on that default, and the suggestion
never names one. The table is computed from recorded outcomes and makes no extra call.

A one-line suggestion names the lowest threshold whose kept calls pass at least 95% while keeping
at least half of all answered calls. It is a data-based starting point, not a guarantee. It comes
from one run of one workload, and the same run measures the pass rates it is judged on, so a
different run can move it. Treat it as a starting point and rerun the comparison, ideally on a
workload that looks like yours, before you rely on it. The report flags a table with fewer than 20 kept calls as thin. When no
threshold qualifies, it says so and suggests nothing.

## Rerun the comparison

```
uv run python evals/harness.py --compare-analyzers gpt-4o-mini,gpt-4.1-mini
```

With no names, `--compare-analyzers` uses those two. It takes 2 to 6 candidates. A candidate is any
model name the SDK resolves (`gpt-4o-mini`, `openai/...`, `claude-...`). Keys come from
`OPENAI_API_KEY` and `ANTHROPIC_API_KEY`; with none for a candidate's provider, or with an unknown
provider prefix, the run stops before any call. `--base-url` sends every OpenAI call of the
comparison, including the worker calls, to an OpenAI-compatible endpoint.

The comparison is a run of its own: no baseline arm, no main evaluation. It costs about 200 calls
per candidate (every task once, plus its Analyzer call), roughly 7 to 20 US cents each at the
default models, an estimate and not a bill. `--offline` runs a scripted stand-in with no key; its
report is headed `OFFLINE STAND-IN: not evidence`. The results file gets an `analyzer_comparison`
object; it holds no key, prompt or reply text. More in [evals/README.md](../evals/README.md).

A candidate that never gave a usable answer, or whose calls all raised, is listed as failed and
left out of the comparison and every ranking, and the command exits 1. The others are still
compared:

```
- `my-model` failed (NoUsableAnswer) and is not compared.
```

## Known constraints

- The Analyzer request is fixed at `temperature` 0 and a very small `max_tokens` (16). A model
  that rejects either setting, or spends the tokens on hidden reasoning (reasoning-style models
  may), gives `provider_error` or `malformed_output` on every call. Prefer a small non-reasoning
  model, and let the valid-output rate show whether yours works.
- Only OpenAI models have been run live as an Analyzer. Anthropic models are supported by the code
  but are pending API credit, so no Anthropic Analyzer has been validated.
- Confidence is not stratified by difficulty, so a rising pass rate can only mean easy tasks get
  high confidence.
