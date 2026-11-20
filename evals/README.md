# Tier Routing evaluation harness

Does Tier Routing cost at least 30% less than sending everything to one premium model, with
Usefulness (the share of tasks answered correctly) no more than 2 points lower? That is the SM-4
success target for the project. This harness measures it on a scripted workload with objective
checks, and checks whether the Analyzer's confidence means anything.

This folder is outside the installed package.

## Run it

From the project root, after `uv sync`:

```
uv run python evals/harness.py
```

It needs `OPENAI_API_KEY` in the environment, or `--api-key`. The key is never printed or written to
a report or results file. With no key it stops before making any call (exit code 2).

**Cost.** A full run is about 300 calls: 100 tasks asked through Tier Routing, the same 100 asked of
the fixed premium model, and 100 Analyzer calls made by the routed arm. At the default models it
costs roughly 10 to 30 US cents, mostly the premium arm. The tool tasks send tool definitions with
every call, which is most of the input tokens. Costs are estimates from PrimaGate's price map, not a
bill.

```
uv run python evals/harness.py --offline
```

runs against a scripted stand-in with no key and no network. It proves the harness and the
arithmetic only. Its report starts with `OFFLINE STAND-IN: not evidence` and can never say MET.

## What it does

`workload.py` holds 100 fixed tasks: 40 easy, 35 medium and 25 hard, of three types:

| Type | Tasks | Check |
|---|---|---|
| `oneshot` | 60 | an exact number, or a whole word that ends a short reply |
| `tool` | 20 | a tool is offered; passes on the right tool name and valid JSON arguments carrying the expected values |
| `multiturn` | 20 | earlier exchanges, then a question whose answer depends on them; checked like a one-shot |

No model and no person judges an answer. Usefulness is the share of tasks that pass.

`harness.py` asks each task once through the Tier Routing alias and once of a fixed premium model.
Every call is an independent streamed call with caching off, `temperature` 0 and `max_tokens` 64, so
the time to the first chunk and to completion are recorded separately. Defaults, all changeable with
options (`--help`):

| Setting | Default |
|---|---|
| SMALL, MEDIUM, POWERFUL | `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-4o` |
| Analyzer | `gpt-4o-mini` |
| Fixed premium model | `gpt-4o` |
| Confidence threshold | 0.7 |

## Reading the report

The report is printed and written to `evals/results/` with a JSON results file of the same name.
Its first lines are the verdict:

- `SM-4 MET` or `SM-4 NOT MET`, with the cost saving and the Usefulness gap beside it.
- `INCONCLUSIVE` when either arm answered under 90% of its calls, or a cost is unknown or zero, or
  the run was offline. An unknown cost is listed and never counted as free.
- The saving and the Usefulness gap are computed over the tasks both arms answered: a task one arm
  errored on is dropped from both. The report states how many tasks that is and each arm's error
  count. A call that raised records its error class and no cost, but it may still have been billed,
  so every cost is a lower bound.
- A note when any routed call fell back to the Interim Binding, since that call was not routed.

For both arms the report gives Usefulness, total cost, cost per successful interaction (total cost
over passes), first-response and completion latency (average and 95th percentile), the fallback
rate (calls whose attempt chain has more than one Attempt) and context omissions. Omissions are shown
for the routed arm only, since the fixed model has no context budget. Conversations with more than 3
earlier exchanges are a proxy for trimming, not a measurement; calls degraded with `over_budget` are
the measured signal, and the report lists the two apart. The Analyzer's cost
and added latency are shown apart from the worker's. Usefulness is split by difficulty and by type.

**By workload type.** A table gives each type's saving and Usefulness gap and marks where the target
holds. When SM-4 is not met, the report states the measured saving and the types where it holds, and
claims nothing beyond them. The owner decides about the release.

**Calibration.** Routed calls are bucketed by the Analyzer's confidence (below 0.5, 0.5 to 0.7, 0.7
to 0.9, 0.9 and above), with the number of calls and how often the chosen Tier was enough, meaning
the routed call passed. The report says plainly whether a higher confidence went with a higher pass
rate in these data. It does not assume calibration. The confidence comes from the routing decision
event's `confidence` field. Calls that degraded or took a low-confidence downgrade are left out,
since the verdict did not choose their Tier. A bucket under 5 calls is shown as "too few" and not
used for the trend, and the trend needs a strict rise, not a tie. Confidence is not stratified by
difficulty, so a rising pass rate may only mean easy tasks get high confidence.

## The results file

`harness-*.json` holds one record per call: task id, type, difficulty, arm, pass, worker and Analyzer
cost, first-response and completion latency, the Analyzer's added latency, Tier, confidence, the
reason a call went to the Interim Binding, the number of attempts and the error class. It holds no
key, no prompt and no reply text. Later evaluation stories read it instead of parsing the report.

## Limits to keep in mind

- The easy, medium and hard labels are an expectation, not a measurement. The report shows the pass
  rate per difficulty and per type for each arm so you can see where a cheaper model really fails.
- A word answer must end a short reply and not be negated; a number answer is the last number in the
  reply. Replies that add a trailing number (`150 minutes (2.5 h)`) fail, and each arm is held to
  the same rules, but a formatting habit can still show up as a difference.
- A tool task passes on the right tool and the expected values; extra arguments are allowed, a
  number must arrive as a number, and text is compared without case or surrounding spaces.
- Granularity is coarse. One task is 1 point of overall Usefulness, so the 2-point tolerance allows
  two missed tasks. Inside a 20-task type one task is 5 points, so the per-type "holds" column is
  indicative only.
- The two arms are asked in turn for each task, once, and which arm goes first alternates task by
  task. There are no repeats, so there is no variance estimate, and the 95th percentile over 100
  calls is indicative only.
- A tool call cut at `max_tokens` counts as failed but is reported apart from wrong answers. When a
  model makes parallel tool calls, only the first by index is scored.
- Calibration counts only routed calls that were answered and had a verdict. A bucket with a handful
  of calls says little; the report says so when there is too little data.
- `max_tokens` is 64 and `temperature` is 0. A model that rejects either will fail every call and
  the run exits with an error.
- Reports are saved with the time in the name and are not committed (`evals/results/` is ignored):
  one run is a measurement, not a claim about the product.
