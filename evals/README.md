# Routing spike

A first, rough measurement of one question: does Tier Routing cost at least 30% less than sending
everything to one premium model, with Usefulness (the share of tasks answered correctly) no more
than 2 points lower? That is the SM-4 success target for the project.

This folder is outside the installed package. Epic 8 extends it into the full evaluation harness
with a larger sample and human rating.

## Run it

From the project root, after `uv sync`:

```
uv run python evals/spike.py
```

It needs `OPENAI_API_KEY` in the environment, or `--api-key`. The key is never printed or written
to the report. With no key it stops before making any call.

At the default models it makes 60 short calls (30 tasks, two ways) plus 30 short Analyzer calls,
which costs roughly 10 to 20 US cents.

```
uv run python evals/spike.py --offline
```

runs against a scripted stand-in with no key and no network. It proves the harness and the
arithmetic only. Its report starts with `OFFLINE STAND-IN: not evidence` and can never say MET.

## What it does

`workload.py` holds 30 fixed tasks: 14 easy, 10 medium and 6 hard. Each has a rule that checks the
reply (an exact number or a whole word), so no model and no person judges an answer. Usefulness is
the share of tasks that pass.

`spike.py` sends each task once through the Tier Routing alias and once to a fixed premium model.
Each is an independent call with caching off. Defaults, all changeable with options (`--help`):

| Setting | Default |
|---|---|
| SMALL, MEDIUM, POWERFUL | `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-4o` |
| Analyzer | `gpt-4o-mini` |
| Fixed premium model | `gpt-4o` |
| Confidence threshold | 0.7 |

## Reading the report

The report is printed and written to `evals/results/`. Its first lines are the verdict:

- `SM-4 MET` or `SM-4 NOT MET`, with the cost saving and the Usefulness gap beside it.
- `INCONCLUSIVE` when fewer than 30 calls were answered in an arm, or a cost is unknown or zero.
  The workload has exactly 30 tasks, so one failed call makes a run inconclusive: rerun it.
- A note when any routed call fell back to the Interim Binding, since that call was not routed.
- The Analyzer's cost and added latency are shown apart from the worker cost, with the share of
  calls on each Tier and the number that fell back to the Interim Binding.

Costs are estimates from PrimaGate's price map, not a bill. A pass rate over 30 tasks is coarse:
one task is 3.3 points, so the 2-point tolerance means the routed arm may not miss a single task
the fixed model got right. Read a NOT MET with that in mind, and rerun before drawing a conclusion.

## Limits to keep in mind

- The easy, medium and hard labels are an expectation, not a measurement. The report shows the pass
  rate per level for each arm so you can see where a cheaper model really fails.
- A word answer must end a short reply and not be negated; a number answer is the last number in the
  reply. Replies that add a trailing number (`150 minutes (2.5 h)`) fail, and each arm is held to
  the same rules, but a formatting habit can still show up as a difference.
- The two arms are asked in turn for each task, once. There are no repeats, so there is no
  variance estimate, and the 95th percentile over 30 calls is indicative only.
- `max_tokens` is 64 and `temperature` is 0. A model that rejects either will fail every call and
  the run exits with an error.
- Reports are saved with the time in the name and are not committed (`evals/results/` is ignored):
  one run is a measurement, not a claim about the product.
