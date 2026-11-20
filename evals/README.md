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

The file is results format version 2. Version 2 adds, to every call, `profile_not_ready` (true when
the call was served by the Interim Binding while the Profile was `preparing` or had
`preparation_failed`), and a top-level `experiments` object, empty unless an experiment ran. Each
experiment in it has a `status` (`ran` or `not_run`), a `reason` (set when it did not run) and its
`calls`, in the same record format as the main `calls` with the arm named `profile` or
`no_history`. Nothing else changed from version 1.

## Isolation experiments

The main run measures Tier Routing as a whole. These two experiments change one thing and compare
the result with the routed arm of the same run. That arm already uses an empty Profile and up to 3
earlier exchanges, so it is the control and is run once, not again for each experiment. Each
experiment's own arm is run once, over the same tasks in workload order with the same settings. No
model judges anything, and the sections report which way a number moved and by how much, with no
claim about why.

```
uv run python evals/harness.py --experiment profile
uv run python evals/harness.py --experiment history
uv run python evals/harness.py --experiment all
```

Each adds a section to the report and fills `experiments` in the results file. The control is
always a fresh routed arm from the same invocation; a saved results file is not reused, because it
holds no replies and is not guaranteed to match the current settings. `--offline` runs both against
the scripted stand-in, which behaves differently with a Profile and without history so the sections
have numbers to show. Offline numbers are never evidence and the report says so.

**Profile experiment.** The real-Profile arm declares `profile_sources`: the harness's own
instructions and the description of each tool its workload uses. A first routed call, left out of
the results, starts the preparation, and the arm then waits for the `profile_ready` event, up to
`--profile-timeout` seconds (60 by default). The wait is not timed or counted as routed calls.
If the Profile does not become ready (a final `preparation_failed` event, or the timeout) the
section reads `NOT RUN: Profile not ready` with the reason, nothing else is claimed, and the run
exits 1. Any call served while the Profile was not ready is flagged (`profile_not_ready`) and its
task is left out of the comparison in both arms. The section gives, for both arms, total cost,
Usefulness, Analyzer overhead (cost and added latency), the Tier chosen per difficulty and the
share of tasks sent to a Tier at least as capable as the task's difficulty needed (an easy task
needs SMALL, a medium task MEDIUM, a hard task POWERFUL), then the direction and size of each move.

**History experiment.** The no-history arm sets the Policy setting `context_exchanges` to 0, so the
Analyzer reads only the latest request. Only the 20 multi-turn tasks are run, since only their
routing can depend on history. The worker still receives the whole conversation, so what changes is
the Analyzer's view. The section gives, for both arms, Usefulness, total cost and Tier agreement
(the share sent to exactly the Tier the task's difficulty calls for). With 20 tasks, one task is 5
points.

**Extra cost.** Each experiment task is one worker call and one Analyzer call, so the Profile
experiment adds about 200 calls (100 tasks) and the history experiment about 40 (the 20 multi-turn
tasks), on top of the 300 of the main run, plus the Profile preparation and one warm-up call. With
a real key the harness prints this count and a rough cost to stderr before it starts, scaled from
the main run's estimate. The Profile preparation and warm-up are not priced into the section's
totals, so the real-Profile arm's cost is a lower bound; the warm-up cost is shown.

**Order and failure.** The main report and results file are printed and saved first. A run in
which either main arm answered nothing stops there with exit 1, before any experiment is paid for.
Experiments then run, and their sections are appended to the saved report and results. If an
experiment fails or an arm answers nothing, the run exits 1, as it does for `NOT RUN: Profile not
ready`; the main results are already kept. `--profile-timeout` must be a finite number above 0.

## Human-rated sample

Automatic checks say whether an answer matches a rule. A person rates a sample too, to see how far
the rules and a human agree. No judge model is ever used.

```
uv run python evals/harness.py --export-rating
```

makes the main run and then writes two files to `evals/results/`:

- `rating-<time>.csv`: 50 responses, drawn from both arms of that run, spread over workload type
  and difficulty in proportion to their size, with a fixed seed so the same results give the same
  sample. Each row has a neutral item id, the conversation, the response (a reply, or a tool call
  with its arguments) and an empty `useful` column. It does not show which arm made a response. If
  fewer than 50 answered calls exist, it exports those and says so.
- `rating-<time>.key.json`: the key from item ids to arm and task id, and the results file the run
  wrote. Keep it away from the person rating.

Open the CSV, and for each row type `yes` if the response is useful for that conversation or `no`
if it is not. Save it in place and score it:

```
uv run python evals/harness.py --score-rating evals/results/rating-<time>.csv
```

This makes no call and needs no key. Any value other than `yes` or `no` (case and spaces do not
matter), including a blank, is an error that names the row, and nothing is scored. A good file gives
the number rated and the human-rated Usefulness for each arm, and the agreement rate between the
human rating and the automatic check, with the task ids of every disagreement.

**Blind rating has limits.** The rater must not open the `.key.json`, and should not be shown the
results files or reports. The blinding hides the arm, not everything: a response's length or style
can still hint at which arm made it, and the same task can appear twice, once per arm, so a rater
may recognise it. Rate each row on its own merits.

**Rating part of the sample.** Delete the rows you do not want to rate; do not leave `useful`
blank, which is an error. Scoring then reports "N of 50" rated and lists the items with no row,
so a partial file is never presented as complete. Spreadsheet formulas are defused on export: a
cell that begins with `=`, `+`, `-`, `@`, a tab or a return gets a leading apostrophe, which is
only in the file and is ignored when scoring. Save as comma separated UTF-8 CSV. `--score-rating`
cannot be combined with `--offline`, `--api-key`, `--experiment` or `--export-rating`.

The CSV holds prompts and replies. It stays in `evals/results/`, which git ignores; do not move it
elsewhere or share it unless you mean to. Reports and results files still hold no prompt or reply.
The CSV also holds no key.

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
