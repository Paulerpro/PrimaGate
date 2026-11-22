# Release notes

## 0.1.0

The first release. Not yet published: see the known blockers below.

### What ships

- One `completion` and `acompletion` call shape over OpenAI and Anthropic, with a response and
  metadata shape that is the same for both. PrimaGate talks to providers directly and needs no
  provider SDK.
- Reliability routing: retries, redundant endpoints (opt-in Provider Routing), Tier Failover and a
  Fallback List, with the Attempt Chain recorded on every outcome.
- Opt-in Tier Routing: an Analyzer you choose picks SMALL, MEDIUM or POWERFUL for each call,
  governed by a Policy and Tier Bindings you declare. See the
  [Analyzer guide](analyzer-guide.md).
- Cost Estimates, token counting, an in-memory response cache, events, payload retention controls,
  cancellation and a file-backed Profile store.
- Semantic versioning over the public surface. See [versioning](versioning.md).

### Known blockers

- The package is not on PyPI yet. The clean-install check, `pip install primagate` plus a first
  call on every supported OS and Python version, cannot run until it is.
- GitHub Actions billing is locked, so the CI matrix has never run green.
- The live suite passed on both OpenAI and Anthropic in the owner's local run on 2026-11-20. It has
  not run in GitHub Actions yet. See [live-conformance](live-conformance.md).
- Claude version 5 models reason before answering, and the reasoning counts against `max_tokens`.
  A small limit can return no text, with `finish_reason` `"length"`. The README says so; a setting
  to control it is not in this version.
- The Tier Routing target (SM-4) was measured on 2026-11-22 and **not met**. On the harness's 100
  scripted tasks, with `gpt-4o-mini`, `gpt-4.1-mini` and `gpt-4o` as the Tiers and `gpt-4o` as the
  fixed model, routing cut total cost by about 78% (three runs: 78.4%, 78.2%, 78.2%) but answered
  4.0 points fewer tasks correctly (93% against 97%), where the target allows 2. No workload type
  met both conditions. The Analyzer added about 0.85 seconds to each call, and its confidence did
  not track how often the chosen Tier was enough, so it cannot be treated as calibrated. This is
  one workload and one model set, so no saving is claimed beyond it. The result rests on the
  harness's objective checks, checked against a human-rated sample of 50 responses. The person
  rating found the fixed model useful on 22 of 22 and Tier Routing on 27 of 28, and agreed with the
  automatic check on 49 of 50. The one disagreement was a valid answer the check had refused
  (`es` for Spanish where it expected the word), so the automatic check slightly understates
  Tier Routing. That does not change the verdict. Tier Routing ships as an opt-in capability. See
  [analyzer-guide](analyzer-guide.md).

[Releasing](releasing.md) tracks each of these as pending, not passed.

### Next version

These are focus areas for a later release, taken from the measurements above. They are not
commitments.

- **Routing accuracy on hard tasks.** The Analyzer sent most hard tasks to the cheapest Tier. Closing
  the 4-point Usefulness gap is the main goal for Tier Routing.
- **Analyzer latency.** The Analyzer adds about 0.85 seconds to each routed call.
- **A calibrated confidence.** The Analyzer's confidence did not track outcomes, so
  `confidence_threshold` cannot yet be set from it.
- **Analyzer answer format.** Some models rarely follow the strict `TIER 0.82` format. Making valid
  answers more reliable, and saying why a routed call fell back, are both planned.
- **Profile value.** A real Profile added cost and no measured gain in the first experiment.
- **Control of Claude reasoning.** A setting to limit or turn off reasoning on Claude version 5
  models, which currently count it against `max_tokens`.
- **The evaluation's tool checks.** `tool-medium-12` expects the language name where an ISO code is
  equally correct. Loosening it is a change to the evidence, so it waits for a new, labelled run.
- **Prices for newer Claude models**, so their cost shows instead of unknown.

### Adoption targets

These are measures taken after release. They are not release gates, and no release waits on them.
Within 90 days of release:

- 500 GitHub stars
- 1,000 weekly installs
- 5 external contributors
