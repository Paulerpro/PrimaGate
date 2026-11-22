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
- The Tier Routing cost saving has not been measured as evidence. Tier Routing ships as an opt-in
  capability and claims no saving.

[Releasing](releasing.md) tracks each of these as pending, not passed.

### Adoption targets

These are measures taken after release. They are not release gates, and no release waits on them.
Within 90 days of release:

- 500 GitHub stars
- 1,000 weekly installs
- 5 external contributors
