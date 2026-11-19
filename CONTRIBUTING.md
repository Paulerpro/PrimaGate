# Contributing to PrimaGate

Thank you for your interest in PrimaGate. This page explains how changes get in.

## Where to start

Read the [code of conduct](CODE_OF_CONDUCT.md) first. It applies to issues, pull requests and every
other project space.

## The issue tracker is the channel

Use the [issue tracker](https://github.com/Paulerpro/PrimaGate/issues) for bug reports, questions
and proposals. For anything larger than a small fix, open an issue before you write the code, so we
can agree on the change first.

A good bug report gives the PrimaGate version, the Python version, your operating system, the
smallest code that shows the problem, and what you expected to happen. Never paste API keys, and
leave out message content you would not want public.

## Security issues

Report security issues privately to nrpaulsilas@gmail.com, not in an issue. Examples are an API key
that leaks into an event, a log or an error, or a broken privacy guarantee such as message content
kept without payload retention.

## How changes are merged

The owner reviews and merges every change. A pull request is merged once the owner has reviewed it
and every check passes.

No DCO and no CLA: you do not need to sign off your commits or sign a contributor agreement in v1.
By opening a pull request you agree that your contribution is released under the project's
[MIT License](LICENSE).

## Pull requests

- Branch from `main` and open the pull request against `main`.
- Link the issue where the change was agreed in the pull request description.
- CI runs on Linux, macOS and Windows with Python 3.11 to 3.14. Do not assume one operating
  system: build paths with `pathlib`, and do not depend on line endings.

## Local checks

The repository uses [uv](https://docs.astral.sh/uv/). Python 3.11 or newer is required. CI installs
with `uv sync --locked` and runs three checks. Run them before you open a pull request:

```bash
uv sync --locked
uv run ruff check .
uv run mypy primagate
uv run pytest
```

Also run the format check locally. CI does not run it yet, but keep the code formatted:

```bash
uv run ruff format --check .
```

To run one test while you work, use `uv run pytest tests/<file> -k <name>`. The tests need no
network and no API keys.

If you add or change a dependency, run `uv lock` and commit `uv.lock`, because CI fails on a stale
lockfile.

## Tests come first

Every behaviour change comes with tests. Write the test first and see it fail before you change
the code.

PrimaGate has a sync API (`completion`) and an async one (`acompletion`). Where a change applies to
both, test it in both sync and async mode, so the two never drift apart.

## Versioning

Read [docs/versioning.md](docs/versioning.md) before you change anything a user can import from
`primagate`. A breaking change to the public surface needs a new major version.
