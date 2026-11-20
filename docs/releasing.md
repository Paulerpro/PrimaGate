# Releasing

A release is a `v*` tag. `.github/workflows/release.yml` then runs the live gate, builds once and
publishes to PyPI by trusted publishing. No PyPI token or password is stored anywhere.

**Status: the release workflow has never run.** It is unverified until the owner pushes the first
tag.

## Before the first tag

Do these once.

1. **Register a pending publisher on PyPI.** The project does not exist on PyPI yet, so the first
   upload needs a pending publisher. On PyPI, open Your projects, then Publishing, and add a GitHub
   publisher with:
   - PyPI project name: `primagate`
   - Owner: `Paulerpro`
   - Repository: `PrimaGate`
   - Workflow: `release.yml`
   - Environment: `pypi`
2. **Create the `pypi` environment on GitHub.** In the repository settings, add an environment named
   `pypi`. Consider requiring the owner's approval on it, so a tag alone cannot publish.
3. **Clear what stops the gate running today.**
   - GitHub Actions billing is locked, so no workflow, including this one, can run.
   - The gate requires both providers. Set spend-limited `OPENAI_API_KEY` and
     `ANTHROPIC_API_KEY` repository secrets, and read [live-conformance.md](live-conformance.md)
     first. The live suite itself passed on both providers in a local run on 2026-11-20.

## The tag rule

A tag is `v` plus `MAJOR.MINOR.PATCH`, digits only, for example `v0.1.0`. There is no pre-release
suffix in v1. The tag must equal `v` plus `primagate.__version__`. The build job fails on a tag that
is not semantic versioning or does not match, before anything is uploaded. Which number to bump
follows [versioning.md](versioning.md).

## Steps

1. Choose the version by [versioning.md](versioning.md) and set `__version__` in
   `primagate/__init__.py`.
2. Merge that change to `main` and check CI is green on the full matrix.
3. Optionally check locally: `uv build` then
   `uv run python scripts/check_dist.py --tag v<version>`.
4. Tag the merge commit `v<version>` and push the tag.
5. Watch the run. `gate` runs the live suite, `build` runs `uv build` and the dist check and uploads
   `dist/`, and `publish` uploads exactly those files. If any job fails, nothing is published.
6. After it publishes, check the version on PyPI and run a clean `pip install primagate` and a first
   call.

## What the dist check enforces

`scripts/check_dist.py` rejects the build when:

- a runtime dependency in the wheel or the sdist `PKG-INFO` lacks a lower bound (`>=`) or an upper
  major cap (`<N`), extras included;
- the sdist contains `uv.lock`. The lockfile stays in the repository for development and CI.

PyPI refuses to upload a version twice. If a release is wrong, fix it and release the next patch
version.

## Release readiness

Check each line before the first tag. Nothing here is marked done unless the evidence exists.

| Check | State | How it is settled |
|-------|-------|-------------------|
| Docs audit | Automated, passes | `uv run python scripts/check_docs.py`; `uv run pytest` runs it too |
| Clean install from PyPI | Pending first publish | Run the `Install check` workflow with the published version |
| CI matrix green | Pending: Actions billing | A green `CI` run on `main` across every OS and Python version |
| No critical bug open | Owner checks | The [issue tracker](https://github.com/Paulerpro/PrimaGate/issues) |
| Live gate | Passed locally on 2026-11-20 (OpenAI and Anthropic); not yet run in GitHub Actions | [live-conformance.md](live-conformance.md) |

The `Install check` workflow (`.github/workflows/install-check.yml`) is started by hand from the
Actions tab with the `version` to test. It creates a fresh virtual environment on Ubuntu, macOS and
Windows for Python 3.11 to 3.14, runs `pip install primagate==<version>`, then runs
`scripts/smoke_install.py`, which makes a first call with a scripted reply and prints `ok`. It cannot
pass until a version is on PyPI.

See the [release notes](release-notes.md) for what the first release ships and its known blockers.
