# AGENTS.md

## Tooling

- Python tooling: **uv only**. Use `pyproject.toml`, `uv sync`, `uv run`, `uv add`. No pip, no manual venv, no requirements.txt.
- Commit `uv.lock`. Never commit `.venv/`.
