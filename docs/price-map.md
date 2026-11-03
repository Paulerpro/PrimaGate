# Bundled price map

PrimaGate's cost Estimates come from `primagate/core/prices.py`. Prices are USD per million
tokens, standard (not batch, not cached) rates. The map was last checked on **2026-11-03**.

## Sources

- OpenAI: https://developers.openai.com/api/docs/pricing
- Anthropic: https://platform.claude.com/docs/en/about-claude/pricing

## What is in it

| Model | Input | Output |
| --- | --- | --- |
| `gpt-4o` | 2.50 | 10.00 |
| `gpt-4o-mini` | 0.15 | 0.60 |
| `gpt-4.1` | 2.00 | 8.00 |
| `gpt-4.1-mini` | 0.40 | 1.60 |
| `claude-fable-5-1` | 10.00 | 50.00 |
| `claude-opus-5-5` | 4.00 | 20.00 |
| `claude-sonnet-5-5` | 2.00 | 10.00 |

A dated snapshot such as `gpt-4o-2024-08-06` is priced as its base model. Nothing else is matched
by prefix.

## What is left out, and why

A model is left out unless its standard price could be read plainly from the pages above, so it
reports an unknown cost until a caller passes `price_overrides` for it. Left out on purpose:

- `claude-haiku-5-5`: priced in two tiers by prompt size, which a flat price cannot express yet.
- `claude-sonnet-5`: its price carries a footnote that was not checked.
- Every other OpenAI and Anthropic model not in the table above.

## Updating it

1. Open both pages above and read the standard input and output price of each model in the table.
2. Edit `BUNDLED` and `PRICES_AS_OF` in `primagate/core/prices.py` and the table here, including
   the date. A test fails if a bundled model is missing from this file or the date differs.
3. Add a model only if its standard price is plain on the page; otherwise leave it out.
4. Run `uv run pytest`.

Prices change without notice. The map is a convenience, not a billing record: pass
`price_overrides` for exact figures.
