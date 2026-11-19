# Forced-failure demo

`forced_failure.py` makes PrimaGate's reliability chain fail on purpose and shows the request
surviving. Each scenario forces a failure at one layer, prints what happened and the Attempt Chain,
and checks the outcome against what should happen.

The chain always runs in this order (step 7 of the main [README](../README.md#7-handle-failures)):

1. **Provider Routing and retries.** When the model is listed in `provider_routing`, a temporary
   failure is retried on its endpoint, then the model moves on to each redundant endpoint you
   declared. A model not listed there gets one request, with no retries.
2. **Tier Failover.** For a Tier Routing call that resolved SMALL or POWERFUL, MEDIUM is tried
   once, with its own retries and endpoints, or recorded as a `skipped` row when it cannot run.
3. **The Fallback List.** Your fallback models, strictly in the order you listed them.

This folder is not part of the installed package.

## Run it

The demo is a file in the repository, not part of the package. Clone
[github.com/Paulerpro/PrimaGate](https://github.com/Paulerpro/PrimaGate), install PrimaGate at the
same version as that checkout, and run the file:

```
pip install primagate
python examples/forced_failure.py
```

The demo uses the PrimaGate you have installed, so if it is older or newer than the checkout, the
scenarios may not match what it does.

Working on PrimaGate itself, run it from the repository root with the checkout's own code:

```
uv run python examples/forced_failure.py
```

This is **recorded mode**, the default. It needs no key and no network: the provider's replies are
scripted and replayed through `httpx2.MockTransport`. It takes well under a second and gives the
same result every time. The test suite runs it, so it cannot go stale.

```
python examples/forced_failure.py --live
```

**Live mode** sends real requests to OpenAI. It needs `OPENAI_API_KEY` in the environment, or
`--api-key`, and stops before any call if neither is set. The key is never printed.

## The scenarios

| # | Scenario | What is forced | Expected outcome |
|---|---|---|---|
| 1 | Endpoint down | The primary endpoint cannot be reached. A redundant endpoint is declared. | The redundant endpoint answers. |
| 2 | Resolved Tier down | Routing resolves SMALL and its model fails. | MEDIUM answers through Tier Failover. |
| 3 | MEDIUM unavailable | SMALL fails and MEDIUM is not in `enabled_tiers`. | MEDIUM is a `skipped` row and the Fallback List entry answers. |
| 4 | Everything down | SMALL, MEDIUM and the Fallback List entry all fail. | `AttemptsExhaustedError` holding all three rows. |
| 5 | Stream fails before the first chunk | SMALL fails before sending anything. | The stream opens on MEDIUM and completes. |
| 6 | Stream fails after the first chunk | SMALL's stream breaks after one chunk. A Fallback List is declared. | `APIConnectionError` while iterating, with the chain. Nothing else is tried. |

For each scenario the demo prints its setup, the outcome (the model that answered and the first
line of its reply, or the error type), and the Attempt Chain, one line per row. A row shows the
model, then each endpoint it was sent to with its retries and the error type it ended with (`ok`
when it answered), or the reason it was skipped. Each scenario ends in `PASS` or `FAIL`; a `FAIL`
lists what was expected and what happened instead. The run ends with a line such as
`6/6 scenarios as expected (100%)`.

The exit code is 0 only when every scenario that ran ended as expected, 1 otherwise, and 2 when live
mode has no key.

Only error types are printed, never error messages, which can repeat parts of a request. No key, no
request body and no message content are printed, apart from the first line of each reply.

## How live mode forces the failures

Live mode cannot script a provider, so it forces each failure through configuration:

- **Endpoint down.** The primary endpoint is set to `https://openai-primary.invalid/v1`. A `.invalid`
  name never resolves, so each try fails at the name lookup and no request, and no key, is sent.
  The redundant endpoint is the real OpenAI API. Live mode keeps PrimaGate's default wait between
  retries, so this scenario takes about a second and a half.
- **Failing models.** Every model meant to fail is named `primagate-missing-<role>`, such as
  `primagate-missing-small`. OpenAI refuses it with `NotFoundError`, and a refused request is not
  billed. The one model meant to answer in each scenario is `gpt-4o-mini`.
- **The Analyzer.** Its answer cannot be scripted live, so it too is a missing model. Routing then
  falls back to the Interim Binding, which is SMALL, so SMALL is still the resolved Tier. That
  logs a `routing_degraded` warning for each routed scenario, which would only repeat what the
  report shows, so the demo shows only errors from the `primagate` logger while it runs.
- **Scenario 6 is skipped** and shows `SKIPPED (recorded only)`, since no setting makes a real
  provider drop a stream after its first chunk. The percentage is over the other 5.

A live run sends 14 requests to OpenAI. Four are answered by `gpt-4o-mini`, each with a
one-sentence question and at most 30 output tokens. The other 10 name missing models and are
refused. The cost is under 0.01 US cents. The Endpoint down scenario also makes three tries on the
unreachable address, which end at the name lookup.

The models answering in live mode are real, so their replies differ from run to run. The outcome
checks look only at which model answered, the Attempt Chain and the models sent, never at the
reply text.
