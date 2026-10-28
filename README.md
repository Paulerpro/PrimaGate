# PrimaGate

One call shape for LLM providers, embedded in your process.

## Quickstart

### 1. Install

```bash
pip install primagate
```

Python 3.11 or newer. PrimaGate talks to providers directly,
so you do not install the OpenAI or Anthropic SDK.

### 2. Set your key

```bash
export OPENAI_API_KEY="sk-..."
```

On Windows PowerShell: `$env:OPENAI_API_KEY = "sk-..."`. To skip the environment variable,
pass the key when you create an instance: `PrimaGate(api_keys={"openai": "sk-..."})`.
For Anthropic models, set `ANTHROPIC_API_KEY` or pass `api_keys={"anthropic": "..."}`; each key is
only ever sent to its own provider. A bare model name starting `claude-` goes to Anthropic and any
other bare name to OpenAI; write `openai/<name>` to force OpenAI for a model with a `claude-` name.

### 3. Make a call

```python
from primagate import completion

response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.choices[0].message.content)
```

`gpt-4o` and `openai/gpt-4o` both route to OpenAI.

### 4. Make the same call asynchronously

```python
import asyncio

from primagate import acompletion


async def main() -> None:
    response = await acompletion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say hello in five words."}],
    )
    print(response.choices[0].message.content)


asyncio.run(main())
```

`acompletion` takes the same arguments as `completion` and returns the same response.
Both use one shared default instance. Create your own with `PrimaGate(...)` when you need
separate keys or settings, and call `await gate.aclose()` (or `gate.close()` from sync code)
when you are done with it.

### 5. Switch providers

Change only the `model` string to move a call between providers. The response has the same shape
either way.

```python
from primagate import completion

messages = [{"role": "user", "content": "Say hello in five words."}]

for model in ["gpt-4o", "claude-sonnet-5-5"]:
    response = completion(model=model, messages=messages)
    print(model, "->", response.choices[0].message.content)
```

Set `ANTHROPIC_API_KEY` for the Claude model, as described in step 2.

### 6. Handle an unsupported feature

When a model cannot do something your request needs (tools, structured output or images),
PrimaGate raises `UnsupportedParamsError` before it sends anything. It never drops a feature
quietly. To continue without the feature, opt in by naming it in `drop_params`, and read what was
dropped from `response.metadata.dropped_params`.

```python
from primagate import UnsupportedParamsError, completion

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this picture?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ],
    }
]

try:
    completion(model="claude-sonnet-5-5", messages=messages)
except UnsupportedParamsError as error:
    print("Caught:", error)

response = completion(model="claude-sonnet-5-5", messages=messages, drop_params=["image"])
print(response.choices[0].message.content)
print("Dropped:", response.metadata.dropped_params)
```

The model answers from the text alone, because the image was dropped.

The names you can drop are `"tools"`, `"response_format"` and `"image"`. `drop_params=True` is
different: it only drops small tuning options a provider ignores or rejects, such as `seed` on
Anthropic, and never a feature. Set `drop_params` on `PrimaGate(...)` to apply it to every call
from that instance; a value passed to a single call wins. `primagate.drop_params = [...]` does the
same for the shared default instance used by `completion` and `acompletion`.

Every error PrimaGate raises derives from `PrimaGateError`, and you can import each one from
`primagate`: for example `RateLimitError`, `AuthenticationError`, `Timeout` and `BadRequestError`.

### 7. Handle failures

When a call fails, PrimaGate raises one of its own errors. They all derive from `PrimaGateError`,
and the class tells you why:

- `Timeout`, `RateLimitError`, `ServiceUnavailableError`, `APIError` and `APIConnectionError`
  are temporary: the provider or the network let you down.
- `AuthenticationError`, `PermissionDeniedError`, `NotFoundError`, `ContextWindowExceededError`
  and `ContentPolicyViolationError` mean this model or key cannot serve the request, though
  another one might.
- `BadRequestError`, `UnsupportedParamsError` and `ImageFetchError` mean the request itself is
  wrong, so trying elsewhere would not help.

A plain call makes one request. If you declare redundant endpoints (`provider_routing`) or
fallback models (`fallbacks`, `default_fallbacks`), PrimaGate uses these classes to decide what
comes next: temporary errors are retried and then moved on, errors of the second kind move on
to the next endpoint or fallback, and errors of the third kind stop at once.

Every response and every error records what happened in `attempt_chain`: one row per model that
was tried, in order, each listing its dispatches (the `endpoint`, the number of `retries` and the
`error` it ended with). A row with `skipped` set is a fallback that was passed over without being
sent, because it could not do what the request needs; `skipped` holds the reason, and such a row
is not an attempt.

When more than one model was tried, or a failed attempt was followed only by fallbacks that were
passed over, PrimaGate raises one `AttemptsExhaustedError` instead of only the last error.
`error.attempts` lists the models that were really tried, `error.attempt_chain` also shows the
ones passed over, and `error.last_error` is the error that ended the chain. A single model that
fails raises its own error, with `attempt_chain` attached. `AttemptsExhaustedError` copies
`status_code` and `llm_provider` from `last_error`. If you caught a specific error such as
`RateLimitError` around a call that has fallbacks, catch `AttemptsExhaustedError` as well and
look at `error.last_error`.

```python
from primagate import AttemptsExhaustedError, PrimaGateError, completion

try:
    response = completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say hello in five words."}],
    )
except AttemptsExhaustedError as error:
    print("Every attempt failed. The last error was:", error.last_error)
    for attempt in error.attempts:
        print(attempt.model, [d.error.type for d in attempt.dispatches if d.error])
except PrimaGateError as error:
    print("The call failed:", error)
    print(error.attempt_chain)
else:
    print(response.choices[0].message.content)
    for attempt in response.metadata.attempt_chain:
        print(attempt.model, [d.endpoint for d in attempt.dispatches])
```

A stream that fails after its first chunk has already reached you ends with the plain error, not
`AttemptsExhaustedError`, and carries `attempt_chain` too.

### 8. Watch what PrimaGate does

Every call reports what it does as events. Each event is delivered to the standard logger named
`primagate` (at `INFO`, which Python hides until you set that level, for example with
`logging.basicConfig(level=logging.INFO)`) and, if you pass `event_handler`, to your function,
once each and in order, on the thread that made the call. The handler must be a plain function,
not `async def`, and it runs inline, so a slow one slows the call:

```python
from primagate import Event, PrimaGate


def show(event: Event) -> None:
    print(event.request_id, event.event_type)


gate = PrimaGate(event_handler=show)
response = gate.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.choices[0].message.content)
```

All the events of one call share a `request_id` (it starts with `req-`), and each has a UTC
`timestamp`. The events are:

- `dispatch_started`: a request is about to be sent to a model at an endpoint.
- `dispatch_failed`: it failed; `next` says what follows: `retry`, `next_endpoint`,
  `next_attempt` or `raise`.
- `retry_scheduled`: PrimaGate will wait `delay` seconds and try again.
- `attempt_skipped`: a fallback was passed over, with the `reason`.
- `call_succeeded` and `call_failed`: the call is over.

An event says what happened, not what was said: it carries model names, endpoints, error class
names and status codes, never message text, error text or keys. Your handler must be quick and
safe to call from several threads. If it raises, PrimaGate logs the traceback and carries on;
the one exception is cancellation, which always propagates. A stream that opens
successfully reports its start but not `call_succeeded`, and a failure after its first chunk
is not reported.

## Notes

- [Proxies and certificates](#install-notes)

## Install notes

PrimaGate talks to providers through `httpx2` and needs no provider SDK. `httpx2` uses
your operating system's certificate store (via `truststore`), so corporate proxies whose
certificates are installed system-wide work without extra configuration.
