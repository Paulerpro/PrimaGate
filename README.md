# PrimaGate

One call shape for LLM providers, embedded in your process.

## Scope

Stage 1 is this SDK, embedded in your process: calls, reliability routing, opt-in Provider Routing
and opt-in Tier Routing. Stage 2, a hosted gateway, is a future direction and is not designed here.
See the [release notes](https://github.com/Paulerpro/PrimaGate/blob/main/docs/release-notes.md) for
what each version ships and what is still unproven.

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

`CancelledError` is not a failure: it means you cancelled the call (step 19), so nothing is
retried or moved on to.

A plain call makes one request. If you declare redundant endpoints (`provider_routing`) or
fallback models (`fallbacks`, `default_fallbacks`), PrimaGate uses these classes to decide what
comes next: temporary errors are retried and then moved on, errors of the second kind move on
to the next endpoint or fallback, and errors of the third kind stop at once.

A Tier Routing call (step 16) goes through up to three steps, in a fixed order:

1. The chosen Tier's model, with its retries and redundant endpoints.
2. Tier Failover: when the chosen Tier is SMALL or POWERFUL, MEDIUM once, as one more attempt
   with its own retries and redundant endpoints (or a `skipped` row saying why it could not run).
   Only MEDIUM is tried, never another Tier, and never more than once. A call that chose MEDIUM
   has no such step.
3. Your fallback models, strictly in the order you listed them.

Errors of the first or second kind move the call to the next step; errors of the third kind at
any step stop the whole chain at once, as always.

The list used is the one you declared for the chosen Tier's model in `fallbacks`, or
`default_fallbacks` when that model has no list of its own. When routing could not choose (step
16), the list is the one for the model of the Tier the call actually went to: your
`interim_binding` Tier, or the nearest able Tier when that one cannot take the request. Each
fallback is a plain model, so it is treated exactly as in a plain call: it never goes back through
Tier Routing, it has its own retries and redundant endpoints, it is judged by what its provider
can do (not by any Tier's `limits`), and it is passed over with a `skipped` row when it cannot
take the request. No model is sent twice in one call: an entry naming the chosen Tier's model is
left out, and so is an entry naming MEDIUM's model when MEDIUM was tried as the failover. A MEDIUM
passed over as `same model as the resolved Tier` is bound to the model that already ran, so its
model stays out too. If MEDIUM was passed over for any other reason, an entry naming its model
still runs, as a fallback. So a routed call makes at most one attempt on the chosen Tier, one on
MEDIUM and one per fallback you declared, and PrimaGate never adds a model you did not list. Each
attempt can itself be several requests: its retries on each of its redundant endpoints.

A model's list in `fallbacks` serves both plain calls to that model and routed calls that resolve
to it, and `default_fallbacks` applies to both; step 16 shows how to keep them apart.

Every response and every error records what happened in `attempt_chain`: one row per model that
was tried, in order, each listing its dispatches (the `endpoint`, the number of `retries` and the
`error` it ended with). A row with `skipped` set is a fallback, or the MEDIUM Tier of a Tier
Routing call, that was passed over without being sent; `skipped` holds the reason, and such a row
is not an attempt. A fallback is passed over when it cannot do what the request needs. MEDIUM is
passed over when it is not in `enabled_tiers` (`MEDIUM is not in enabled_tiers`), when it cannot
take the request (the same reason routing gives for removing it), or when it is bound to the
model that just failed (`same model as the resolved Tier`), so no model is run twice.

When more than one model was tried, or a failed attempt was followed only by candidates that were
passed over, PrimaGate raises one `AttemptsExhaustedError` instead of only the last error.
`error.attempts` lists the models that were really tried, `error.attempt_chain` also shows the
ones passed over, and `error.last_error` is the error that ended the chain. A single model that
fails with nothing passed over raises its own error, with `attempt_chain` attached.
`AttemptsExhaustedError` copies `status_code` and `llm_provider` from `last_error`. If you caught
a specific error such as `RateLimitError` around a call that has fallbacks, catch
`AttemptsExhaustedError` as well and look at `error.last_error`.

The same holds for a Tier Routing call. When the chosen Tier SMALL or POWERFUL fails with an
error of the first or second kind, the call raises `AttemptsExhaustedError` even if MEDIUM was
passed over and nothing else was sent, because the passed-over MEDIUM is a row of its own.
`error.last_error` is always the error of the last step that ran: the chosen Tier's, MEDIUM's, or
the last fallback's once fallbacks run. Catch `AttemptsExhaustedError` around routed calls too.
An error of the third kind on the chosen Tier, with nothing else tried, is raised as itself, as
before. Once a second model has been tried (MEDIUM or a fallback), an error of the third kind
still ends the chain at once, but it arrives as the `last_error` of an `AttemptsExhaustedError`.

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

In short, the layers always run in this order: Provider Routing and retries (a model's own
endpoints), then Tier Failover (MEDIUM once, only for a Tier Routing call whose chosen Tier is
SMALL or POWERFUL), then the Fallback List. To watch each layer fail and the request survive, run
the forced-failure demo in the repository. It needs no key in its default, recorded mode, and can
also run against OpenAI. See
[examples/README.md](https://github.com/Paulerpro/PrimaGate/blob/main/examples/README.md) for how
to run both modes.

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
- `attempt_skipped`: a fallback, or the MEDIUM Tier Failover, was passed over, with the `reason`.
- `cache_hit`: the call was answered from memory and nothing was sent (see step 12). It is the
  only event of such a call.
- `cost_unknown`: the model that answered has no price, so there is no cost Estimate (see step 10).
- `call_succeeded` and `call_failed`: a call that was sent is over. A call answered from memory
  ends with `cache_hit` instead.
- `call_cancelled`: you cancelled the call, so it stopped (see step 19). It carries only the
  `request_id`. It can follow `call_succeeded` when the stop lands after the answer arrived but
  before the answer was stored in memory: the call then raises and the answer is not cached.
- `routing_decision`, `low_confidence_downgrade`, `premium_routing_selected` and
  `routing_degraded`: decision events for Tier Routing (see steps 13 and 16).
- `spending_ceiling_exceeded`: routed calls, together with Profile preparation's Analyzer
  requests, have cost more than your Policy's `spending_ceiling`, and says how much: it carries
  the `ceiling` and the `total`, in US dollars and estimated from the price map (see step 15).
  When a preparation request is the one that passes it, its `request_id` starts with `prep-`.
- `preparation_started`, `preparation_failed` and `profile_ready`: PrimaGate is turning your
  Profile sources into a Profile in the background. It starts on the first routed request
  (step 18). They belong to the preparation, not to a call, so their `request_id` starts
  with `prep-`. Each carries the `input_fingerprint` and the `prep_prompt_version`.
  `preparation_failed` is a warning that adds a `reason` (`overflow`, `provider_error`,
  `malformed_output` or `store_error`, when your Profile store raised), the `attempt` number (0
  when nothing was sent) and whether it was `final`. `profile_ready` adds the `profile_id`, which
  is the preparation's `prep-` id. They never carry your sources or what the Analyzer said.

An event says what happened, not what was said: it carries model names, endpoints, error class
names and status codes, never message text, error text or keys. Your handler must be quick and
safe to call from several threads: besides your own calls, preparation (step 18) calls it from its
daemon thread, or from a background task on your event loop. If it raises, PrimaGate logs the
traceback and carries on; the one exception is cancellation, which always propagates. A stream
that opens successfully reports its start but not `call_succeeded`, and a failure after its first
chunk is not reported.

### 9. Keep conversations out of your logs

By default, events and log lines say what PrimaGate did and never what was said: no message
text goes into them. The `payload_retention` setting is the only switch for that. When you set it
to `True`, the messages you sent are added to each `dispatch_started` event (its `messages` field)
and the messages you got back to `call_succeeded` (its `reply_messages` field), and both are also
written to the `primagate` logger. A retry or a fallback sends the conversation again, so each
repeats it. A streamed reply is never retained, because a stream reports no `call_succeeded`.

```python
from primagate import DispatchStarted, Event, PrimaGate


def show(event: Event) -> None:
    if isinstance(event, DispatchStarted):
        print("Sent:", event.messages)


gate = PrimaGate(payload_retention=True, event_handler=show)
response = gate.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.choices[0].message.content)
```

What the setting does not change:

- Provider API keys never appear in events, logs, `repr` or error tracebacks, whatever the
  setting. A key that a provider echoes back in an error message is replaced by `***`.
- An endpoint shown in an event or in `attempt_chain` is cut down to its scheme, host, port and
  path, so a key placed in a URL's user info or query string stays out. A URL that cannot be cut
  down that way is shown as the provider's name.
- The provider's error text is never put into an event, because providers sometimes repeat your
  prompt in it. The error you catch still carries that text.
- The provider's error text also stays in the `attempt_chain` you are given, which holds no
  message content but is yours to keep private.

Turn `payload_retention` on only where logs and handlers are as private as your users' messages.
What the Tier Routing Analyzer reads is described in step 17.

### 10. See what a call cost

Every response says how many tokens it used and what that probably cost:

```python
from primagate import completion

response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.choices[0].message.content)

meta = response.metadata
if meta.cost:
    print(f"About ${meta.cost.total:.6f} for {meta.token_counts.total_tokens} tokens")
else:
    print("No cost Estimate for this response")
```

`metadata.token_counts` holds the prompt, completion and total tokens the provider reported, and
`metadata.cost` holds the `input`, `output` and `total` cost in US dollars. They are Estimates:
the prices come from a small table bundled with PrimaGate, which can be out of date, so do not
use them as a bill. Nothing is ever blocked, retried or changed because of an Estimate.

An Estimate counts only the prompt and completion tokens at the standard rates. It leaves out
discounts and surcharges such as cached input, batch pricing and long-prompt tiers, so a call that
uses them can cost less or more than the figure.

If a model is not in the table, `metadata.cost` is `None`, `metadata.unknown_model` is `True`,
and PrimaGate logs a `cost_unknown` warning on the `primagate` logger once per call. The call still
succeeds. If a response carries no usage, there are no counts and no cost. For a streamed call the
same metadata is on the last chunk, but a stream raises no `cost_unknown` event, so check
`unknown_model` there.

Pass your own prices, in US dollars per million tokens, to replace the bundled ones or to price a
model that has none. Your prices win, and a price for a base model such as `gpt-4o` also covers
its dated snapshots. Each key is a model name (`gpt-4o` or `openai/gpt-4o`, once) and each value
has exactly `input` and `output`, finite and not below zero; a mistake raises `BadRequestError`
when you create the instance:

```python
from primagate import PrimaGate

gate = PrimaGate(price_overrides={"gpt-4o": {"input": 2.0, "output": 8.0}})
response = gate.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.choices[0].message.content)
```

The bundled table, where its prices come from and how to refresh it are in
[docs/price-map.md](docs/price-map.md).

### 11. Count tokens when the provider does not

OpenAI and Anthropic report how many tokens a call used, and PrimaGate uses their figures. When a
response reports none, PrimaGate counts the text itself:

- With the optional `tokens` extra installed (`pip install "primagate[tokens]"`), it counts with
  tiktoken.
- Without it, or if tiktoken cannot load, it uses a rough count of one token per four characters.
  The response then says so: `metadata.token_counts.heuristic` is `True`.

Either way the call succeeds, and the cost Estimate from step 10 is priced from those counts. The
counts are in `response.metadata`; `response.usage` stays `None` because it holds only what the
provider reported. A stream that reports no usage is not counted.

The counts cover message text and tool-call names and arguments. They leave out tool definitions,
response schemas and images, and tiktoken's vocabulary is an approximation for models that are not
OpenAI's, so treat them as a floor, not a bill.

To count your own way, pass a function that takes the model name and some text and returns a whole
number of tokens. It replaces the default, and its counts are not marked as a rough count. If it
raises or returns anything else, PrimaGate uses the rough count for that call and logs a warning;
the call never fails because of counting:

```python
from primagate import PrimaGate


def count(model: str, text: str) -> int:
    return len(text.split())


gate = PrimaGate(token_counter=count)
response = gate.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.choices[0].message.content)
```

The function must be a plain function, not `async def`. It is called twice per counted response,
once for the prompt and once for the reply, and from a worker thread when you use `acompletion`,
so it must be safe to call from several threads and quick.

**Using tiktoken offline.** tiktoken downloads each vocabulary file the first time it needs one.
On a machine without internet access, run your app once where there is access, with the
environment variable `TIKTOKEN_CACHE_DIR` pointing at a folder, and copy that folder to the
offline machine. Set `TIKTOKEN_CACHE_DIR` there to the same path, and tiktoken finds the files
without downloading. If it still cannot load them, PrimaGate logs one warning and uses the rough
count from then on.

### 12. Don't pay twice for the same call

PrimaGate remembers answers in memory. When you make the exact same call again on the same
`PrimaGate` instance (or through the module-level functions, which share one), the answer comes
back from memory and nothing is sent to the provider:

```python
from primagate import completion

messages = [{"role": "user", "content": "Say hello in five words."}]
first = completion(model="gpt-4o", messages=messages)
second = completion(model="gpt-4o", messages=messages)

print(second.choices[0].message.content)
print("From memory:", second.metadata.cache_hit)
```

"The exact same call" means the same model name (as you wrote it), messages, `tools`,
`response_format`, sampling settings (`temperature`, `top_p`, `max_tokens`, `stop`, `n`, `seed`)
and `drop_params`. Change any of them and the call goes to the provider. A call that differs only
in `drop_params` is a different call.

Each caller gets its own copy, so changing a response never changes what the next caller gets. A
copy from memory has `metadata.cache_hit` set to `True` and nothing else in its metadata: no
attempt chain, token counts or cost, because nothing was spent. (`response.usage` still shows what
the original answer used.) Its only event is `cache_hit`, which names the model and a salted hash of the
request, never the content, whatever `payload_retention` says.

Only successful, non-streamed answers are remembered, including one that a fallback model gave, which
is then served for the same call even after your first choice recovers. Calls made at the same
time are each sent, because nothing is remembered until the first answer arrives. A stream is never served from memory and
never saved. The memory is per instance and lives until the instance is gone.

To skip the memory for one call, pass `cache=False`. That call is always sent, and its answer is
not saved. To change how much is kept, set `cache_max_entries` when you create the instance:

```python
from primagate import PrimaGate

gate = PrimaGate(cache_max_entries=100)
messages = [{"role": "user", "content": "Say hello in five words."}]
fresh = gate.completion(model="gpt-4o", messages=messages, cache=False)

print(fresh.choices[0].message.content)
print("From memory:", fresh.metadata.cache_hit)
```

The memory holds at most `cache_max_entries` answers, 256 unless you say otherwise. When it is
full, the answer used least recently is dropped to make room, so an answer you keep reusing
stays. Set it to `0` to turn the memory off for the whole instance. A near-identical call, such
as the same prompt with one character changed, is never answered from memory. A stream ignores
`cache` and the bound. Both limits count answers, not bytes, so very large answers still take
more memory.

### 13. Trace a decision

Tier Routing (step 16) sends each request to a model chosen by an Analyzer. Every such decision is
reported as an event, and an event never holds conversation content, so you can explain a choice
without reading anything anyone said. The four decision events are `routing_decision`,
`low_confidence_downgrade` (the Analyzer was not confident, so the default Tier was used),
`premium_routing_selected` (the chosen Tier costs more than the cheapest eligible one) and
`routing_degraded` (routing could not decide, so the Interim Binding was used). All carry the
fields below; `routing_degraded` adds a `reason`. Their details may grow, but these fields will
stay:

- `profile_id`: the Profile the decision used.
- `input_fingerprint`: the fingerprint of the app inputs that Profile was prepared from.
- `prep_prompt_version` and `analyzer_prompt_version`: the prompts that made the Profile and the
  decision.
- `binding_id`: the model binding that was chosen.

Each is always a string. A further field, `confidence`, is a number from 0 to 1: the confidence the
Analyzer gave its verdict. It is present only when the Analyzer produced a verdict (a
`low_confidence_downgrade` carries the low figure it gave) and absent when the Analyzer gave no
verdict (as when routing degraded), and it holds no content. Reading it is how you check, from your own data, whether a higher confidence goes
with a better outcome. `profile_id`, `input_fingerprint` and `prep_prompt_version` name the
Profile pinned for the call (step 18), even when the call then went to the Interim Binding; when
none was pinned, they are `"unknown"`. Tier Routing emits these events once a call uses the alias;
the sample below hands in one made-up event so it runs without an Analyzer, to show how to look one
up: index them in your handler, and find the ones for an id when a response looks wrong.

```python
from collections import defaultdict

from primagate import DecisionEvent, Event, PrimaGate, RoutingDecision

by_profile: dict[str, list[DecisionEvent]] = defaultdict(list)
by_fingerprint: dict[str, list[DecisionEvent]] = defaultdict(list)
by_binding: dict[str, list[DecisionEvent]] = defaultdict(list)


def keep(event: Event) -> None:
    if isinstance(event, DecisionEvent):
        by_profile[event.profile_id].append(event)
        by_fingerprint[event.input_fingerprint].append(event)
        by_binding[event.binding_id].append(event)


gate = PrimaGate(event_handler=keep)
response = gate.completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.choices[0].message.content)

# A stand-in for an event you will receive, so the lookup below has something to find.
keep(
    RoutingDecision(
        "req-example",
        profile_id="profile-1",
        input_fingerprint="fingerprint-1",
        prep_prompt_version="prep-1",
        analyzer_prompt_version="analyzer-1",
        binding_id="premium-binding",
    )
)

for event in by_binding.get("premium-binding", []):
    print(event.event_type, event.request_id, event.profile_id, event.input_fingerprint)
```

Every event is also written to the `primagate` logger as `name=value` pairs, so without a handler
you can search your logs for `binding_id=premium-binding`. A call answered from memory emits
`cache_hit` and no decision event, because no decision was made. Its `key` is a hash made with a
secret that each `PrimaGate` instance creates for itself: it shows that two hits were the same
request, and it cannot be used to check a guess at a prompt. It changes every time the process
starts, so do not store it.

### 14. Declare what each Tier means

Tier Routing (step 16) sends each request to the cheapest of three
Tiers that can do the job: `SMALL`, `MEDIUM` and `POWERFUL`, in that order of strength. You can
say which model plays each Tier, how to call it, what it can do and what it costs. Only
those three labels are accepted, exactly as written.

```python
from primagate import UNKNOWN, PrimaGate

gate = PrimaGate(
    tier_bindings={
        "SMALL": {
            "model": "gpt-4o-mini",
            "settings": {"temperature": 0.2, "max_tokens": 512},
            "limits": {"context_window": 128000, "supported_features": ["tools", "streaming"]},
            "pricing": {"input": 0.15, "output": 0.6},
        },
        "MEDIUM": {"model": "claude-sonnet-5-5"},
    }
)

# A direct call to a model, like this one, gets none of the SMALL binding's settings.
response = gate.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.choices[0].message.content)
print("cost known:", response.metadata.cost is not None)
print("an undisclosed price is", UNKNOWN)
```

A binding names its `model` (a bare name or `provider/model`) and may add:

- `settings`: `temperature`, `top_p`, `max_tokens`, `seed` and `stop`, which are your own choices,
  so one you do not set is simply absent.
- `limits`: `context_window`, `max_output_tokens` and `supported_features`, a list drawn from
  `tools`, `structured_output`, `images` and `streaming`.
- `pricing`: `input` and `output`, in US dollars per million tokens.

A fact the provider does not publish, such as a context window or a price you leave out, is
stored as the single value `UNKNOWN` and is never left out or replaced by `None`. You can also
write the string `"unknown"` to say so on purpose.

When Tier Routing (step 16) sends a call to a Tier, the binding's `settings` fill in whatever the
request leaves unset. A value you pass on the call itself always wins, so you can override a
setting per request. Passing `None` means "use the binding", so a call can override a binding's
setting but cannot remove it. Calls that name a model directly, like the example above, and the
fallbacks a routed call moves on to get none of a binding's settings.

Setup refuses settings that could never work, whatever `drop_params` says: a setting the bound
provider cannot honour, such as `seed` on Anthropic, and a `max_tokens` above the binding's known
`limits.max_output_tokens`. It also refuses enabled Tiers that bind the same model with
different settings. Tiers may share a model when their settings match.

In `limits`, `supported_features` narrows which Tiers can take a request: a Tier whose list leaves
out a feature the request needs is passed over, while an unknown list narrows nothing.
`context_window` and `max_output_tokens` are recorded but not enforced on calls.

A price in a binding is used for cost estimates (step 10) for that model, after any
`price_overrides` you pass and before the bundled table. A price needs both `input` and `output`
to count, and one model may not be given two different prices.

### 15. Declare your routing rules

Tier Routing follows a Policy that you write once at setup.
The fields that depend on your app and your Analyzer have no default, so setup refuses a Policy that
leaves them out; the optional fields below have the defaults they name. A Policy can only enable
Tiers that have a binding (step 14). Tier labels are exact upper-case words, and naming one twice is
refused.

```python
from primagate import PrimaGate

gate = PrimaGate(
    tier_bindings={
        "SMALL": {"model": "gpt-4o-mini"},
        "MEDIUM": {"model": "gpt-4o"},
    },
    analyzer="gpt-4o-mini",
    policy={
        "enabled_tiers": ["SMALL", "MEDIUM"],
        "confidence_threshold": 0.8,
        "context_token_budget": 2000,
        "interim_binding": "MEDIUM",
    },
)

response = gate.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.choices[0].message.content)
```

`analyzer` is the model that chooses between Tiers (step 16). It is required when the Policy enables
two or more Tiers and optional with one. It must be a concrete model, never the alias.

The call above names a plain model, so the Policy is not involved in it yet.

Required:

- `enabled_tiers`: the Tiers routing may use, from `SMALL`, `MEDIUM` and `POWERFUL`.
- `confidence_threshold`: above 0 and up to 1. How sure the Analyzer must be before its choice is
  taken. Analyzers differ, so tune it for yours.
- `context_token_budget`: how many tokens of recent conversation the Analyzer may read.
- `interim_binding`: the Tier (not the binding itself) used when routing cannot decide, such as while your app is still being
  learned. You name it yourself. `MEDIUM` is a sensible choice, or the cheapest enabled Tier when
  `MEDIUM` is not enabled, but PrimaGate never picks one for you.

Optional:

- `cost_preference`: `balanced` (the default), `cost_conscious` or `capability_first`. Either of the
  last two is accepted, with one warning in the `primagate` log each time a `PrimaGate` is created.
  It has no effect yet: it is advice for later, and routing does not use it.
- `ambiguous_context_default`: the Tier used when the Analyzer is not confident enough. `MEDIUM` when
  you leave it out, so a Policy that does not enable `MEDIUM` must set it.
- `context_exchanges`: how many earlier exchanges the Analyzer may read, a whole number from 0 to 3.
  3 when you leave it out, which is what every Policy did before this field existed. The
  current exchange is always sent whole, and the earlier ones are still kept only within
  `context_token_budget`. 0 shows the Analyzer only the latest request. A value outside 0 to 3,
  or one that is not a whole number, is refused when the instance is created. Adding the field
  changed the Policy `version` that recorded decision events carry, once.
- `max_concurrent_analyzer_calls`: a cap on Analyzer requests in flight at once, counted for each
  `PrimaGate` instance (its sync and async calls together). Left out, the limit is 8, a provisional
  number that will be tuned. Past the limit a call does not wait: it goes to your `interim_binding`
  at once (step 16). The count is not shared between instances or processes, and only calls that
  really ask the Analyzer take a place in it: a repeat answered from memory, or a request only one
  Tier can take, does not.
- `spending_ceiling`: US dollars. A warning, never a limit. Once the estimated cost of the calls
  made with the alias, each one's answer plus its Analyzer request, passes it (a non-streamed
  answer is counted whichever model gave it: the chosen Tier, MEDIUM after Tier Failover, or a
  fallback), one
  `spending_ceiling_exceeded` event is sent with the ceiling and the total, and it is also
  written to the `primagate` log as a warning. It is sent once, and nothing about any call
  changes: later calls are routed and answered exactly as before. The total is kept in memory
  for each `PrimaGate` instance, so a restart starts again from zero. Plain model calls, calls
  answered from memory, streams, models with no known price (step 10) and calls that end in an
  error are not counted, nor are the failed attempts before a retry succeeds, so the total can
  run a little below what you were billed. The one exception for streams: a routed stream that is
  cancelled still counts the Analyzer request it already paid for (step 19), although a routed
  stream that finishes is not counted at all.

`interim_binding` and `ambiguous_context_default` must name enabled Tiers. A Policy is fixed once
set up, so one request can never see its rules change halfway. Today a Policy only checks your setup:
it does not change any call.

`profile_sources` describes your app, so a Profile of it can be prepared for the Analyzer (step 17):

```python
from primagate import PrimaGate

gate = PrimaGate(
    tier_bindings={
        "SMALL": {"model": "gpt-4o-mini"},
        "MEDIUM": {"model": "gpt-4o"},
    },
    analyzer="gpt-4o-mini",
    policy={
        "enabled_tiers": ["SMALL", "MEDIUM"],
        "confidence_threshold": 0.8,
        "context_token_budget": 2000,
        "interim_binding": "MEDIUM",
    },
    profile_sources={
        "instructions": "Answer questions about orders.",
        "tool_descriptions": ["Look up an order by its number", "Cancel an order"],
        "skill_descriptions": ["Explain the refund policy"],
    },
)

response = gate.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.choices[0].message.content)
```

Every key is optional. `instructions` is text, and the two descriptions are lists of non-blank
text (a bare string is not a list, and tool definitions as dicts are not accepted). Sources need an
`analyzer` and a Policy with two or more Tiers, since only the Analyzer reads a Profile. They are
read once, when the instance is created: changing your own list later changes nothing, and to
change a source you create a new instance, for example on restart. A Profile is keyed by a
fingerprint of the sources and the `analyzer` model, so changing either makes a stored Profile
stale. Descriptions are matched as exact text with their order ignored, so added spaces or a
repeated description count as a change. Upgrading PrimaGate can also make a stored Profile stale,
when the prompt that prepares it or the shape of a Profile has changed. The `analyzer` name is taken
as written, so when the provider changes the model behind an undated alias such as `gpt-4o-mini`,
the Profile is not prepared again. Name a dated model version if the fingerprint should follow
the exact model. Prices, Tier Bindings and the rest of the Policy are not part
of the fingerprint. The first routed request prepares a Profile from the sources in the background
(step 18), which sends them to the Analyzer's provider. Once that Profile is ready, the Analyzer
reads it to choose a Tier; until then, routed calls go to your `interim_binding` Tier.

### 16. Ask for Tier Routing

Pass `TIER_ROUTING_ALIAS` (the string `"primagate/auto"`) as `model=` to ask for Tier Routing. Any
other model name is a plain call and skips it. The alias needs `tier_bindings` and a `policy`
(steps 14 and 15); without them the call is refused before anything is sent.

```python
from primagate import TIER_ROUTING_ALIAS, PrimaGate

gate = PrimaGate(
    tier_bindings={"MEDIUM": {"model": "gpt-4o"}},
    policy={
        "enabled_tiers": ["MEDIUM"],
        "confidence_threshold": 0.8,
        "context_token_budget": 2000,
        "interim_binding": "MEDIUM",
    },
)

response = gate.completion(
    model=TIER_ROUTING_ALIAS,
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(response.choices[0].message.content)
```

Before anything is sent, PrimaGate removes every enabled Tier whose model cannot do what the
request needs. It uses the same rules as a plain call, including `drop_params` (step 6), and a
Tier's `limits` can only take features away from what its provider offers, never add them:

- If no Tier can take the request, for example an image when none of them handles images, the call
  fails at once with an `UnsupportedParamsError` that names each Tier and what it lacks.
- If exactly one Tier can, the call goes to that Tier's model.
- If two or more can, the Analyzer (step 17) chooses: it is asked once, with only the Tiers that
  can take the request, and the call goes to the Tier it names. An answer below your
  `confidence_threshold` takes your `ambiguous_context_default` instead. If that Tier cannot take
  the request, the nearest Tier that can is used, stronger ones first.

Every alias call is logged as a `routing_decision` event (step 13), and an answer below the
threshold also as `low_confidence_downgrade`. A Tier above the cheapest one that could have taken
the request also logs `premium_routing_selected`. The events carry ids and the `binding_id` (for
example `MEDIUM:openai/gpt-4o`), never message content. Each call is decided on its own, so each
step of a tool loop has its own decision. `profile_id` and `input_fingerprint` say `unknown`
unless a prepared Profile was pinned for the call (step 18). The response's `metadata.tier` names
the Tier that answered and `metadata.analyzer` holds what the Analyzer cost, apart from
`metadata.cost`. `metadata.tier` is empty when no Tier answered: for an answer from a fallback
model (below) and for an answer from memory. Tell them apart by `metadata.cache_hit`, which is
true only for the answer from memory; a fresh routed answer, a fallback's included, also has
`metadata.analyzer` set whenever the Analyzer was asked.

A repeated call that is answered from memory asks nobody: it has no `routing_decision`, and its
`metadata.tier` and `metadata.analyzer` are empty. Every other alias call with two or more
Tiers able to take it costs one extra, short request to the Analyzer's provider before the real
one, and that request's tokens are in `metadata.analyzer`, not in `metadata.cost`. An Analyzer
model with no known price reports `unknown_model` and no cost, as any model does (step 10).

When routing cannot choose, your request is not failed. If the Analyzer is unreachable or times
out, answers in the wrong shape, names a Tier that cannot take the request, the latest turn is
larger than `context_token_budget`, the Analyzer cap is reached, or your declared sources have no
prepared Profile yet (step 18), the call goes to your `interim_binding` Tier instead. That is one
attempt on a worker model, with no second Analyzer request (retries, provider routing and Tier
Failover still apply as for any routed call), and a `routing_degraded` event (step 13) says why
with one of these reasons: `provider_error`, `malformed_output`, `tier_not_enabled`,
`no_conversation`, `over_budget`, `over_capacity` (the cap in step 15 was reached), `no_analyzer`
(no Analyzer is set), `analyzer_unavailable` (no adapter is registered for the Analyzer's
provider), `preparing` (the Profile was still being prepared when the call started) or
`preparation_failed` (preparing it failed, step 18). If the `interim_binding` Tier cannot take
the request, the nearest Tier that can is used, stronger ones first. Because the Interim Tier may
sit above the cheapest able one, `premium_routing_selected` can follow `routing_degraded` for the
same call. An answer given
this way is never stored in the cache (step 12), so the next identical call asks the Analyzer
again.

If the Tier that was chosen is SMALL or POWERFUL and its model fails after its retries and
redundant endpoints, the call is not lost: PrimaGate tries MEDIUM once, as step 7 describes.
This is Tier Failover. It never asks the Analyzer again and never re-runs routing: MEDIUM is sent
exactly as routing prepared it for this request, with its own `limits` and dropped parameters. The
response's `metadata.tier` then says `MEDIUM`, `metadata.analyzer` still shows the one Analyzer
request, and the spending ceiling (step 15) counts MEDIUM's answer plus that one Analyzer cost.
The answer is cached like any routed answer, unless the call had gone to the Interim Binding. If
MEDIUM is not enabled, cannot take the request, or is bound to the model that failed, it is passed
over with a `skipped` row in `attempt_chain` and an `attempt_skipped` event. That row names
MEDIUM's model, or just `MEDIUM` when you declared no MEDIUM binding at all, since there is no
provider or model to name. When MEDIUM was the chosen Tier, there is no failover.

The decision events (`routing_decision` and the others in step 13) keep the Tier routing chose;
only `metadata.tier` moves to the Tier that answered. The move itself shows as a `dispatch_failed`
event whose `next` is `next_attempt`, the same value a move to a fallback uses, followed by a
`dispatch_started` for MEDIUM's model.

If MEDIUM fails too, or there was no failover, the call moves on to your fallback models (step 7),
using the list declared for the chosen Tier's model. An answer from a fallback has an empty
`metadata.tier`, since no Tier gave it, and `metadata.analyzer` still shows the one Analyzer
request. The spending ceiling counts that fallback's answer plus the one Analyzer cost. The answer
is cached like any routed answer, unless the call had gone to the Interim Binding: it is stored
under the call's own key, so an identical routed call gets the fallback's answer back from memory
(step 12) even after the chosen Tier has recovered. The alias cannot be used in a fallback list.

A routed stream (`stream=True`) follows the same chain. A failure before the first chunk reaches
you is retried and moves on to MEDIUM and then your fallback models exactly as above, and the
stream you get back opens on whichever model sends the first chunk. If every step fails before
a first chunk, the call itself raises, just as a non-streamed call does (`AttemptsExhaustedError`
once more than one row is in the chain). Once a chunk has reached you, nothing is retried or moved
on: the error is raised while you iterate, as the provider's own mapped error (such as
`APIConnectionError`), never `AttemptsExhaustedError`, so `except AttemptsExhaustedError` does not
catch it. Its `attempt_chain` still holds every earlier attempt and `skipped` row plus the one that
was streaming. The last chunk's `metadata` carries `tier` and `analyzer` as a non-streamed answer
does: `tier` is the chosen Tier, `MEDIUM` after Tier Failover, or empty for a fallback's answer;
`analyzer` is the one Analyzer request, or empty when the call went to the Interim Binding
without asking the Analyzer. The spending-ceiling counting described above is for non-streamed
calls: a routed stream that finishes is not counted at all (streams are not counted, step 15),
but one that is cancelled, including during failover, still counts the Analyzer request already
paid (step 19).

Upgrading from a version without this step: routed calls now use the fallback lists you declared,
so if you set `default_fallbacks`, or a list in `fallbacks` for a model one of your Tiers is bound
to, routed calls whose Tier fails now send those fallbacks too. A model's list serves both plain
calls to it and routed calls that resolve to it. To keep `default_fallbacks` off routed calls that
resolve to a model, give that model an empty list, such as `fallbacks={"gpt-4o-mini": []}`; that
also turns fallbacks off for plain calls to that model.

### 17. What the Analyzer reads

Tier Routing asks one model, the Analyzer, which Tier a request needs. That model is not your
worker model, and you choose it. It runs in your process with your own keys, and PrimaGate sends
it only this:

- The Profile of your app: what your instructions, tool descriptions and skill descriptions say
  about it, once they have been boiled down (step 18): its purpose, the kinds of request, the
  rules its answers follow, what its tools can do, and what the sources leave unclear. Without
  `profile_sources`, it only says that all of that is unknown. Preparing the Profile sends the
  sources themselves to the Analyzer's provider.
- A bounded slice of the recent conversation: the latest turn in full, plus at most three earlier
  turns that fit your `context_token_budget` (step 15). That includes the tool calls the assistant
  made, with their arguments, and the tool results. System and developer messages and images are
  not included, and nothing is summarised.

**Both go to whichever Provider hosts the Analyzer model**, so pick an Analyzer whose provider you
are content to show that text to. The request your worker model receives is never changed or cut
down for the Analyzer.

The Analyzer must answer with exactly one Tier word and a confidence, like `MEDIUM 0.82`. Anything
else, including the right words inside a longer reply, is treated as a failed analysis and never
guessed at. An answer below your `confidence_threshold` is not a failure: it takes your
`ambiguous_context_default`. To choose an Analyzer and a `confidence_threshold` from measured
evidence, see [docs/analyzer-guide.md](docs/analyzer-guide.md). The Analyzer's own cost is reported
on its own line, apart from the cost of your worker model.

Each instance holds a Profile store, kept in memory by default and shared by `completion` and
`acompletion`. The default in-memory store writes nothing to disk, so it is empty after a restart.
Only background preparation (step 18) reads or writes it; a call never does. `profile_store=` is
there so another store can be plugged in. A custom store needs a `non_blocking` flag and plain (not
`async def`), thread-safe `read()` and `write(profile)` methods. `read()` returns the stored
Profile or `None`, and `write` replaces it, so a reader sees the old Profile or the new one, never
a mix. An error your store raises during preparation ends that preparation as a `store_error`
failure (step 8); it never reaches your calls. If `non_blocking` is false, `read()` and `write()`
run in a worker thread when preparing from `acompletion`, so your event loop is never held up.

`profile_store=primagate.FileProfileStore("profile.json")` keeps the Profile in a file, so a
prepared Profile is reused across restarts with no new Analyzer request, as long as its fingerprint
still matches. Its format, how it is versioned and its limits when several processes share one
file are in [docs/profile-store.md](docs/profile-store.md).

### 18. Background preparation and closing

Preparing the Profile is the only work PrimaGate does in the background. Nothing else runs outside
your own calls.

- **When it starts.** On an instance with `profile_sources`, the first routed request
  (`model=TIER_ROUTING_ALIAS`) with two or more Tiers able to take it starts it. Plain model calls,
  calls answered from memory and instances without sources never start it. That request does not
  wait for it and cannot fail because of it.
- **What it does to routed calls.** Each routed call with two or more Tiers able to take it looks
  at the preparation once, when it starts, and keeps what it saw to its end:
  - *Preparing* (including the call that starts it): the call goes to your `interim_binding` Tier
    with no Analyzer request, and `routing_degraded` says `preparing`.
  - *Ready*: the Analyzer chooses with the prepared Profile, and the decision events (step 13)
    carry its `profile_id`, `input_fingerprint` and `prep_prompt_version`.
  - *Failed*: the call goes to your `interim_binding` Tier with no Analyzer request, and
    `routing_degraded` says `preparation_failed`, until a later preparation succeeds.

  A call already running when the Profile becomes ready finishes as it started; only calls that
  start later use it. An answer from the `interim_binding` Tier is never cached, so once the
  Profile is ready the same request asks the Analyzer. Calls that only one Tier can take, and
  instances without `profile_sources`, are routed as before. After a restart, the first routed
  call is served while preparing even when a file store holds a matching Profile: no call reads
  the store, and the background job loads that Profile for the calls after it.

  So until a Profile is ready, routed calls stay on your `interim_binding` Tier and the Analyzer
  is never asked. That is the case while preparing; after a failure, until the 10-minute restart
  succeeds; after an `overflow` failure, until the process restarts; and, in an app that runs a
  fresh event loop per request or makes one call per process, for good, since preparation never
  gets to finish. If your Interim Tier is the pricier one, those calls cost more than routed ones
  would. Each such call logs `routing_degraded` at WARNING; during warm-up that is expected.
- **How often.** Only one preparation per fingerprint runs at a time, however many requests arrive
  together. Once it is ready it is not prepared again in this process. With a file store, a
  matching Profile from an earlier run is loaded silently: no Analyzer request and no preparation
  events.
- **When it fails.** A final `preparation_failed` event (step 8) says why. A later routed request
  tries again once 10 minutes have passed since the failure. Sources too large to prepare
  (`overflow`) are not tried again until the process restarts.
- **Where it runs.** From `acompletion`, it is a task on your event loop. If that loop ends first,
  as when `asyncio.run(...)` returns, the preparation is cut off with nothing written and no event,
  and it counts as a failure: routed calls go to your `interim_binding` Tier, and the 10-minute
  wait applies before a routed request starts it again.
  An app that runs a fresh event loop per request (`asyncio.run` per request, `async_to_sync`,
  task workers) should make its calls on a long-lived loop, or use `completion`, so preparation
  can finish. From `completion`, it runs on a daemon thread, which never keeps your process from
  exiting. A short-lived script may therefore exit before the preparation finishes, and its next
  run pays for it again. Events reach your handler from that task or thread.
- **Closing.** `gate.close()` stops a running preparation before its next step: it sends nothing
  more and starts no Profile write. `await gate.aclose()` does the same, then cancels the
  preparation tasks on its event loop and waits for them. A store write already under way may
  still complete, atomically and never partly written, and its `profile_ready` may then not be
  sent. Otherwise the store is left as it was and no final event is sent. After either, nothing
  starts again.
- **Forking.** Create the instance after your process forks, for example inside each worker of a
  pre-forking server, never before. Fork safety is not guaranteed in this version: a child forked
  while a preparation runs can inherit its locks half-held.

### 19. Cancel a call

A call can be stopped cleanly in either mode. Once it stops, nothing more is sent, nothing is
cached, no retry or fallback is tried, and one `call_cancelled` event (step 8) is sent. An
answer the provider already billed for (the model's, or the Analyzer's on a routed call) still
counts towards your `spending_ceiling`; for a routed stream, only the Analyzer's is counted, even
mid-failover, since streamed answers are not counted (step 15). A cancelled call is not a
failure, so it never sends `call_failed`. A call whose answer or error has already been
announced ends as announced: a failure raises its own error, and an answer is returned unless
the stop lands before it is stored in memory or counted against your ceiling. Then
`call_succeeded` is followed by `call_cancelled` and the call raises.

**With `completion`**, pass `cancel=CancelToken()` and call `token.cancel()` from any thread, or
pass `deadline=` as a number of seconds from the start of the call, or both. The call checks
them before each step it takes (each request, retry wait, cache or Analyzer step) and before
each chunk of a stream, and raises `primagate.CancelledError` at the first one it finds
cancelled or past its deadline. A stream stopped this way is closed and yields no more chunks.
Without either setting, a call behaves exactly as before.

```python
from primagate import CancelledError, CancelToken, PrimaGate

gate = PrimaGate()
token = CancelToken()
# Any other thread can stop the call, for example a "Stop" button's handler: token.cancel()
try:
    response = gate.completion(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say hello in five words."}],
        cancel=token,
        deadline=30,
    )
except CancelledError:
    print("The call was stopped before it finished.")
else:
    print(response.choices[0].message.content)
```

**With `acompletion`**, cancel the task, as for any other asyncio work: `task.cancel()`,
`asyncio.wait_for(...)` or `asyncio.timeout(...)`. The call stops where it is, sends
`call_cancelled`, and lets `asyncio.CancelledError` through, so your task stays cancelled.
Cancelling the task while you read a stream does the same and closes the stream.
`acompletion` takes `cancel=` and `deadline=` too, checked at the same points, and then raises
`primagate.CancelledError`.

```python
import asyncio

from primagate import PrimaGate


async def main() -> None:
    gate = PrimaGate()
    try:
        # wait_for cancels the call's task if no answer has come within 30 seconds.
        response = await asyncio.wait_for(
            gate.acompletion(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Say hello in five words."}],
            ),
            timeout=30,
        )
        print(response.choices[0].message.content)
    except TimeoutError:
        print("No answer within 30 seconds.")
    finally:
        await gate.aclose()


asyncio.run(main())
```

`cancel` must be a `CancelToken`, and `deadline` a number of seconds above 0; anything else
raises `BadRequestError` before the call starts.

- **A request already sent is not interrupted** by a token or deadline: PrimaGate cannot safely
  cut off a blocking network read, so the call stops at its next step after that request
  returns, and your `timeout` (60 seconds by default) still bounds how long that is. A retry
  wait already begun in `completion` likewise runs out before the call stops. Cancelling an
  `acompletion` task, by contrast, interrupts the request or the wait at once.
- **Close a stream you stop reading**, with `close()`, `await stream.aclose()` or a `with`
  (`async with`) block. Cancellation is noticed on the next read, so a stream you simply drop
  is never told.
- **A token is one-shot.** Once cancelled it stays cancelled, and every later call given it
  stops at once. One token shared by several calls stops them all. Use a new token per call
  unless you mean that.
- **The deadline keeps running while you read a stream**: it counts from the start of the call,
  not from the first chunk.
- **In async code that passes `cancel=` or `deadline=`, expect two errors**:
  `asyncio.CancelledError` when the task was cancelled, and `primagate.CancelledError`
  (`from primagate import CancelledError`) when the token or deadline stopped the call. If you
  catch `asyncio.CancelledError`, raise it again so the task stays cancelled.
- **Analyzer permits come back.** A call cancelled while the Analyzer is deciding releases its
  place under `max_concurrent_analyzer_calls`, so the next call can ask.
- **Background preparation is not a call.** Cancelling a call never stops a preparation it
  started; `close()` and `aclose()` do (step 18).

## Contributing and versioning

- [Contributing](https://github.com/Paulerpro/PrimaGate/blob/main/CONTRIBUTING.md): the issue
  tracker, how changes are merged, and the checks to run.
- [Code of conduct](https://github.com/Paulerpro/PrimaGate/blob/main/CODE_OF_CONDUCT.md).
- [Versioning](https://github.com/Paulerpro/PrimaGate/blob/main/docs/versioning.md): what semantic
  versioning covers.

## Notes

- [Proxies and certificates](#install-notes)

## Install notes

PrimaGate talks to providers through `httpx2` and needs no provider SDK. `httpx2` uses
your operating system's certificate store (via `truststore`), so corporate proxies whose
certificates are installed system-wide work without extra configuration.
