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
- `cache_hit`: the call was answered from memory and nothing was sent (see step 12). It is the
  only event of such a call.
- `cost_unknown`: the model that answered has no price, so there is no cost Estimate (see step 10).
- `call_succeeded` and `call_failed`: a call that was sent is over. A call answered from memory
  ends with `cache_hit` instead.
- `routing_decision`, `low_confidence_downgrade`, `premium_routing_selected` and
  `routing_degraded`: decision events for Tier Routing (see steps 13 and 16).
- `spending_ceiling_exceeded`: routed calls have cost more than your Policy's
  `spending_ceiling`, and says how much: it carries the `ceiling` and the `total`, in US dollars
  and estimated from the price map (see step 15).

An event says what happened, not what was said: it carries model names, endpoints, error class
names and status codes, never message text, error text or keys. Your handler must be quick and
safe to call from several threads. If it raises, PrimaGate logs the traceback and carries on;
the one exception is cancellation, which always propagates. A stream that opens
successfully reports its start but not `call_succeeded`, and a failure after its first chunk
is not reported.

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

Each is always a string. When no Profile is ready yet, the value is `"unknown"`. Tier Routing emits
these events once a call uses the alias; the sample below hands in one made-up event so it runs
without an Analyzer, to show how to look one up: index them in your handler, and find the ones for an id when a response looks wrong.

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
already say which model plays each Tier, how to call it, what it can do and what it costs. Only
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

Declaring a binding is only a declaration for now. Its `settings` and `limits` are stored and
checked but not applied to any call, and ordinary calls behave as before, with one exception: a
price in a binding is used for cost estimates (step 10) for that model, after any
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
- `max_concurrent_analyzer_calls`: a cap on Analyzer requests in flight at once, counted for each
  `PrimaGate` instance (its sync and async calls together). Left out, the limit is 8, a provisional
  number that will be tuned. Past the limit a call does not wait: it goes to your `interim_binding`
  at once (step 16). The count is not shared between instances or processes, and only calls that
  really ask the Analyzer take a place in it: a repeat answered from memory, or a request only one
  Tier can take, does not.
- `spending_ceiling`: US dollars. A warning, never a limit. Once the estimated cost of the calls
  made with the alias, each one's answer plus its Analyzer request, passes it, one
  `spending_ceiling_exceeded` event is sent with the ceiling and the total, and it is also
  written to the `primagate` log as a warning. It is sent once, and nothing about any call
  changes: later calls are routed and answered exactly as before. The total is kept in memory
  for each `PrimaGate` instance, so a restart starts again from zero. Plain model calls, calls
  answered from memory, streams, models with no known price (step 10) and calls that end in an
  error are not counted, nor are the failed attempts before a retry succeeds, so the total can
  run a little below what you were billed.

`interim_binding` and `ambiguous_context_default` must name enabled Tiers. A Policy is fixed once
set up, so one request can never see its rules change halfway. Today a Policy only checks your setup:
it does not change any call.

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
step of a tool loop has its own decision. `profile_id` and `input_fingerprint` say `unknown` until
Profiles exist. The response's `metadata.tier` names the Tier that answered and `metadata.analyzer`
holds what the Analyzer cost, apart from `metadata.cost`.

A repeated call that is answered from memory asks nobody: it has no `routing_decision`, and its
`metadata.tier` and `metadata.analyzer` are empty. Every other alias call with two or more
Tiers able to take it costs one extra, short request to the Analyzer's provider before the real
one, and that request's tokens are in `metadata.analyzer`, not in `metadata.cost`. An Analyzer
model with no known price reports `unknown_model` and no cost, as any model does (step 10).

When routing cannot choose, your request is not failed. If the Analyzer is unreachable or times
out, answers in the wrong shape, names a Tier that cannot take the request, the latest turn is
larger than `context_token_budget`, or the Analyzer cap is reached, the call goes to your
`interim_binding` Tier instead. That is one dispatch to a worker model, with no second Analyzer
request (retries and provider routing for that model still apply as for any call), and a
`routing_degraded` event (step 13) says why with one of these reasons: `provider_error`,
`malformed_output`, `tier_not_enabled`, `no_conversation`, `over_budget` or `over_capacity` (the cap
in step 15 was reached). If the `interim_binding` Tier cannot take the request, the nearest Tier
that can is used, stronger ones first. Because the Interim Tier may sit above the cheapest able
one, `premium_routing_selected` can follow `routing_degraded` for the same call. An answer given
this way is never stored in the cache (step 12), so the next identical call asks the Analyzer
again.

A fallback list is not applied to alias calls yet. The alias cannot be used in a fallback list.

### 17. What the Analyzer reads

Tier Routing asks one model, the Analyzer, which Tier a request needs. That model is not your
worker model, and you choose it. It runs in your process with your own keys, and PrimaGate sends
it only this:

- The Profile of your app: what your instructions, tool descriptions and skill descriptions say
  about it, once they have been boiled down. Profiles are not prepared yet, so for now it only
  says that all of that is unknown. When they are, the sources are sent to the Analyzer's provider
  to make the Profile.
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
`ambiguous_context_default`. The Analyzer's own cost is reported on its own line, apart from the
cost of your worker model.

Each instance holds a Profile store, kept in memory by default and shared by `completion` and
`acompletion`. Nothing goes to disk, so it is empty after a restart. Until Profiles are prepared it
is never read or written; `profile_store=` is there so another store can be plugged in later. A
custom store needs a `non_blocking` flag and plain (not `async def`), thread-safe `read()` and
`write(profile)` methods. `read()` returns the stored Profile or `None`, and `write` replaces it, so
a reader sees the old Profile or the new one, never a mix. Errors it raises reach the caller
unchanged. If `non_blocking` is false, async calls run `read()` and `write()` in a worker thread so
your event loop is never held up.

## Notes

- [Proxies and certificates](#install-notes)

## Install notes

PrimaGate talks to providers through `httpx2` and needs no provider SDK. `httpx2` uses
your operating system's certificate store (via `truststore`), so corporate proxies whose
certificates are installed system-wide work without extra configuration.
