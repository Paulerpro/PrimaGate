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

## Notes

- [Proxies and certificates](#install-notes)

## Install notes

PrimaGate talks to providers through `httpx2` and needs no provider SDK. `httpx2` uses
your operating system's certificate store (via `truststore`), so corporate proxies whose
certificates are installed system-wide work without extra configuration.
