# PrimaGate

One call shape for LLM providers, embedded in your process.

```python
from primagate import completion

response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
print(response.choices[0].message.content)
```

The key comes from `OPENAI_API_KEY`, or pass `PrimaGate(api_key=...)` and call
`.completion(...)` on that instance. A `model` of `gpt-4o` and `openai/gpt-4o` both
route to OpenAI.

## Install notes

PrimaGate talks to providers through `httpx2` and needs no provider SDK. `httpx2` uses
your operating system's certificate store (via `truststore`), so corporate proxies whose
certificates are installed system-wide work without extra configuration.
