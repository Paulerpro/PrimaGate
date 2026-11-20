"""First call from a clean install: no key, no network, public API only. Prints `ok`.

It must pass under `python -I` in an environment holding only the installed package, so it imports
nothing from the repository. The reply is scripted at the transport, which is the same seam the
recorded conformance suite uses, so a failure here means the install is broken, not the provider.
"""

from __future__ import annotations

import sys

REPLY = "Hello from the scripted reply"


def main() -> int:
    import httpx2

    from primagate import PrimaGate

    def handle(request: httpx2.Request) -> httpx2.Response:
        if request.url.host != "api.openai.com":
            raise AssertionError(f"unexpected host: {request.url.host}")
        return httpx2.Response(
            200,
            json={
                "id": "chatcmpl-smoke",
                "created": 1,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": REPLY},
                    }
                ],
            },
        )

    # A made-up key: the scripted transport never leaves the process, so nothing real is sent.
    gate = PrimaGate(api_keys={"openai": "sk-smoke"}, transport=httpx2.MockTransport(handle))
    try:
        response = gate.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say hello."}],
        )
    finally:
        gate.close()
    got = response.choices[0].message.content
    if got != REPLY:
        print(f"error: expected {REPLY!r}, got {got!r}", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
