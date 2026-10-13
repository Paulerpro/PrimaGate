from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence

import httpx2

from primagate.drivers.sync import SyncDriver
from primagate.schema.errors import AuthenticationError, BadRequestError, PrimaGateError
from primagate.schema.types import ChatCompletion, ChatCompletionRequest, Message

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TIMEOUT = 60.0

MessageInput = Message | Mapping[str, str | None]


class PrimaGate:
    """Owns configuration and clients. Everything stateful lives on the instance."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._http_transport = transport
        self._driver: SyncDriver | None = None
        self._closed = False
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"PrimaGate(base_url={self._base_url!r}, closed={self._closed})"

    def completion(
        self,
        *,
        model: str,
        messages: Sequence[MessageInput],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        stop: str | list[str] | None = None,
        n: int | None = None,
        seed: int | None = None,
    ) -> ChatCompletion:
        if isinstance(messages, str | bytes):
            raise BadRequestError("messages must be a list of messages", llm_provider="openai")
        driver = self._get_driver()
        request = ChatCompletionRequest(
            model=model,
            messages=[_message(m) for m in messages],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            n=n,
            seed=seed,
        )
        return driver.run(request)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if self._driver is not None:
                self._driver.close()

    def _get_driver(self) -> SyncDriver:
        with self._lock:
            if self._closed:
                raise PrimaGateError("PrimaGate instance is closed")
            if self._driver is None:
                key = self._api_key or os.environ.get("OPENAI_API_KEY")
                if not key:
                    raise AuthenticationError(
                        "No OpenAI API key: pass api_key or set OPENAI_API_KEY",
                        llm_provider="openai",
                    )
                self._driver = SyncDriver.with_builtin_adapters(
                    api_key=key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                    http_transport=self._http_transport,
                )
            return self._driver


def _message(value: MessageInput) -> Message:
    if isinstance(value, Message):
        return value
    if not isinstance(value, Mapping) or not value.get("role"):
        raise BadRequestError("Each message needs a 'role'", llm_provider="openai")
    return Message(role=str(value["role"]), content=value.get("content"), name=value.get("name"))


_default: PrimaGate | None = None
_default_lock = threading.Lock()


def get_default() -> PrimaGate:
    global _default
    with _default_lock:
        if _default is None:
            _default = PrimaGate()
        return _default


def completion(
    *,
    model: str,
    messages: Sequence[MessageInput],
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    stop: str | list[str] | None = None,
    n: int | None = None,
    seed: int | None = None,
) -> ChatCompletion:
    return get_default().completion(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        n=n,
        seed=seed,
    )
