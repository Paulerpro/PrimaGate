from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import cast

import httpx2

from primagate.drivers.async_ import AsyncDriver
from primagate.drivers.providers import check_provider_names
from primagate.drivers.sync import SyncDriver
from primagate.schema.errors import BadRequestError, PrimaGateError
from primagate.schema.types import ChatCompletion, ChatCompletionRequest, Message

DEFAULT_TIMEOUT = 60.0

MessageInput = Message | Mapping[str, str | None]


class PrimaGate:
    """Owns configuration and clients. Everything stateful lives on the instance."""

    def __init__(
        self,
        api_keys: Mapping[str, str] | None = None,
        *,
        base_urls: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        # Test seam: must serve both modes, as httpx2.MockTransport does.
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        check_provider_names("api_keys", api_keys or {})
        check_provider_names("base_urls", base_urls or {})
        self._api_keys = dict(api_keys or {})
        self._base_urls = dict(base_urls or {})
        self._timeout = timeout
        self._http_transport = transport
        self._driver: SyncDriver | None = None
        self._async_driver: AsyncDriver | None = None
        self._closed = False
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"PrimaGate(closed={self._closed})"

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
        driver = self._get_driver()
        request = _request(model, messages, temperature, top_p, max_tokens, stop, n, seed)
        return driver.run(request)

    async def acompletion(
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
        driver = self._get_async_driver()
        request = _request(model, messages, temperature, top_p, max_tokens, stop, n, seed)
        return await driver.run(request)

    def close(self) -> None:
        """Close the sync client. Async clients are released by `aclose()`."""
        with self._lock:
            self._closed = True
            if self._driver is not None:
                self._driver.close()

    async def aclose(self) -> None:
        with self._lock:
            self._closed = True
            driver, async_driver = self._driver, self._async_driver
        if driver is not None:
            driver.close()
        if async_driver is not None:
            await async_driver.aclose()

    def _get_driver(self) -> SyncDriver:
        with self._lock:
            self._check_open()
            if self._driver is None:
                self._driver = SyncDriver.with_builtin_adapters(
                    api_keys=self._api_keys,
                    base_urls=self._base_urls,
                    timeout=self._timeout,
                    http_transport=self._http_transport,
                )
            return self._driver

    def _get_async_driver(self) -> AsyncDriver:
        with self._lock:
            self._check_open()
            if self._async_driver is None:
                self._async_driver = AsyncDriver.with_builtin_adapters(
                    api_keys=self._api_keys,
                    base_urls=self._base_urls,
                    timeout=self._timeout,
                    http_transport=cast(httpx2.AsyncBaseTransport | None, self._http_transport),
                )
            return self._async_driver

    def _check_open(self) -> None:
        if self._closed:
            raise PrimaGateError("PrimaGate instance is closed")


def _request(
    model: str,
    messages: Sequence[MessageInput],
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    stop: str | list[str] | None,
    n: int | None,
    seed: int | None,
) -> ChatCompletionRequest:
    if isinstance(messages, str | bytes):
        raise BadRequestError("messages must be a list of messages")
    return ChatCompletionRequest(
        model=model,
        messages=[_message(m) for m in messages],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        n=n,
        seed=seed,
    )


def _message(value: MessageInput) -> Message:
    if isinstance(value, Message):
        return value
    if not isinstance(value, Mapping) or not value.get("role"):
        raise BadRequestError("Each message needs a 'role'")
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


async def acompletion(
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
    return await get_default().acompletion(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop,
        n=n,
        seed=seed,
    )
