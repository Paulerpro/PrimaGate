from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Required, TypedDict, cast, overload

import httpx2
from typing_extensions import Unpack

from primagate.drivers.async_ import AsyncDriver, AsyncStream
from primagate.drivers.providers import (
    check_drop_params,
    check_provider_names,
    parse_provider_routing,
)
from primagate.drivers.sync import SyncDriver, SyncStream
from primagate.schema.errors import BadRequestError, PrimaGateError
from primagate.schema.types import (
    ChatCompletion,
    ChatCompletionRequest,
    FunctionCall,
    Message,
    ToolCall,
)

DEFAULT_TIMEOUT = 60.0

MessageInput = Message | Mapping[str, Any]
DropParams = bool | Sequence[str] | None


class CallParams(TypedDict, total=False):
    """The one place the call's parameters are spelled out."""

    model: Required[str]
    messages: Required[Sequence[MessageInput]]
    temperature: float | None
    top_p: float | None
    max_tokens: int | None
    stop: str | list[str] | None
    n: int | None
    seed: int | None
    tools: Sequence[Mapping[str, Any]] | None
    response_format: Mapping[str, Any] | type | None
    drop_params: DropParams


class PlainCallParams(CallParams, total=False):
    stream: Literal[False] | None


class StreamCallParams(CallParams):
    stream: Required[Literal[True]]


class AnyCallParams(CallParams, total=False):
    stream: bool | None


class PrimaGate:
    """Owns configuration and clients. Everything stateful lives on the instance."""

    def __init__(
        self,
        api_keys: Mapping[str, str] | None = None,
        *,
        base_urls: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        drop_params: DropParams = None,
        provider_routing: Mapping[str, Sequence[Mapping[str, str]]] | None = None,
        # Test seam: must serve both modes, as httpx2.MockTransport does.
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        check_provider_names("api_keys", api_keys or {})
        check_provider_names("base_urls", base_urls or {})
        check_drop_params(drop_params)
        # Validated here so a bad declaration fails at setup, not on the first failing call.
        self._provider_routing = parse_provider_routing(provider_routing)
        self._drop_params = drop_params
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

    @overload
    def completion(self, **params: Unpack[StreamCallParams]) -> SyncStream: ...

    @overload
    def completion(self, **params: Unpack[PlainCallParams]) -> ChatCompletion: ...

    @overload
    def completion(self, **params: Unpack[AnyCallParams]) -> ChatCompletion | SyncStream: ...

    def completion(self, **params: Unpack[AnyCallParams]) -> ChatCompletion | SyncStream:
        return self._completion(params)

    def _completion(self, params: AnyCallParams) -> ChatCompletion | SyncStream:
        driver = self._get_driver()
        request, drop = _request(params), self._effective_drop_params(params)
        if request.stream:
            return driver.stream(request, drop)
        return driver.run(request, drop)

    @overload
    async def acompletion(self, **params: Unpack[StreamCallParams]) -> AsyncStream: ...

    @overload
    async def acompletion(self, **params: Unpack[PlainCallParams]) -> ChatCompletion: ...

    @overload
    async def acompletion(
        self, **params: Unpack[AnyCallParams]
    ) -> ChatCompletion | AsyncStream: ...

    async def acompletion(self, **params: Unpack[AnyCallParams]) -> ChatCompletion | AsyncStream:
        return await self._acompletion(params)

    async def _acompletion(self, params: AnyCallParams) -> ChatCompletion | AsyncStream:
        driver = self._get_async_driver()
        request, drop = _request(params), self._effective_drop_params(params)
        if request.stream:
            return await driver.stream(request, drop)
        return await driver.run(request, drop)

    def _effective_drop_params(self, params: AnyCallParams) -> DropParams:
        # Only None means unset, so False and [] switch an inherited opt-in off for one call.
        value = params.get("drop_params")
        return self._drop_params if value is None else value

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


# Spelled out because postponed annotations hide Required[] from TypedDict introspection.
_REQUIRED_PARAMS = frozenset({"model", "messages"})
_PARAM_NAMES = frozenset(AnyCallParams.__annotations__)


def _request(params: AnyCallParams) -> ChatCompletionRequest:
    # **params no longer lets Python reject a misspelled or missing keyword, so do it here.
    unknown = sorted(params.keys() - _PARAM_NAMES)
    if unknown:
        raise TypeError(f"Unexpected keyword argument(s): {', '.join(unknown)}")
    for required in sorted(_REQUIRED_PARAMS - params.keys()):
        raise TypeError(f"Missing required keyword argument: '{required}'")
    messages = params["messages"]
    if isinstance(messages, str | bytes):
        raise BadRequestError("messages must be a list of messages")
    return ChatCompletionRequest(
        model=params["model"],
        messages=[_message(m) for m in messages],
        temperature=params.get("temperature"),
        top_p=params.get("top_p"),
        max_tokens=params.get("max_tokens"),
        stop=params.get("stop"),
        n=params.get("n"),
        seed=params.get("seed"),
        stream=params.get("stream") or None,
        tools=_tools(params.get("tools")),
        response_format=_response_format(params.get("response_format")),
    )


def _message(value: MessageInput) -> Message:
    if isinstance(value, Message):
        message = value
    elif isinstance(value, Mapping) and value.get("role"):
        message = Message(
            role=str(value["role"]),
            content=value.get("content"),
            name=value.get("name"),
            tool_calls=_tool_calls(value.get("tool_calls")),
            tool_call_id=value.get("tool_call_id"),
        )
    else:
        raise BadRequestError("Each message needs a 'role'")
    if message.role == "tool" and not message.tool_call_id:
        raise BadRequestError("A 'tool' message needs a 'tool_call_id'")
    return message


def _tool_calls(raw: Sequence[ToolCall | Mapping[str, Any]] | None) -> list[ToolCall] | None:
    if not raw:
        return None
    calls: list[ToolCall] = []
    for item in raw:
        if isinstance(item, ToolCall):
            calls.append(item)
            continue
        try:
            function = item["function"]
            call = ToolCall(
                id=item["id"],
                type=item.get("type", "function"),
                function=FunctionCall(name=function["name"], arguments=function["arguments"]),
            )
            valid = all(
                isinstance(v, str) for v in (call.id, call.function.name, call.function.arguments)
            )
        except (KeyError, TypeError, AttributeError):
            valid = False
        if not valid:
            raise BadRequestError(
                "Each tool call needs string 'id', 'function.name' and 'function.arguments'"
            )
        calls.append(call)
    return calls


def _tools(raw: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]] | None:
    if raw is None:
        return None
    if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
        raise BadRequestError("tools must be a list of tool definitions")
    tools: list[dict[str, Any]] = []
    for tool in raw:
        function = tool.get("function") if isinstance(tool, Mapping) else None
        name = function.get("name") if isinstance(function, Mapping) else None
        if (
            not isinstance(tool, Mapping)
            or tool.get("type") != "function"
            or not isinstance(name, str)
            or not name
        ):
            raise BadRequestError(
                "Each tool must be {'type': 'function', 'function': {'name': ..., ...}}"
            )
        tools.append(dict(tool))
    # An empty list means no tools; some providers reject an explicit empty "tools" array.
    return tools or None


def _response_format(raw: Mapping[str, Any] | type | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    # Duck-typed so pydantic is never imported; any class exposing the method works.
    schema_of = getattr(raw, "model_json_schema", None)
    if isinstance(raw, type) and callable(schema_of):
        return {"type": "json_schema", "json_schema": {"name": raw.__name__, "schema": schema_of()}}
    if not isinstance(raw, Mapping):
        raise BadRequestError("response_format must be a dict or a class with model_json_schema()")
    kind = raw.get("type")
    if kind == "text":
        return None
    if kind == "json_object":
        return dict(raw)
    if kind != "json_schema":
        raise BadRequestError("response_format type must be 'json_schema', 'json_object' or 'text'")
    spec = raw.get("json_schema")
    name = spec.get("name") if isinstance(spec, Mapping) else None
    schema = spec.get("schema") if isinstance(spec, Mapping) else None
    if not isinstance(name, str) or not name or not isinstance(schema, Mapping):
        raise BadRequestError(
            "A json_schema response_format needs json_schema.name and json_schema.schema"
        )
    return dict(raw)


def _with_module_drop_params(params: AnyCallParams) -> AnyCallParams:
    """The module-level setting stands in for the default instance's own, read on every call."""
    import primagate

    module_value = getattr(primagate, "drop_params", None)
    if params.get("drop_params") is None and module_value is not None:
        return cast(AnyCallParams, {**params, "drop_params": module_value})
    return params


_default: PrimaGate | None = None
_default_lock = threading.Lock()


def get_default() -> PrimaGate:
    global _default
    with _default_lock:
        if _default is None:
            _default = PrimaGate()
        return _default


@overload
def completion(**params: Unpack[StreamCallParams]) -> SyncStream: ...


@overload
def completion(**params: Unpack[PlainCallParams]) -> ChatCompletion: ...


@overload
def completion(**params: Unpack[AnyCallParams]) -> ChatCompletion | SyncStream: ...


def completion(**params: Unpack[AnyCallParams]) -> ChatCompletion | SyncStream:
    return get_default()._completion(_with_module_drop_params(params))


@overload
async def acompletion(**params: Unpack[StreamCallParams]) -> AsyncStream: ...


@overload
async def acompletion(**params: Unpack[PlainCallParams]) -> ChatCompletion: ...


@overload
async def acompletion(**params: Unpack[AnyCallParams]) -> ChatCompletion | AsyncStream: ...


async def acompletion(**params: Unpack[AnyCallParams]) -> ChatCompletion | AsyncStream:
    return await get_default()._acompletion(_with_module_drop_params(params))
