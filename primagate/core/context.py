from __future__ import annotations

import copy
from dataclasses import dataclass

from primagate.core.counting import heuristic_count, messages_text
from primagate.schema.types import ChatCompletionRequest, Message

# Earlier exchanges only; the current one is always shown whole on top of these.
MAX_EXCHANGES = 3
# The Profile already carries the app's instructions, so these stay out of the Analyzer's view.
_LEFT_OUT = frozenset({"system", "developer"})


@dataclass(frozen=True)
class RoutingContext:
    """What the Analyzer may read: copies, oldest first, current exchange last."""

    messages: tuple[Message, ...]
    earlier_kept: int
    # An estimate summed per exchange, a guard on cost and not a billing figure.
    tokens: int


@dataclass(frozen=True)
class OverBudget:
    """The current exchange alone is larger than the budget, so no view can be built."""

    tokens: int
    budget: int


def bounded_context(request: ChatCompletionRequest, budget: int) -> RoutingContext | OverBudget:
    """The slice of a request the Analyzer sees, never altering the request itself.

    The current exchange is never trimmed. Earlier ones are kept newest first, at most
    `MAX_EXCHANGES` and within `budget` tokens, and an exchange is dropped whole or not at all.
    """
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ValueError("budget must be a whole number of 1 or more")
    exchanges = _exchanges(request.messages)
    if not exchanges:
        return RoutingContext((), 0, 0)
    *earlier, current = exchanges
    used = _tokens(current)
    if used > budget:
        return OverBudget(used, budget)
    kept = earlier[-MAX_EXCHANGES:]
    sizes = [_tokens(exchange) for exchange in kept]
    total = used + sum(sizes)
    while kept and total > budget:
        total -= sizes.pop(0)
        kept.pop(0)
    # Copies, so nothing done to the view can reach the request that goes to the worker model.
    shown = tuple(copy.deepcopy(m) for exchange in (*kept, current) for m in exchange)
    return RoutingContext(shown, len(kept), total)


def _exchanges(messages: list[Message]) -> list[list[Message]]:
    """Split into exchanges: each starts at a user message and keeps everything it set going."""
    groups: list[list[Message]] = []
    for message in messages:
        if message.role in _LEFT_OUT:
            continue
        # A user message opens a new exchange; anything before the first one, such as a greeting,
        # becomes an exchange of its own.
        if message.role == "user" or not groups:
            groups.append([])
        groups[-1].append(message)
    return groups


def _tokens(exchange: list[Message]) -> int:
    return heuristic_count(messages_text(exchange))
