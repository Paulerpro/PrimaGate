from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from primagate.core import steps
from primagate.core.call import advance
from primagate.drivers._shared import EventHandler, deliver
from primagate.drivers.spend import SpendTracker
from primagate.schema import events
from primagate.schema.errors import CancelledError, PrimaGateError


class CancelToken:
    """Lets a caller stop a call from any thread: pass it as `cancel=`, then call `cancel()`.

    The call stops at its next step, before anything more is sent. A request already in flight
    is not interrupted; its `timeout` still bounds the wait. One token may serve several calls,
    and cancelling it stops them all.
    """

    def __init__(self) -> None:
        self._set = threading.Event()

    def cancel(self) -> None:
        self._set.set()

    @property
    def cancelled(self) -> bool:
        return self._set.is_set()

    def __repr__(self) -> str:
        return f"CancelToken(cancelled={self.cancelled})"


@dataclass(frozen=True)
class Watch:
    """A call's cancel token and deadline, checked by the driver at each step."""

    token: CancelToken | None
    # On the `time.monotonic()` clock, so a change to the wall clock cannot move it.
    expires: float | None

    @classmethod
    def begin(cls, token: CancelToken | None, deadline: float | None) -> Watch | None:
        """None when the call has neither, so it runs exactly as one without cancellation."""
        if token is None and deadline is None:
            return None
        return cls(token, None if deadline is None else time.monotonic() + deadline)

    def reached(self) -> bool:
        if self.token is not None and self.token.cancelled:
            return True
        return self.expires is not None and time.monotonic() >= self.expires


def stop(action: steps.Action, handler: EventHandler | None, spend: SpendTracker) -> steps.Stopped:
    """Tell core that `action` was cancelled, then count and deliver what it asks for.

    Plain and synchronous in both modes, so an async call already being cancelled still
    delivers its event without waiting on anything.
    """
    following = advance(action, steps.Cancelled())
    while not isinstance(following, steps.Stopped):
        outcome: steps.Outcome
        if isinstance(following, steps.Emit):
            deliver(following.events, handler)
            outcome = steps.EmitOutcome()
        elif isinstance(following, steps.RecordSpend):
            total, crossed = spend.add(following.amount, following.ceiling)
            outcome = steps.SpendOutcome(total, crossed)
        else:
            raise PrimaGateError(f"A cancelled call went on to {type(following).__name__}")
        following = advance(following, outcome)
    return following


def cancelled() -> CancelledError:
    return CancelledError("The call was cancelled by its cancel token or deadline")


def due(watch: Watch | None, action: steps.Action) -> bool:
    """Whether to stop instead of performing `action`.

    Let through: an action that ends the call, and the announcement of how it ended, since the
    answer or error is already in hand and stopping would hide it; and asking for the call's
    id, which sends nothing, so every cancelled call is reported under its own id.
    """
    if watch is None or isinstance(
        action, steps.Return | steps.Raise | steps.Stopped | steps.NewId
    ):
        return False
    if isinstance(action, steps.Emit) and any(
        isinstance(e, events.CallSucceeded | events.CallFailed) for e in action.events
    ):
        return False
    return watch.reached()
