from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from primagate.core import steps
from primagate.core.counting import heuristic_count

logger = logging.getLogger("primagate")

TokenCounter = Callable[[str, str], int]

# Used when tiktoken does not know a model, as is the case for every non-OpenAI model.
_FALLBACK_ENCODING = "o200k_base"


class TokenCounting:
    """Counts tokens for responses that reported no usage. It never raises.

    The caller's counter wins. Otherwise tiktoken is tried, imported only here and only on first
    use because it needs `requests` and may download a vocabulary; any failure to import or load
    it is remembered so a broken setup costs one attempt, not one per call. The heuristic is the
    floor under both.
    """

    def __init__(self, counter: TokenCounter | None = None) -> None:
        self._counter = counter
        self._lock = threading.Lock()
        self._tiktoken_unavailable = False
        self._encodings: dict[str, Any] = {}

    def count(self, model: str, prompt: str, completion: str) -> steps.CountOutcome:
        if self._counter is not None:
            return self._with_custom(self._counter, model, prompt, completion)
        if not self._tiktoken_unavailable:
            try:
                encoding = self._encoding(model)
                return steps.CountOutcome(
                    len(encoding.encode(prompt, disallowed_special=())),
                    len(encoding.encode(completion, disallowed_special=())),
                    heuristic=False,
                )
            except ImportError:
                # Not installing the optional extra is a normal setup, so it is not worth a warning.
                self._tiktoken_unavailable = True
            except Exception as err:
                self._tiktoken_unavailable = True
                logger.warning(
                    "tiktoken is not usable (%s); token counts use the built-in heuristic",
                    type(err).__name__,
                )
        return _heuristic(prompt, completion)

    def _with_custom(
        self, counter: TokenCounter, model: str, prompt: str, completion: str
    ) -> steps.CountOutcome:
        try:
            return steps.CountOutcome(
                _valid(counter(model, prompt)), _valid(counter(model, completion)), heuristic=False
            )
        except Exception as err:
            logger.warning(
                "token_counter failed (%s); this call's token counts use the built-in heuristic",
                type(err).__name__,
            )
            return _heuristic(prompt, completion)

    def _encoding(self, model: str) -> Any:
        with self._lock:
            cached = self._encodings.get(model)
            if cached is not None:
                return cached
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding(_FALLBACK_ENCODING)
            self._encodings[model] = encoding
            return encoding


def _valid(value: object) -> int:
    # bool is an int subclass, and a counter returning True is a bug worth catching.
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("a token count must be an integer of 0 or more")
    return value


def _heuristic(prompt: str, completion: str) -> steps.CountOutcome:
    return steps.CountOutcome(heuristic_count(prompt), heuristic_count(completion), heuristic=True)
