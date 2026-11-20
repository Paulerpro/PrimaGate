from __future__ import annotations

from typing import Any

from primagate.schema.errors import BadRequestError
from primagate.schema.wire import Capabilities

DEFAULT_PROVIDER = "openai"
KNOWN_PROVIDERS = frozenset({"openai", "anthropic"})


def resolve_model(name: str) -> tuple[str, str]:
    """Split a model string into (provider, model).

    A bare `claude-*` name is Anthropic's; any other bare name is the default provider's, so
    OpenAI-compatible servers with unusual model names keep working.
    """
    prefix, sep, rest = name.partition("/")
    if not sep:
        return ("anthropic" if name.startswith("claude-") else DEFAULT_PROVIDER), name
    if prefix in KNOWN_PROVIDERS:
        return prefix, rest
    raise BadRequestError(f"Unknown provider prefix '{prefix}' in model '{name}'")


def short_answer_settings(declared: Capabilities, model: str) -> dict[str, Any]:
    """Request fields for the Analyzer's and Profile preparation's small, repeatable answers.

    `temperature` 0 keeps them repeatable, unless the model rejects the setting. A model that
    reasons by default would spend the few answer tokens on that reasoning, so it is told not to.
    """
    return {
        "temperature": None if "temperature" in declared.hints_for(model) else 0,
        "disable_thinking": declared.thinks_by_default(model) or None,
    }
