from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any, Final

from primagate.core.counting import heuristic_count
from primagate.core.steps import Action, StoreRead, StoreWrite
from primagate.schema.profile import (
    PREPARATION_OVERFLOW,
    PROFILE_SCHEMA_VERSION,
    InvalidProfileError,
    PreparationOverflow,
    Profile,
    ProfileContent,
    ProfileSources,
)

_PROFILE_FIELDS = (
    "profile_id",
    "input_fingerprint",
    "schema_version",
    "created_at",
    "content",
    "source_ref",
)
_CONTENT_TEXT = ("purpose",)
_CONTENT_LISTS = ("interactions", "constraints", "capabilities", "unknowns")
PLACEHOLDER_UNKNOWNS = ("purpose", "interactions", "constraints", "capabilities")
# Names the prompt that turns sources into a Profile. It is part of the fingerprint, so changing
# that prompt means bumping this, or Profiles prepared by the old prompt would still be served.
PREP_PROMPT_VERSION: Final = "prep-1"


def input_fingerprint(
    sources: ProfileSources,
    prep_prompt_version: str,
    analyzer_model_version: str,
    schema_version: int = PROFILE_SCHEMA_VERSION,
) -> str:
    """A stable hash of everything a Profile is prepared from, and nothing else.

    Pricing and Tier Bindings are not parameters, so changing them cannot make a Profile stale.
    The lists are sorted because the order sources were declared in carries no meaning, and
    the JSON keeps each field apart so text cannot slide from one field into another.
    """
    for label, version in (
        ("prep_prompt_version", prep_prompt_version),
        ("analyzer_model_version", analyzer_model_version),
    ):
        # An empty version would give a fingerprint that never moves when the prompt or model does.
        if not isinstance(version, str) or not version:
            raise ValueError(f"{label} must be non-empty text")
    canonical = {
        "instructions": sources.instructions,
        "tool_descriptions": sorted(sources.tool_descriptions),
        "skill_descriptions": sorted(sources.skill_descriptions),
        "prep_prompt_version": prep_prompt_version,
        "analyzer_model_version": analyzer_model_version,
        "schema_version": schema_version,
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def usable_profile(
    stored: Profile | None, fingerprint: str, schema_version: int = PROFILE_SCHEMA_VERSION
) -> Profile | None:
    """The stored Profile if it still matches the inputs and shape in force, else a miss."""
    if stored is None:
        return None
    if stored.input_fingerprint != fingerprint or stored.schema_version != schema_version:
        return None
    return stored


def read_profile(fingerprint: str, then: Callable[[Profile | None], Action]) -> StoreRead:
    """Ask for the stored Profile and hand `then` it, or None when it is missing or no longer fits.

    The fit is judged here rather than in the store, so a stale or old-schema Profile is a miss
    whichever store holds it.
    """
    return StoreRead(lambda read: then(usable_profile(read.profile, fingerprint)))


def write_profile(profile: Profile, then: Action) -> StoreWrite:
    """Store `profile` in place of whatever the store held, then go on with `then`."""
    return StoreWrite(profile, then)


def check_sources(sources: ProfileSources, limit: int) -> str | PreparationOverflow:
    """The text to prepare from, or exactly `PREPARATION_OVERFLOW` when it is too large.

    `limit` is in tokens by the built-in estimate. It is the caller's, since it depends on the
    Analyzer's context window. Sources with nothing in them give empty text, which is not overflow.
    The descriptions are sorted as the fingerprint sorts them, so equal fingerprints always give
    the Analyzer equal text.
    """
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a whole number of 1 or more")
    parts = [
        sources.instructions,
        *sorted(sources.tool_descriptions),
        *sorted(sources.skill_descriptions),
    ]
    text = "\n".join(part for part in parts if part)
    return PREPARATION_OVERFLOW if heuristic_count(text) > limit else text


def empty_profile(
    profile_id: str, created_at: str, fingerprint: str, source_ref: str = "none"
) -> Profile:
    """A Profile with nothing prepared yet, saying so by naming everything it does not know.

    Its arguments are held to the same rules as a stored Profile, so one that could not be read
    back is never built.
    """
    _non_empty("profile_id", profile_id)
    _non_empty("input_fingerprint", fingerprint)
    _timestamp(created_at)
    return Profile(
        profile_id=profile_id,
        input_fingerprint=fingerprint,
        schema_version=PROFILE_SCHEMA_VERSION,
        created_at=created_at,
        content=ProfileContent(unknowns=PLACEHOLDER_UNKNOWNS),
        source_ref=source_ref,
    )


def parse_profile(raw: object) -> Profile:
    """Read a stored mapping as a Profile, or raise `InvalidProfileError` naming the field.

    An unrecognised key is counted, not named, since a key could be a secret.
    """
    if not isinstance(raw, Mapping):
        raise InvalidProfileError("profile must be a mapping")
    extra = [k for k in raw if k not in _PROFILE_FIELDS]
    if extra:
        raise InvalidProfileError(f"profile has {len(extra)} unrecognised field(s)")
    for name in _PROFILE_FIELDS:
        if name not in raw:
            raise InvalidProfileError(f"profile is missing {name}")
    for name in ("profile_id", "input_fingerprint"):
        _non_empty(name, raw[name])
    if not isinstance(raw["source_ref"], str):
        raise InvalidProfileError("profile source_ref must be text")
    version = raw["schema_version"]
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise InvalidProfileError("profile schema_version must be a whole number of 1 or more")
    return Profile(
        profile_id=raw["profile_id"],
        input_fingerprint=raw["input_fingerprint"],
        schema_version=version,
        created_at=_timestamp(raw["created_at"]),
        content=parse_content(raw["content"]),
        source_ref=raw["source_ref"],
    )


def parse_content(raw: object) -> ProfileContent:
    """Read a mapping as Profile content, or raise `InvalidProfileError` naming the field.

    One set of rules for stored and freshly prepared content, so preparation can never write a
    Profile that a store would refuse to read back.
    """
    return _content(raw)


def _non_empty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise InvalidProfileError(f"profile {name} must be non-empty text")


def _timestamp(value: object) -> str:
    if not isinstance(value, str):
        raise InvalidProfileError("profile created_at must be an ISO 8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise InvalidProfileError("profile created_at must be an ISO 8601 UTC timestamp") from None
    # A timestamp with no zone could be anyone's local time, so it is not accepted as UTC.
    if parsed.utcoffset() != timedelta(0):
        raise InvalidProfileError("profile created_at must be an ISO 8601 UTC timestamp")
    return value


def _content(raw: object) -> ProfileContent:
    if not isinstance(raw, Mapping):
        raise InvalidProfileError("profile content must be a mapping")
    extra = [k for k in raw if k not in (*_CONTENT_TEXT, *_CONTENT_LISTS)]
    if extra:
        raise InvalidProfileError(f"profile content has {len(extra)} unrecognised field(s)")
    # Required, not defaulted: a missing list of gaps would otherwise read as "no gaps".
    if "unknowns" not in raw:
        raise InvalidProfileError("profile content is missing unknowns")
    parts: dict[str, Any] = {}
    purpose = raw.get("purpose", "")
    if not isinstance(purpose, str):
        raise InvalidProfileError("profile content purpose must be text")
    parts["purpose"] = purpose
    for name in _CONTENT_LISTS:
        parts[name] = _texts(name, raw.get(name, []))
    return ProfileContent(**parts)


def _texts(name: str, value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list | tuple)
        or isinstance(value, str)
        or not all(isinstance(v, str) and v.strip() for v in value)
    ):
        raise InvalidProfileError(f"profile content {name} must be a list of non-blank text")
    return tuple(value)
