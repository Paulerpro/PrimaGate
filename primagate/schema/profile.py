from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

# Bumped whenever the shape below changes, so a stored Profile of an older shape is never reused.
PROFILE_SCHEMA_VERSION: Final = 1


class InvalidProfileError(ValueError):
    """A Profile that does not meet the contract; a store treats it as a miss, not a failure."""


class PreparationOverflow:
    """The one result for sources too large to prepare.

    A distinct type, so it can never be mistaken for the prepared text it stands in for. Compare
    with `is PREPARATION_OVERFLOW`, never by truthiness: it is truthy, and empty text is not.
    """

    _instance: PreparationOverflow | None = None

    def __new__(cls) -> PreparationOverflow:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "PREPARATION_OVERFLOW"

    # Copies and pickles must come back as the same object, or `is PREPARATION_OVERFLOW` breaks.
    def __reduce__(self) -> str:
        return "PREPARATION_OVERFLOW"

    def __copy__(self) -> PreparationOverflow:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> PreparationOverflow:
        return self


PREPARATION_OVERFLOW: Final = PreparationOverflow()


class PrepState(Enum):
    """Where preparation stands for one fingerprint. Values only: nothing persists them."""

    PREPARING = "preparing"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True)
class ProfileSources:
    """What the Profile is prepared from; changing any of it makes the Profile stale."""

    instructions: str = ""
    tool_descriptions: tuple[str, ...] = ()
    skill_descriptions: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ProfileContent:
    """The app as the Analyzer should understand it; gaps are named, never guessed."""

    # No default: leaving it out must fail, so a gap can never be hidden by omission.
    unknowns: tuple[str, ...]
    purpose: str = ""
    interactions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Enforced here, not only when parsing, so no Profile can claim an empty app with no gaps.
        if self.is_empty and not any(u.strip() for u in self.unknowns):
            raise InvalidProfileError(
                "profile content is empty so unknowns must name what is not known"
            )

    @property
    def is_empty(self) -> bool:
        # Blank text is not content, so a purpose of spaces cannot stand in for a prepared one.
        has_text = bool(self.purpose.strip())
        return not (has_text or self.interactions or self.constraints or self.capabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "interactions": list(self.interactions),
            "constraints": list(self.constraints),
            "capabilities": list(self.capabilities),
            "unknowns": list(self.unknowns),
        }


@dataclass(frozen=True)
class Profile:
    profile_id: str
    input_fingerprint: str
    schema_version: int
    # ISO 8601 UTC, handed in by the driver since core reads no clock.
    created_at: str
    content: ProfileContent
    # Opaque: says where the sources came from, for whoever reads the stored Profile.
    source_ref: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "input_fingerprint": self.input_fingerprint,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "content": self.content.to_dict(),
            "source_ref": self.source_ref,
        }
