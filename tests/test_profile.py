import copy
import dataclasses
import inspect
import pickle
import re
from typing import Any

import pytest

import primagate
from primagate.core import profile as core_profile
from primagate.core.profile import (
    check_sources,
    empty_profile,
    input_fingerprint,
    parse_profile,
    usable_profile,
)
from primagate.schema.profile import (
    PREPARATION_OVERFLOW,
    PROFILE_SCHEMA_VERSION,
    InvalidProfileError,
    PreparationOverflow,
    Profile,
    ProfileContent,
    ProfileSources,
)

SOURCES = ProfileSources(
    instructions="You are a support agent for Acme.",
    tool_descriptions=("lookup_order: find an order", "refund: refund an order"),
    skill_descriptions=("tone: be polite",),
)
FP = input_fingerprint(SOURCES, "prep-1", "gpt-4o-mini")
CONTENT = ProfileContent(
    purpose="Answers customer questions about orders.",
    interactions=("looks up orders",),
    constraints=("never refunds over 100",),
    capabilities=("order lookup",),
    unknowns=(),
)
PROFILE = Profile(
    profile_id="profile-1",
    input_fingerprint=FP,
    schema_version=PROFILE_SCHEMA_VERSION,
    created_at="2026-11-16T10:00:00Z",
    content=CONTENT,
    source_ref="setup",
)


def _fp(
    sources: ProfileSources = SOURCES,
    prompt: str = "prep-1",
    model: str = "gpt-4o-mini",
    schema: int = PROFILE_SCHEMA_VERSION,
) -> str:
    return input_fingerprint(sources, prompt, model, schema)


def test_a_profile_carries_the_six_fields_and_content_carries_unknowns() -> None:
    names = [f.name for f in dataclasses.fields(Profile)]
    assert names == [
        "profile_id",
        "input_fingerprint",
        "schema_version",
        "created_at",
        "content",
        "source_ref",
    ]
    assert "unknowns" in {f.name for f in dataclasses.fields(ProfileContent)}
    assert PROFILE_SCHEMA_VERSION == 1


def test_unknowns_has_no_default_so_it_cannot_be_left_out() -> None:
    with pytest.raises(TypeError):
        ProfileContent(purpose="x")  # type: ignore[call-arg]


def test_a_profile_is_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        PROFILE.profile_id = "other"  # type: ignore[misc]


def test_to_dict_and_parse_round_trip_exactly() -> None:
    again = parse_profile(copy.deepcopy(PROFILE.to_dict()))
    assert again == PROFILE
    named = dataclasses.replace(PROFILE, content=dataclasses.replace(CONTENT, unknowns=("a", "b")))
    assert parse_profile(named.to_dict()) == named


def _raw(**changes: Any) -> dict[str, Any]:
    raw = copy.deepcopy(PROFILE.to_dict())
    for key, value in changes.items():
        raw[key] = value
    return raw


def test_a_profile_without_unknowns_is_rejected_naming_the_field() -> None:
    raw = _raw()
    del raw["content"]["unknowns"]
    with pytest.raises(InvalidProfileError, match="unknowns"):
        parse_profile(raw)


@pytest.mark.parametrize("bad", [None, "none", ["a", 1], 5, {"a": 1}])
def test_unknowns_must_be_a_list_of_text(bad: Any) -> None:
    raw = _raw()
    raw["content"]["unknowns"] = bad
    with pytest.raises(InvalidProfileError, match="unknowns"):
        parse_profile(raw)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("profile_id", ""),
        ("profile_id", 5),
        ("input_fingerprint", None),
        ("schema_version", "1"),
        ("schema_version", True),
        ("schema_version", 0),
        ("created_at", "yesterday"),
        ("created_at", 5),
        ("created_at", "2026-11-16T10:00:00"),
        ("source_ref", None),
        ("content", []),
        ("content", None),
    ],
)
def test_a_bad_field_is_rejected_naming_it(field: str, bad: Any) -> None:
    with pytest.raises(InvalidProfileError, match=field):
        parse_profile(_raw(**{field: bad}))


@pytest.mark.parametrize(
    "field",
    ["profile_id", "input_fingerprint", "schema_version", "created_at", "content", "source_ref"],
)
def test_a_missing_field_is_rejected_naming_it(field: str) -> None:
    raw = _raw()
    del raw[field]
    with pytest.raises(InvalidProfileError, match=field):
        parse_profile(raw)


@pytest.mark.parametrize("field", ["purpose", "interactions", "constraints", "capabilities"])
def test_a_bad_content_field_is_rejected(field: str) -> None:
    raw = _raw()
    raw["content"][field] = 5
    with pytest.raises(InvalidProfileError, match=field):
        parse_profile(raw)


def test_the_input_must_be_a_mapping() -> None:
    for bad in (None, [], "profile", 5):
        with pytest.raises(InvalidProfileError):
            parse_profile(bad)  # type: ignore[arg-type]


def test_an_unknown_extra_field_is_rejected() -> None:
    with pytest.raises(InvalidProfileError):
        parse_profile(_raw(surprise=1))


def test_wholly_empty_content_must_name_its_unknowns_even_in_memory() -> None:
    empty = {
        "purpose": "",
        "interactions": [],
        "constraints": [],
        "capabilities": [],
        "unknowns": [],
    }
    with pytest.raises(InvalidProfileError, match="unknowns"):
        parse_profile(_raw(content=empty))
    with pytest.raises(InvalidProfileError, match="unknowns"):
        ProfileContent(unknowns=())
    with pytest.raises(InvalidProfileError, match="unknowns"):
        ProfileContent(unknowns=(), purpose="   ")
    named = ProfileContent(unknowns=("purpose",))
    assert parse_profile(_raw(content=named.to_dict())).content == named


@pytest.mark.parametrize("field", ["interactions", "constraints", "capabilities", "unknowns"])
def test_blank_list_entries_are_rejected(field: str) -> None:
    raw = _raw()
    raw["content"][field] = ["fine", "  "]
    with pytest.raises(InvalidProfileError, match=field):
        parse_profile(raw)


def test_an_unrecognised_key_inside_content_is_rejected_without_naming_it() -> None:
    raw = _raw()
    raw["content"]["from_a_newer_shape"] = ["x"]
    with pytest.raises(InvalidProfileError, match="unrecognised") as caught:
        parse_profile(raw)
    assert "from_a_newer_shape" not in str(caught.value)


def test_a_timestamp_may_use_z_or_an_explicit_zero_offset_but_not_another_zone() -> None:
    parse_profile(_raw(created_at="2026-11-16T10:00:00+00:00"))
    parse_profile(_raw(created_at="2026-11-16T10:00:00.250Z"))
    with pytest.raises(InvalidProfileError, match="created_at"):
        parse_profile(_raw(created_at="2026-11-16T10:00:00+02:00"))


def test_lists_may_come_back_as_tuples() -> None:
    raw = _raw()
    raw["content"]["interactions"] = ("looks up orders",)
    assert parse_profile(raw).content.interactions == ("looks up orders",)


def test_prepared_content_may_have_no_gaps() -> None:
    assert parse_profile(PROFILE.to_dict()).content.unknowns == ()


def test_the_fingerprint_is_stable_for_equal_inputs() -> None:
    assert _fp() == _fp()
    again = ProfileSources(
        SOURCES.instructions, SOURCES.tool_descriptions, SOURCES.skill_descriptions
    )
    assert _fp(again) == _fp()
    assert len(_fp()) >= 16 and _fp() == FP


def test_reordering_tools_or_skills_does_not_change_the_fingerprint() -> None:
    two_skills = ProfileSources("i", ("a", "b"), ("tone: polite", "length: short"))
    flipped = ProfileSources("i", ("b", "a"), ("length: short", "tone: polite"))
    assert _fp(flipped) == _fp(two_skills)
    only_tools = ProfileSources("i", ("b", "a"), ("tone: polite", "length: short"))
    only_skills = ProfileSources("i", ("a", "b"), ("length: short", "tone: polite"))
    assert _fp(only_tools) == _fp(two_skills) == _fp(only_skills)


@pytest.mark.parametrize("version", ["", None, 5])
def test_an_empty_or_non_text_version_is_refused(version: Any) -> None:
    with pytest.raises(ValueError, match="version"):
        input_fingerprint(SOURCES, version, "gpt-4o-mini")
    with pytest.raises(ValueError, match="version"):
        input_fingerprint(SOURCES, "prep-1", version)


def test_equal_fingerprints_give_the_analyzer_equal_text() -> None:
    a = ProfileSources("i", ("b", "a"), ("y", "x"))
    b = ProfileSources("i", ("a", "b"), ("x", "y"))
    assert _fp(a) == _fp(b) and check_sources(a, 1000) == check_sources(b, 1000)


def test_sources_with_nothing_in_them_are_empty_text_not_overflow() -> None:
    assert check_sources(ProfileSources(), 1) == ""


@pytest.mark.parametrize(
    "changed",
    [
        ProfileSources(
            "Different instructions.", SOURCES.tool_descriptions, SOURCES.skill_descriptions
        ),
        ProfileSources(
            SOURCES.instructions, ("lookup_order: find an order",), SOURCES.skill_descriptions
        ),
        ProfileSources(
            SOURCES.instructions,
            ("lookup_order: find orders", "refund: refund an order"),
            SOURCES.skill_descriptions,
        ),
        ProfileSources(SOURCES.instructions, SOURCES.tool_descriptions, ()),
        ProfileSources(SOURCES.instructions, SOURCES.tool_descriptions, ("tone: be rude",)),
    ],
)
def test_a_changed_source_changes_the_fingerprint(changed: ProfileSources) -> None:
    assert _fp(changed) != _fp()


def test_a_changed_prompt_version_analyzer_model_or_schema_changes_it() -> None:
    assert _fp(prompt="prep-2") != _fp()
    assert _fp(model="gpt-4o") != _fp()
    assert _fp(schema=2) != _fp()


def test_a_tool_description_cannot_be_confused_with_a_skill_description() -> None:
    as_tool = ProfileSources("i", ("x",), ())
    as_skill = ProfileSources("i", (), ("x",))
    assert _fp(as_tool) != _fp(as_skill)


def test_field_boundaries_cannot_be_shifted_to_collide() -> None:
    assert _fp(ProfileSources("ab", ("c",), ())) != _fp(ProfileSources("a", ("bc",), ()))


def test_pricing_and_bindings_have_no_way_into_the_fingerprint() -> None:
    names = list(inspect.signature(input_fingerprint).parameters)
    assert names == ["sources", "prep_prompt_version", "analyzer_model_version", "schema_version"]


def test_a_stored_profile_with_the_same_fingerprint_and_schema_is_used() -> None:
    assert usable_profile(PROFILE, FP) is PROFILE


def test_a_different_fingerprint_or_schema_or_nothing_is_a_miss() -> None:
    assert usable_profile(PROFILE, _fp(prompt="prep-2")) is None
    old = dataclasses.replace(PROFILE, schema_version=PROFILE_SCHEMA_VERSION + 1)
    assert usable_profile(old, FP) is None
    assert usable_profile(PROFILE, FP, schema_version=PROFILE_SCHEMA_VERSION + 1) is None
    assert usable_profile(None, FP) is None


def test_a_source_over_the_limit_is_exactly_the_overflow_marker() -> None:
    big = ProfileSources("x" * 4000, (), ())
    assert check_sources(big, 500) is PREPARATION_OVERFLOW


def test_a_source_within_the_limit_gives_its_text_for_preparation() -> None:
    text = check_sources(SOURCES, 10_000)
    assert isinstance(text, str)
    for part in (*SOURCES.tool_descriptions, SOURCES.instructions, *SOURCES.skill_descriptions):
        assert part in text


def test_the_limit_is_inclusive() -> None:
    text = check_sources(ProfileSources("abcd" * 10, (), ()), 10_000)
    assert isinstance(text, str)
    tokens = core_profile.heuristic_count(text)
    assert check_sources(ProfileSources("abcd" * 10, (), ()), tokens) == text
    assert check_sources(ProfileSources("abcd" * 10, (), ()), tokens - 1) is PREPARATION_OVERFLOW


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, None])
def test_a_bad_limit_is_refused(limit: Any) -> None:
    with pytest.raises(ValueError):
        check_sources(SOURCES, limit)


def test_the_overflow_marker_is_one_object_and_never_equal_to_text() -> None:
    assert PreparationOverflow() is PREPARATION_OVERFLOW
    assert PREPARATION_OVERFLOW != "PREPARATION_OVERFLOW"
    assert copy.deepcopy(PREPARATION_OVERFLOW) is PREPARATION_OVERFLOW
    assert copy.copy(PREPARATION_OVERFLOW) is PREPARATION_OVERFLOW
    assert pickle.loads(pickle.dumps(PREPARATION_OVERFLOW)) is PREPARATION_OVERFLOW
    assert bool(PREPARATION_OVERFLOW) is True
    assert "PREPARATION_OVERFLOW" in repr(PREPARATION_OVERFLOW)


def test_the_placeholder_profile_is_valid_empty_and_names_its_unknowns() -> None:
    placeholder = empty_profile("profile-0", "2026-11-16T10:00:00Z", FP)
    assert placeholder.content.purpose == "" and placeholder.content.interactions == ()
    assert placeholder.content.unknowns == (
        "purpose",
        "interactions",
        "constraints",
        "capabilities",
    )
    assert (
        placeholder.input_fingerprint == FP and placeholder.schema_version == PROFILE_SCHEMA_VERSION
    )
    assert parse_profile(placeholder.to_dict()) == placeholder
    assert usable_profile(placeholder, FP) is placeholder


@pytest.mark.parametrize(
    "args",
    [
        ("", "2026-11-16T10:00:00Z", FP),
        ("p", "2026-11-16T10:00:00Z", ""),
        ("p", "yesterday", FP),
        ("p", "2026-11-16T10:00:00", FP),
    ],
)
def test_a_placeholder_that_could_not_be_read_back_is_never_built(
    args: tuple[str, str, str],
) -> None:
    with pytest.raises(InvalidProfileError):
        empty_profile(*args)


def test_core_makes_no_id_and_reads_no_clock() -> None:
    source = inspect.getsource(core_profile)
    for banned in (r"uuid", r"\.now\(", r"utcnow", r"time\.", r"random", r"secrets"):
        assert not re.search(banned, source), banned


def test_the_profile_adds_no_public_name() -> None:
    for name in ("Profile", "ProfileSources", "PREPARATION_OVERFLOW", "ProfileContent"):
        assert not hasattr(primagate, name)
