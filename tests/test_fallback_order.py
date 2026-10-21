from types import MappingProxyType

from primagate.core.fallbacks import fallback_candidates
from primagate.schema.wire import NO_FALLBACKS, Fallbacks

A = ("openai", "gpt-4o")
B = ("anthropic", "claude-sonnet-5-5")
C = ("openai", "gpt-4o-mini")
D = ("openai", "gpt-4.1")


def _lists(per_model: dict, default: tuple = ()) -> Fallbacks:  # type: ignore[type-arg]
    return Fallbacks(MappingProxyType(per_model), default)


def test_a_per_model_list_keeps_its_declared_order() -> None:
    assert fallback_candidates(A, _lists({A: (C, B)})) == (C, B)


def test_the_default_applies_to_a_model_with_no_entry() -> None:
    assert fallback_candidates(A, _lists({B: (C,)}, default=(D,))) == (D,)


def test_a_per_model_entry_replaces_the_default() -> None:
    assert fallback_candidates(A, _lists({A: (C,)}, default=(D,))) == (C,)


def test_an_empty_per_model_entry_opts_that_model_out_of_the_default() -> None:
    assert fallback_candidates(A, _lists({A: ()}, default=(D,))) == ()


def test_nothing_declared_means_no_candidates() -> None:
    assert fallback_candidates(A, NO_FALLBACKS) == ()
    assert fallback_candidates(A, _lists({})) == ()


def test_the_primary_and_repeats_are_removed_keeping_the_first() -> None:
    assert fallback_candidates(A, _lists({A: (C, A, B, C, B)})) == (C, B)
    assert fallback_candidates(A, _lists({}, default=(A, D, A))) == (D,)


def test_only_declared_models_ever_appear() -> None:
    declared = {B, C, D}
    result = fallback_candidates(A, _lists({A: (B, C)}, default=(D,)))
    assert set(result) <= declared and D not in result
