import threading

import pytest

import primagate
from primagate.drivers import capacity
from primagate.drivers.capacity import DEFAULT_MAX_CONCURRENT_ANALYZER_CALLS, AnalyzerCap


def test_permits_are_given_up_to_the_limit_and_never_waited_for() -> None:
    cap = AnalyzerCap(2)
    assert [cap.try_acquire() for _ in range(3)] == [True, True, False]
    assert cap.in_flight == 2 and cap.limit == 2


def test_a_released_permit_can_be_taken_again() -> None:
    cap = AnalyzerCap(1)
    assert cap.try_acquire() and not cap.try_acquire()
    cap.release()
    assert cap.in_flight == 0 and cap.try_acquire()


def test_releasing_with_nothing_held_raises() -> None:
    with pytest.raises(RuntimeError):
        AnalyzerCap(1).release()


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, None, "2"])
def test_a_bad_limit_is_refused(limit: object) -> None:
    with pytest.raises(ValueError):
        AnalyzerCap(limit)  # type: ignore[arg-type]


def test_racing_threads_never_exceed_the_limit() -> None:
    cap = AnalyzerCap(5)
    start = threading.Barrier(50)
    won: list[bool] = []

    def grab() -> None:
        start.wait()
        won.append(cap.try_acquire())

    threads = [threading.Thread(target=grab) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(won) == 5 and cap.in_flight == 5


def test_the_default_is_one_named_constant_and_nothing_is_exported() -> None:
    assert DEFAULT_MAX_CONCURRENT_ANALYZER_CALLS >= 1
    assert capacity.DEFAULT_MAX_CONCURRENT_ANALYZER_CALLS == DEFAULT_MAX_CONCURRENT_ANALYZER_CALLS
    assert not hasattr(primagate, "AnalyzerCap")
