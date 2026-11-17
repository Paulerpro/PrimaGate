import threading

import primagate
from primagate.drivers.spend import SpendTracker


def test_the_total_adds_up_and_the_first_pass_over_the_ceiling_is_reported_once() -> None:
    tracker = SpendTracker()
    assert tracker.add(0.4, 1.0) == (0.4, False)
    total, crossed = tracker.add(0.7, 1.0)
    assert crossed and abs(total - 1.1) < 1e-9
    assert tracker.add(5.0, 1.0)[1] is False
    assert tracker.total > 6.0


def test_a_total_equal_to_the_ceiling_is_not_over_it() -> None:
    tracker = SpendTracker()
    assert tracker.add(1.0, 1.0) == (1.0, False)
    assert tracker.add(0.01, 1.0)[1] is True


def test_a_zero_amount_never_crosses_an_unreached_ceiling() -> None:
    assert SpendTracker().add(0.0, 1.0) == (0.0, False)


def test_racing_threads_report_exactly_one_crossing() -> None:
    tracker = SpendTracker()
    start = threading.Barrier(40)
    results: list[bool] = []

    def spend() -> None:
        start.wait()
        results.append(tracker.add(1.0, 10.0)[1])

    threads = [threading.Thread(target=spend) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(True) == 1 and tracker.total == 40.0


def test_nothing_new_is_exported_except_the_event() -> None:
    assert not hasattr(primagate, "SpendTracker")
    assert hasattr(primagate, "SpendingCeilingExceeded")
