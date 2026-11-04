import threading

from primagate.drivers.cache import ResponseCache
from primagate.schema.types import ChatCompletion, Choice, Message


def _answer(text: str) -> ChatCompletion:
    message = Message(role="assistant", content=text)
    return ChatCompletion(
        id=text, created=1, model="gpt-4o", choices=[Choice(index=0, message=message)]
    )


def _held(cache: ResponseCache, *keys: str) -> list[str]:
    return [key for key in keys if cache.get(key) is not None]


def test_a_cache_keeps_at_most_its_maximum() -> None:
    cache = ResponseCache(2)
    for key in "abc":
        cache.put(key, _answer(key))
    assert _held(cache, "a", "b", "c") == ["b", "c"]


def test_the_least_recently_used_answer_leaves_first() -> None:
    cache = ResponseCache(2)
    cache.put("a", _answer("a"))
    cache.put("b", _answer("b"))
    assert cache.get("a") is not None
    cache.put("c", _answer("c"))
    assert _held(cache, "a", "b", "c") == ["a", "c"]


def test_storing_a_key_again_counts_as_use_and_does_not_grow_the_cache() -> None:
    cache = ResponseCache(2)
    cache.put("a", _answer("a"))
    cache.put("b", _answer("b"))
    cache.put("a", _answer("a2"))
    cache.put("c", _answer("c"))
    assert _held(cache, "a", "b", "c") == ["a", "c"]
    found = cache.get("a")
    assert found is not None and found.id == "a2"


def test_a_maximum_of_zero_keeps_nothing() -> None:
    cache = ResponseCache(0)
    cache.put("a", _answer("a"))
    assert cache.get("a") is None


def test_the_default_maximum_is_256() -> None:
    cache = ResponseCache()
    for index in range(300):
        cache.put(str(index), _answer("x"))
    assert len(_held(cache, *(str(i) for i in range(300)))) == 256
    assert cache.get("0") is None and cache.get("299") is not None


def test_snapshots_are_copied_in_and_out() -> None:
    cache = ResponseCache(2)
    stored = _answer("a")
    cache.put("a", stored)
    stored.choices[0].message.content = "changed after storing"
    first = cache.get("a")
    assert first is not None and first.choices[0].message.content == "a"
    first.choices[0].message.content = "changed after reading"
    second = cache.get("a")
    assert second is not None and second.choices[0].message.content == "a"


def test_many_threads_never_push_it_over_the_maximum() -> None:
    cache = ResponseCache(5)

    def work(start: int) -> None:
        for index in range(start, start + 200):
            cache.put(str(index), _answer("x"))
            cache.get(str(index - 3))

    threads = [threading.Thread(target=work, args=(n * 1000,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    keys = [str(n * 1000 + i) for n in range(4) for i in range(200)]
    assert len(_held(cache, *keys)) <= 5
