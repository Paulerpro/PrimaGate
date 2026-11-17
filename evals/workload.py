"""A fixed, scripted workload for the routing spike.

Every task has an objective check, a rule over the reply text, so no model and no person judges
an answer. The answers are worked out here once and are the same on every run.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Asked of every task so a reply is short enough to check by rule.
SUFFIX = " Reply with only the answer, no explanation."


@dataclass(frozen=True)
class Task:
    id: str
    level: str
    prompt: str
    answer: str
    # "number" checks the last number in the reply, "word" checks for the word on its own.
    kind: str

    def __post_init__(self) -> None:
        # A misspelt kind would otherwise fall through to the word rule without a sound.
        if self.kind not in ("number", "word"):
            raise ValueError(f"unknown kind {self.kind!r}")

    def check(self, reply: str | None) -> bool:
        if not reply:
            return False
        if self.kind == "number":
            found = re.findall(r"-?\d+(?:\.\d+)?", reply.replace(",", ""))
            return bool(found) and float(found[-1]) == float(self.answer)
        # The word must end a short reply and be unnegated, so "not cold" and a list such as
        # "a, b, c" are not read as the answers "cold" and "b".
        words = re.findall(r"[a-z0-9]+", reply.lower())
        return bool(words) and words[-1] == self.answer and "not" not in words and len(words) <= 6


def _task(n: int, level: str, prompt: str, answer: str, kind: str = "number") -> Task:
    return Task(f"{level}-{n:02d}", level, prompt + SUFFIX, answer, kind)


TASKS: tuple[Task, ...] = (
    _task(1, "easy", "What is 17 + 26?", "43"),
    _task(2, "easy", "What is 9 * 8?", "72"),
    _task(3, "easy", "What is the capital of France?", "paris", "word"),
    _task(4, "easy", "What word is the opposite of 'hot'?", "cold", "word"),
    _task(5, "easy", "What is 100 - 37?", "63"),
    _task(6, "easy", "How many days are in a week?", "7"),
    _task(7, "easy", "What is the first letter of the word 'banana'?", "b", "word"),
    _task(8, "easy", "Which is larger, 12 or 21?", "21"),
    _task(9, "easy", "What colour do you get by mixing blue and yellow?", "green", "word"),
    _task(10, "easy", "What is 144 / 12?", "12"),
    _task(11, "easy", "Spell 'cat' backwards.", "tac", "word"),
    _task(12, "easy", "How many letters are in the word 'house'?", "5"),
    _task(13, "easy", "What is the plural of 'mouse'?", "mice", "word"),
    _task(14, "easy", "What is 5 squared?", "25"),
    _task(
        15, "medium", "A shop sells pens at 3 dollars each. How many dollars do 14 pens cost?", "42"
    ),
    _task(16, "medium", "What is 15% of 240?", "36"),
    _task(17, "medium", "Convert 2.5 hours to minutes.", "150"),
    _task(18, "medium", "A train travels 60 km/h for 3.5 hours. How many km does it go?", "210"),
    _task(19, "medium", "What is (8 + 4) * (7 - 3)?", "48"),
    _task(20, "medium", "How many vowels (a, e, i, o, u) are in the word 'education'?", "5"),
    _task(
        21,
        "medium",
        "If today is Wednesday, what day of the week is it 10 days from now?",
        "saturday",
        "word",
    ),
    _task(22, "medium", "What is the sum of the digits of 98765?", "35"),
    _task(23, "medium", "A rectangle is 12 m by 7 m. What is its perimeter in metres?", "38"),
    _task(24, "medium", "What is the next number in the sequence 2, 6, 12, 20, 30?", "42"),
    _task(
        25,
        "hard",
        "A bat and a ball cost 1.10 dollars together. The bat costs 1.00 dollar more than the "
        "ball. How many cents does the ball cost?",
        "5",
    ),
    _task(
        26,
        "hard",
        "I have 3 boxes. Each box holds 4 smaller boxes, and each smaller box holds 5 pencils. "
        "How many pencils are there in total?",
        "60",
    ),
    _task(
        27,
        "hard",
        "Tom is twice as old as Ann. In 6 years the sum of their ages will be 42. How old is "
        "Tom now?",
        "20",
    ),
    _task(
        28,
        "hard",
        "What is the smallest positive whole number divisible by 2, 3, 4, 5 and 6?",
        "60",
    ),
    _task(29, "hard", "How many 3-digit numbers are divisible by 7?", "128"),
    _task(
        30,
        "hard",
        "One pipe fills a tank in 6 hours and another fills it in 3 hours. Working together, "
        "how many hours do they take?",
        "2",
    ),
)
