"""A fixed, scripted workload for the routing evaluation.

Every task has an objective check, a rule over the reply or the tool call, so no model and no
person judges an answer. The answers are worked out here once and are the same on every run.

Three types of task: `oneshot` (a word or number answer), `tool` (a tool is offered and the check
is the tool name and its arguments) and `multiturn` (earlier exchanges, then a question whose
answer depends on them).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Asked of every text task so a reply is short enough to check by rule.
SUFFIX = " Reply with only the answer, no explanation."

DIFFICULTIES = ("easy", "medium", "hard")
TYPES = ("oneshot", "tool", "multiturn")
KINDS = ("number", "word", "tool")


def _tool(name: str, description: str, **properties: str) -> dict[str, Any]:
    """A function tool whose every argument is required except one named `unit`."""
    props: dict[str, Any] = {}
    for key, kind in properties.items():
        props[key] = {"type": kind}
    if "unit" in props:
        props["unit"] = {"type": "string", "enum": ["celsius", "fahrenheit"]}
    required = [k for k in props if k != "unit"]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


TOOL_SPECS: dict[str, dict[str, Any]] = {
    spec["function"]["name"]: spec
    for spec in (
        _tool("get_weather", "Current weather for a city.", city="string", unit="string"),
        _tool("set_timer", "Start a countdown timer.", minutes="integer"),
        _tool("get_stock_price", "Latest price for a stock ticker symbol.", symbol="string"),
        _tool(
            "convert_currency",
            "Convert an amount; currencies are three-letter ISO 4217 codes such as USD.",
            amount="number",
            from_currency="string",
            to_currency="string",
        ),
        _tool(
            "translate_text",
            "Translate text into a language.",
            text="string",
            target_language="string",
        ),
        _tool("send_email", "Send an email.", to="string", subject="string"),
        _tool(
            "create_event",
            "Add a calendar event; the date is YYYY-MM-DD and the time is HH:MM.",
            title="string",
            date="string",
            time="string",
        ),
        _tool(
            "search_flights",
            "Search flights; the date is YYYY-MM-DD.",
            origin="string",
            destination="string",
            date="string",
        ),
    )
}


def _same(expected: Any, given: Any) -> bool:
    if isinstance(expected, bool) or isinstance(given, bool):
        return expected is given
    if isinstance(expected, int | float):
        # A number must arrive as a number: "10" for 10 is a different argument.
        if not isinstance(given, int | float):
            return False
        try:
            return float(expected) == float(given)
        except OverflowError:
            # An integer too large for a float cannot be the small value that was asked for.
            return False
    if isinstance(expected, str):
        return isinstance(given, str) and given.strip().lower() == expected.lower()
    return bool(expected == given)


@dataclass(frozen=True)
class Task:
    id: str
    difficulty: str
    type: str
    # The final user message.
    prompt: str
    answer: str = ""
    # "number" checks the last number in the reply, "word" checks for the word on its own, and
    # "tool" checks the tool call.
    kind: str = "number"
    # Earlier (user, assistant) exchanges of a multi-turn conversation.
    history: tuple[tuple[str, str], ...] = ()
    # The tools offered, by name, and the call that passes.
    tools: tuple[str, ...] = ()
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A misspelt value would otherwise fall through to a different rule without a sound.
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.type not in TYPES:
            raise ValueError(f"unknown type {self.type!r}")
        if self.difficulty not in DIFFICULTIES:
            raise ValueError(f"unknown difficulty {self.difficulty!r}")
        if self.type == "multiturn" and not self.history:
            raise ValueError("a multi-turn task needs earlier exchanges")
        if self.type != "multiturn" and self.history:
            raise ValueError("only a multi-turn task has earlier exchanges")
        if (self.type == "tool") != (self.kind == "tool"):
            raise ValueError("a task is a tool task exactly when its kind is 'tool'")
        if self.type == "tool":
            unknown = [t for t in self.tools if t not in TOOL_SPECS]
            if not self.tool or self.tool not in self.tools or unknown or not self.args:
                raise ValueError("a tool task offers its tool and expects its arguments")

    def messages(self) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for question, reply in self.history:
            out += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
        return [*out, {"role": "user", "content": self.prompt}]

    def tool_specs(self) -> list[dict[str, Any]] | None:
        return [TOOL_SPECS[name] for name in self.tools] if self.type == "tool" else None

    def check(self, reply: str | None, call: tuple[str, str] | None = None) -> bool:
        """Whether the reply, or for a tool task the (name, JSON arguments) call, passes."""
        if self.kind == "tool":
            return self._check_call(call)
        if not reply:
            return False
        if self.kind == "number":
            found = re.findall(r"-?\d+(?:\.\d+)?", reply.replace(",", ""))
            return bool(found) and float(found[-1]) == float(self.answer)
        # The word must end a short reply and be unnegated, so "not cold" and a list such as
        # "a, b, c" are not read as the answers "cold" and "b".
        words = re.findall(r"[a-z0-9]+", reply.lower())
        return bool(words) and words[-1] == self.answer and "not" not in words and len(words) <= 6

    def _check_call(self, call: tuple[str, str] | None) -> bool:
        if call is None or call[0] != self.tool:
            return False
        try:
            given = json.loads(call[1])
        except ValueError:
            return False
        if not isinstance(given, dict):
            return False
        return all(k in given and _same(v, given[k]) for k, v in self.args.items())


def _task(n: int, difficulty: str, prompt: str, answer: str, kind: str = "number") -> Task:
    return Task(
        f"oneshot-{difficulty}-{n:02d}", difficulty, "oneshot", prompt + SUFFIX, answer, kind
    )


def _use(
    n: int,
    difficulty: str,
    prompt: str,
    tool: str,
    args: dict[str, Any],
    offered: tuple[str, ...] = (),
) -> Task:
    tools = offered or (tool,)
    return Task(
        f"tool-{difficulty}-{n:02d}",
        difficulty,
        "tool",
        prompt,
        kind="tool",
        tools=tools,
        tool=tool,
        args=args,
    )


def _talk(
    n: int,
    difficulty: str,
    history: tuple[tuple[str, str], ...],
    prompt: str,
    answer: str,
    kind: str = "number",
) -> Task:
    return Task(
        f"multiturn-{difficulty}-{n:02d}",
        difficulty,
        "multiturn",
        prompt + SUFFIX,
        answer,
        kind,
        history=history,
    )


ONESHOT: tuple[Task, ...] = (
    # Easy
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
    _task(15, "easy", "What is 23 + 48?", "71"),
    _task(16, "easy", "What is 7 * 6?", "42"),
    _task(17, "easy", "What is 90 - 54?", "36"),
    _task(18, "easy", "What is the capital of Italy?", "rome", "word"),
    _task(19, "easy", "What word is the opposite of 'up'?", "down", "word"),
    _task(20, "easy", "How many months are in a year?", "12"),
    _task(21, "easy", "What is 81 / 9?", "9"),
    _task(22, "easy", "What is the first letter of the word 'zebra'?", "z", "word"),
    # Medium
    _task(
        1, "medium", "A shop sells pens at 3 dollars each. How many dollars do 14 pens cost?", "42"
    ),
    _task(2, "medium", "What is 15% of 240?", "36"),
    _task(3, "medium", "Convert 2.5 hours to minutes.", "150"),
    _task(4, "medium", "A train travels 60 km/h for 3.5 hours. How many km does it go?", "210"),
    _task(5, "medium", "What is (8 + 4) * (7 - 3)?", "48"),
    _task(6, "medium", "How many vowels (a, e, i, o, u) are in the word 'education'?", "5"),
    _task(
        7,
        "medium",
        "If today is Wednesday, what day of the week is it 10 days from now?",
        "saturday",
        "word",
    ),
    _task(8, "medium", "What is the sum of the digits of 98765?", "35"),
    _task(9, "medium", "A rectangle is 12 m by 7 m. What is its perimeter in metres?", "38"),
    _task(10, "medium", "What is the next number in the sequence 2, 6, 12, 20, 30?", "42"),
    _task(11, "medium", "What is 18% of 250?", "45"),
    _task(12, "medium", "Convert 3.25 hours to minutes.", "195"),
    _task(13, "medium", "A car travels 80 km/h for 2.5 hours. How many km does it go?", "200"),
    _task(14, "medium", "What is (9 + 6) * (10 - 4)?", "90"),
    _task(15, "medium", "How many consonants are in the word 'planet'?", "4"),
    _task(
        16,
        "medium",
        "If today is Monday, what day of the week is it 17 days from now?",
        "thursday",
        "word",
    ),
    _task(17, "medium", "What is the sum of the digits of 87654?", "30"),
    _task(
        18,
        "medium",
        "A square has a perimeter of 36 m. What is its area in square metres?",
        "81",
    ),
    _task(19, "medium", "What is the next number in the sequence 3, 8, 15, 24, 35?", "48"),
    _task(
        20,
        "medium",
        "A shirt costs 40 dollars and is discounted by 25%. What is the sale price in dollars?",
        "30",
    ),
    _task(21, "medium", "What is the average of 12, 18, 24 and 30?", "21"),
    # Hard
    _task(
        1,
        "hard",
        "A bat and a ball cost 1.10 dollars together. The bat costs 1.00 dollar more than the "
        "ball. How many cents does the ball cost?",
        "5",
    ),
    _task(
        2,
        "hard",
        "I have 3 boxes. Each box holds 4 smaller boxes, and each smaller box holds 5 pencils. "
        "How many pencils are there in total?",
        "60",
    ),
    _task(
        3,
        "hard",
        "Tom is twice as old as Ann. In 6 years the sum of their ages will be 42. How old is "
        "Tom now?",
        "20",
    ),
    _task(
        4,
        "hard",
        "What is the smallest positive whole number divisible by 2, 3, 4, 5 and 6?",
        "60",
    ),
    _task(5, "hard", "How many 3-digit numbers are divisible by 7?", "128"),
    _task(
        6,
        "hard",
        "One pipe fills a tank in 6 hours and another fills it in 3 hours. Working together, "
        "how many hours do they take?",
        "2",
    ),
    _task(
        7,
        "hard",
        "A farm has chickens and cows, 30 heads and 74 legs in all. How many cows are there?",
        "7",
    ),
    _task(8, "hard", "What is the sum of all whole numbers from 1 to 50?", "1275"),
    _task(9, "hard", "How many 2-digit numbers are divisible by 6?", "15"),
    _task(
        10,
        "hard",
        "A number is tripled and then 7 is subtracted, giving 38. What is the number?",
        "15",
    ),
    _task(
        11,
        "hard",
        "Train A leaves at 60 km/h. Two hours later Train B leaves the same station on the same "
        "line at 90 km/h. How many hours after B leaves does B catch A?",
        "4",
    ),
    _task(
        12, "hard", "What is the smallest positive whole number divisible by 4, 5, 6 and 8?", "120"
    ),
    _task(13, "hard", "What is the remainder when 2 to the power of 10 is divided by 7?", "2"),
    _task(
        14,
        "hard",
        "Worker A can finish a job in 10 days and worker B in 15 days. Working together, how "
        "many days do they take?",
        "6",
    ),
    _task(15, "hard", "In how many different orders can 3 books be arranged on a shelf?", "6"),
    _task(16, "hard", "What is the sum of the first 10 odd numbers?", "100"),
    _task(
        17,
        "hard",
        "Two siblings' ages add up to 30 and differ by 8. How old is the older one?",
        "19",
    ),
)

TOOLED: tuple[Task, ...] = (
    # Easy: one tool offered, one clear argument.
    _use(1, "easy", "What's the weather in Paris right now?", "get_weather", {"city": "Paris"}),
    _use(2, "easy", "Set a timer for 10 minutes.", "set_timer", {"minutes": 10}),
    _use(
        3,
        "easy",
        "What is the current price of the stock with ticker AAPL?",
        "get_stock_price",
        {"symbol": "AAPL"},
    ),
    _use(4, "easy", "How is the weather in Tokyo today?", "get_weather", {"city": "Tokyo"}),
    _use(5, "easy", "Start a 25 minute timer.", "set_timer", {"minutes": 25}),
    _use(6, "easy", "Look up the stock price of MSFT.", "get_stock_price", {"symbol": "MSFT"}),
    _use(7, "easy", "Is it raining in Nairobi?", "get_weather", {"city": "Nairobi"}),
    _use(8, "easy", "Please set a 45 minute timer.", "set_timer", {"minutes": 45}),
    _use(9, "easy", "What does NVDA trade at right now?", "get_stock_price", {"symbol": "NVDA"}),
    _use(10, "easy", "Tell me the weather in Oslo.", "get_weather", {"city": "Oslo"}),
    # Medium: several arguments, a few tools offered.
    _use(
        11,
        "medium",
        "Convert 250 USD to EUR.",
        "convert_currency",
        {"amount": 250, "from_currency": "USD", "to_currency": "EUR"},
        ("get_weather", "convert_currency", "translate_text"),
    ),
    _use(
        12,
        "medium",
        "Translate 'good morning' into Spanish.",
        "translate_text",
        {"text": "good morning", "target_language": "Spanish"},
        ("convert_currency", "translate_text", "send_email"),
    ),
    _use(
        13,
        "medium",
        "Email bob@example.com with the subject 'Lunch'.",
        "send_email",
        {"to": "bob@example.com", "subject": "Lunch"},
        ("set_timer", "send_email", "translate_text"),
    ),
    _use(
        14,
        "medium",
        "What's the weather in Berlin, in fahrenheit?",
        "get_weather",
        {"city": "Berlin", "unit": "fahrenheit"},
        ("get_weather", "get_stock_price", "set_timer"),
    ),
    _use(
        15,
        "medium",
        "How much is 80 GBP in JPY?",
        "convert_currency",
        {"amount": 80, "from_currency": "GBP", "to_currency": "JPY"},
        ("convert_currency", "get_stock_price", "send_email"),
    ),
    _use(
        16,
        "medium",
        "Put an event called 'Review' on 2026-03-05 at 14:00 in my calendar.",
        "create_event",
        {"title": "Review", "date": "2026-03-05", "time": "14:00"},
        ("create_event", "set_timer", "send_email"),
    ),
    # Hard: every tool offered and an indirect request.
    _use(
        17,
        "hard",
        "I'm flying from Lagos to Accra on 2026-05-12. What are my options?",
        "search_flights",
        {"origin": "Lagos", "destination": "Accra", "date": "2026-05-12"},
        tuple(TOOL_SPECS),
    ),
    _use(
        18,
        "hard",
        "How many Swiss francs would 1200 Canadian dollars get me?",
        "convert_currency",
        {"amount": 1200, "from_currency": "CAD", "to_currency": "CHF"},
        tuple(TOOL_SPECS),
    ),
    _use(
        19,
        "hard",
        "Remind me in an hour and a half to check the oven.",
        "set_timer",
        {"minutes": 90},
        tuple(TOOL_SPECS),
    ),
    _use(
        20,
        "hard",
        "Let priya@example.org know the deploy is done; use 'Deploy done' as the subject.",
        "send_email",
        {"to": "priya@example.org", "subject": "Deploy done"},
        tuple(TOOL_SPECS),
    ),
)


def _chat(*lines: str) -> tuple[tuple[str, str], ...]:
    """Alternating user and assistant lines, as (user, assistant) pairs."""
    return tuple(zip(lines[::2], lines[1::2], strict=True))


CONVERSATIONS: tuple[Task, ...] = (
    # Easy
    _talk(
        1,
        "easy",
        _chat("My favourite colour is teal.", "Noted: teal.", "I live in Lisbon.", "Lisbon."),
        "Which city did I say I live in?",
        "lisbon",
        "word",
    ),
    _talk(
        2,
        "easy",
        _chat("My dog is called Biscuit.", "Nice name.", "My cat is called Pepper.", "Lovely."),
        "What is my dog called?",
        "biscuit",
        "word",
    ),
    _talk(
        3,
        "easy",
        _chat("I bought 4 apples.", "Okay.", "Then I bought 3 more.", "Okay."),
        "How many apples do I have now?",
        "7",
    ),
    _talk(
        4,
        "easy",
        _chat("Let x be 12.", "Understood: x is 12."),
        "What is x plus 5?",
        "17",
    ),
    _talk(
        5,
        "easy",
        _chat("My brother is 9 years old.", "Got it."),
        "How old is my brother?",
        "9",
    ),
    _talk(
        6,
        "easy",
        _chat("The meeting is on Thursday.", "Noted.", "It starts at 3 pm.", "Noted."),
        "On what day is the meeting?",
        "thursday",
        "word",
    ),
    _talk(
        7,
        "easy",
        _chat(
            "I have 10 euros.",
            "Okay.",
            "I spent 4 on lunch.",
            "Okay.",
            "A friend paid me back nothing.",
            "Okay.",
            "I found no more money.",
            "Okay.",
            "That is everything.",
            "Understood.",
        ),
        "How many euros do I have left?",
        "6",
    ),
    _talk(
        8,
        "easy",
        _chat("The code word is harbor.", "Understood."),
        "What is the code word?",
        "harbor",
        "word",
    ),
    # Medium
    _talk(
        9,
        "medium",
        _chat(
            "I start with 20 dollars.",
            "Understood.",
            "I earn 15 more.",
            "Understood.",
            "I then spend 12.",
            "Understood.",
        ),
        "How many dollars do I have now?",
        "23",
    ),
    _talk(
        10,
        "medium",
        _chat(
            "Write down the number 3.",
            "Written.",
            "Write down the number 5.",
            "Written.",
            "Write down the number 8.",
            "Written.",
            "Write down the number 2.",
            "Written.",
            "Write down the number 7.",
            "Written.",
        ),
        "What is the sum of the numbers I had you write down?",
        "25",
    ),
    _talk(
        11,
        "medium",
        _chat(
            "Bread costs 2 dollars.",
            "Noted.",
            "Milk costs 3 dollars.",
            "Noted.",
            "Eggs cost 4 dollars.",
            "Noted.",
            "Cheese costs 6 dollars.",
            "Noted.",
            "Apples cost 5 dollars.",
            "Noted.",
        ),
        "I bought one of each item. What was the total in dollars?",
        "20",
    ),
    _talk(
        12,
        "medium",
        _chat(
            "Ana is older than Ben.",
            "Noted.",
            "Ben is older than Cy.",
            "Noted.",
            "Cy is older than Dee.",
            "Noted.",
            "Dee is older than Eli.",
            "Noted.",
        ),
        "Who is the youngest of them?",
        "eli",
        "word",
    ),
    _talk(
        13,
        "medium",
        _chat("A rectangle has a width of 6.", "Okay.", "Its length is 9.", "Okay."),
        "What is the area of that rectangle?",
        "54",
    ),
    _talk(
        14,
        "medium",
        _chat(
            "Our trip is 3 days long.",
            "Okay.",
            "We drive 240 km each day.",
            "Okay.",
            "The car uses 10 litres per 100 km.",
            "Okay.",
        ),
        "How many litres of fuel do we use on the whole trip?",
        "72",
    ),
    _talk(
        15,
        "medium",
        _chat(
            "My sister Mia is 4 years older than me.",
            "Okay.",
            "I am 11.",
            "Okay.",
            "Our brother is 3 years younger than Mia.",
            "Okay.",
        ),
        "How old is our brother?",
        "12",
    ),
    _talk(
        16,
        "medium",
        _chat("Take the number 15.", "Taken.", "Double it.", "30."),
        "Now subtract 8. What is the result?",
        "22",
    ),
    # Hard
    _talk(
        17,
        "hard",
        _chat(
            "Box A holds 12 balls.",
            "Noted.",
            "Box B holds 5 fewer than Box A.",
            "Noted.",
            "Box C holds twice what Box B holds.",
            "Noted.",
            "I move 3 balls from Box C into Box A.",
            "Done.",
            "Box D holds as many as Box A held at the start.",
            "Noted.",
        ),
        "How many balls does Box A hold now?",
        "15",
    ),
    _talk(
        18,
        "hard",
        _chat(
            "Tom has 3 times as many marbles as Sue.",
            "Okay.",
            "Sue has 8 marbles.",
            "Okay.",
            "Tom gives Sue 6 marbles.",
            "Okay.",
            "Then Sue loses 2 marbles.",
            "Okay.",
        ),
        "How many marbles does Sue have now?",
        "12",
    ),
    _talk(
        19,
        "hard",
        _chat(
            "A train leaves at 9:00 am.",
            "Okay.",
            "It travels at 90 km/h while moving.",
            "Okay.",
            "It stops for 30 minutes after 2 hours of travel.",
            "Okay.",
        ),
        "How many km has it covered 3 hours after leaving, counting the stop as part of the 3 "
        "hours?",
        "225",
    ),
    _talk(
        20,
        "hard",
        _chat(
            "Remember the list 4, 9, 16.",
            "Remembered.",
            "Add 25 to the list.",
            "The list is 4, 9, 16, 25.",
            "Remove 9.",
            "The list is 4, 16, 25.",
            "Add 36.",
            "The list is 4, 16, 25, 36.",
            "How many numbers are in the list?",
            "4.",
        ),
        "What is the sum of the numbers in the list now?",
        "81",
    ),
)

TASKS: tuple[Task, ...] = (*ONESHOT, *TOOLED, *CONVERSATIONS)
