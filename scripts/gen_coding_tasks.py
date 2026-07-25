#!/usr/bin/env python3
"""Generate a diverse set of small Python-function tasks for a generalization run.

Each task = a natural-language prompt + a ``## tests`` block of asserts (the same
format CodeVerifier already parses). Asserts are DERIVED from a known-correct
reference implementation, so they're always valid.

Outputs (deterministic, seeded):
  data/prompts/coding_train.jsonl   — Prompt schema rows (teacher generates on these)
  data/eval_general/coding.jsonl    — held-out eval items {prompt, asserts, metric}

The train/eval split is by TASK (disjoint function specs), so the eval set truly
tests generalization — the student never saw those functions during training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]


def _lit(v: Any) -> str:
    """Render a Python literal for an assert (repr handles str/list/bool/num)."""
    return repr(v)


def _call(name: str, args: tuple) -> str:
    return f"{name}({', '.join(_lit(a) for a in args)})"


class Spec:
    """One function task: name, signature, description, reference impl, test inputs."""

    def __init__(self, name: str, sig: str, desc: str, ref: Callable, cases: list[tuple]):
        self.name = name
        self.sig = sig
        self.desc = desc
        self.ref = ref
        self.cases = cases

    def asserts(self) -> list[str]:
        out = []
        for args in self.cases:
            expected = self.ref(*args)
            out.append(f"assert {_call(self.name, args)} == {_lit(expected)}")
        return out

    def prompt_text(self) -> str:
        tests = "\n".join(self.asserts())
        return (
            f"Write a Python function `{self.sig}` that {self.desc}.\n"
            f"Return only a python code block.\n\n"
            f"## tests\n{tests}\n"
        )


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def base_specs() -> list[Spec]:
    S = Spec
    return [
        # ---- arithmetic ----
        S("subtract", "subtract(a, b)", "returns a minus b", lambda a, b: a - b,
          [(5, 3), (0, 0), (-2, -5)]),
        S("power", "power(a, b)", "returns a raised to the power b", lambda a, b: a ** b,
          [(2, 3), (5, 0), (3, 2)]),
        S("modulo", "modulo(a, b)", "returns the remainder of a divided by b", lambda a, b: a % b,
          [(7, 3), (10, 5), (9, 2)]),
        S("average", "average(a, b)", "returns the average of a and b as a float", lambda a, b: (a + b) / 2,
          [(2, 4), (0, 0), (5, 5)]),
        S("negate", "negate(x)", "returns the negation of x", lambda x: -x, [(3,), (-4,), (0,)]),
        S("double", "double(x)", "returns x times two", lambda x: x * 2, [(3,), (0,), (-5,)]),
        S("square", "square(x)", "returns x squared", lambda x: x * x, [(3,), (0,), (-4,)]),
        S("cube", "cube(x)", "returns x cubed", lambda x: x ** 3, [(2,), (0,), (-3,)]),
        S("half", "half(x)", "returns x divided by two as a float", lambda x: x / 2, [(4,), (0,), (-6,)]),
        S("increment", "increment(x)", "returns x plus one", lambda x: x + 1, [(3,), (-1,), (0,)]),
        S("add3", "add3(a, b, c)", "returns the sum of a, b and c", lambda a, b, c: a + b + c,
          [(1, 2, 3), (0, 0, 0), (-1, 1, 5)]),
        S("multiply3", "multiply3(a, b, c)", "returns the product of a, b and c", lambda a, b, c: a * b * c,
          [(2, 3, 4), (1, 0, 9), (2, 2, 2)]),
        S("sum_to_n", "sum_to_n(n)", "returns the sum of all integers from 1 to n",
          lambda n: n * (n + 1) // 2, [(5,), (1,), (10,)]),
        S("abs_diff", "abs_diff(a, b)", "returns the absolute difference between a and b",
          lambda a, b: abs(a - b), [(5, 3), (3, 5), (0, 0)]),
        S("int_div", "int_div(a, b)", "returns the integer (floor) division of a by b", lambda a, b: a // b,
          [(7, 2), (10, 5), (9, 4)]),
        # ---- comparison / conditional ----
        S("min2", "min2(a, b)", "returns the smaller of a and b", lambda a, b: min(a, b),
          [(1, 2), (5, 5), (-3, -1)]),
        S("max3", "max3(a, b, c)", "returns the largest of a, b and c", lambda a, b, c: max(a, b, c),
          [(1, 2, 3), (5, 5, 1), (-1, -2, -3)]),
        S("min3", "min3(a, b, c)", "returns the smallest of a, b and c", lambda a, b, c: min(a, b, c),
          [(1, 2, 3), (5, 5, 1), (-1, -2, -3)]),
        S("is_greater", "is_greater(a, b)", "returns True iff a is greater than b", lambda a, b: a > b,
          [(3, 2), (2, 3), (2, 2)]),
        S("is_between", "is_between(x, lo, hi)", "returns True iff lo <= x <= hi",
          lambda x, lo, hi: lo <= x <= hi, [(5, 0, 10), (-1, 0, 10), (10, 0, 10)]),
        S("sign", "sign(x)", "returns -1, 0, or 1 for the sign of x",
          lambda x: (x > 0) - (x < 0), [(5,), (-3,), (0,)]),
        S("clamp_low", "clamp_low(x, lo)", "returns x but not less than lo", lambda x, lo: max(x, lo),
          [(5, 0), (-3, 0), (2, 10)]),
        S("larger_abs", "larger_abs(a, b)", "returns whichever of a or b has the larger absolute value",
          lambda a, b: a if abs(a) >= abs(b) else b, [(3, -5), (-7, 2), (4, 1)]),
        S("grade", "grade(score)", "returns 'pass' if score >= 50 else 'fail'",
          lambda s: "pass" if s >= 50 else "fail", [(75,), (49,), (50,)]),
        S("letter_grade", "letter_grade(s)",
          "returns 'A' if s>=90, 'B' if s>=80, 'C' if s>=70, else 'F'",
          lambda s: "A" if s >= 90 else "B" if s >= 80 else "C" if s >= 70 else "F",
          [(95,), (85,), (60,)]),
        S("sign_word", "sign_word(x)", "returns 'positive', 'negative', or 'zero' for x",
          lambda x: "positive" if x > 0 else "negative" if x < 0 else "zero", [(5,), (-3,), (0,)]),
        S("fizz", "fizz(n)", "returns 'fizz' if n is divisible by 3 else the string form of n",
          lambda n: "fizz" if n % 3 == 0 else str(n), [(3,), (4,), (9,)]),
        # ---- boolean ----
        S("is_odd", "is_odd(n)", "returns True iff n is odd", lambda n: n % 2 == 1, [(3,), (4,), (0,)]),
        S("is_positive", "is_positive(x)", "returns True iff x is greater than zero", lambda x: x > 0,
          [(3,), (-1,), (0,)]),
        S("is_zero", "is_zero(x)", "returns True iff x equals zero", lambda x: x == 0, [(0,), (3,), (-2,)]),
        S("is_nonneg", "is_nonneg(x)", "returns True iff x is greater than or equal to zero",
          lambda x: x >= 0, [(0,), (5,), (-1,)]),
        S("both_true", "both_true(a, b)", "returns True iff both a and b are True",
          lambda a, b: a and b, [(True, True), (True, False), (False, False)]),
        S("either_true", "either_true(a, b)", "returns True iff a or b is True",
          lambda a, b: a or b, [(True, False), (False, False), (True, True)]),
        S("negate_bool", "negate_bool(b)", "returns the boolean opposite of b", lambda b: not b,
          [(True,), (False,)]),
        S("logical_xor", "logical_xor(a, b)", "returns the exclusive-or of booleans a and b",
          lambda a, b: a != b, [(True, False), (True, True), (False, False)]),
        S("same_sign", "same_sign(a, b)", "returns True iff a and b have the same sign (treat 0 as non-negative)",
          lambda a, b: (a >= 0) == (b >= 0), [(3, 5), (-2, 4), (-1, -9)]),
        # ---- strings ----
        S("reverse_str", "reverse_str(s)", "returns the string s reversed", lambda s: s[::-1],
          [("abc",), ("",), ("racecar",)]),
        S("shout", "shout(s)", "returns s in upper case", lambda s: s.upper(), [("hi",), ("Abc",), ("",)]),
        S("whisper", "whisper(s)", "returns s in lower case", lambda s: s.lower(), [("HI",), ("aBc",), ("",)]),
        S("first_char", "first_char(s)", "returns the first character of s", lambda s: s[0],
          [("abc",), ("x",), ("hi",)]),
        S("last_char", "last_char(s)", "returns the last character of s", lambda s: s[-1],
          [("abc",), ("x",), ("hi",)]),
        S("count_char", "count_char(s, c)", "returns how many times c appears in s",
          lambda s, c: s.count(c), [("banana", "a"), ("abc", "z"), ("", "x")]),
        S("count_vowels", "count_vowels(s)", "returns the number of vowels (aeiou) in s",
          lambda s: sum(1 for c in s if c in "aeiou"), [("hello",), ("xyz",), ("",)]),
        S("starts_with", "starts_with(s, p)", "returns True iff s starts with prefix p",
          lambda s, p: s.startswith(p), [("hello", "he"), ("hello", "wo"), ("", "")]),
        S("ends_with", "ends_with(s, suf)", "returns True iff s ends with suffix suf",
          lambda s, suf: s.endswith(suf), [("hello", "lo"), ("hello", "hi"), ("ab", "b")]),
        S("str_len", "str_len(s)", "returns the length of s", lambda s: len(s),
          [("abc",), ("",), ("hello",)]),
        S("repeat", "repeat(s, n)", "returns s repeated n times", lambda s, n: s * n,
          [("ab", 3), ("x", 0), ("z", 1)]),
        S("concat", "concat(a, b)", "returns string a joined with string b", lambda a, b: a + b,
          [("ab", "cd"), ("", "x"), ("hi", "")]),
        S("remove_spaces", "remove_spaces(s)", "returns s with all spaces removed",
          lambda s: s.replace(" ", ""), [("a b c",), ("  x",), ("no",)]),
        S("capitalize_first", "capitalize_first(s)", "returns s with only its first letter capitalized",
          lambda s: s.capitalize(), [("hello",), ("aBC",), ("",)]),
        S("count_words", "count_words(s)", "returns the number of whitespace-separated words in s",
          lambda s: len(s.split()), [("a b c",), ("hello",), ("",)]),
        S("is_palindrome", "is_palindrome(s)", "returns True iff s reads the same forwards and backwards",
          lambda s: s == s[::-1], [("racecar",), ("abc",), ("",)]),
        S("char_at", "char_at(s, i)", "returns the character of s at index i", lambda s, i: s[i],
          [("abc", 1), ("hello", 0), ("xy", -1)]),
        # ---- lists ----
        S("sum_list", "sum_list(xs)", "returns the sum of the numbers in xs", lambda xs: sum(xs),
          [([1, 2, 3],), ([],), ([-1, 1],)]),
        S("max_list", "max_list(xs)", "returns the largest number in xs", lambda xs: max(xs),
          [([1, 5, 3],), ([-2, -9],), ([7],)]),
        S("min_list", "min_list(xs)", "returns the smallest number in xs", lambda xs: min(xs),
          [([1, 5, 3],), ([-2, -9],), ([7],)]),
        S("product_list", "product_list(xs)", "returns the product of the numbers in xs",
          lambda xs: math.prod(xs), [([1, 2, 3],), ([5],), ([2, 2, 2],)]),
        S("length", "length(xs)", "returns the number of items in xs", lambda xs: len(xs),
          [([1, 2, 3],), ([],), (["a", "b"],)]),
        S("contains", "contains(xs, x)", "returns True iff x is in xs", lambda xs, x: x in xs,
          [([1, 2, 3], 2), ([1, 2], 9), ([], 0)]),
        S("first", "first(xs)", "returns the first item of xs", lambda xs: xs[0],
          [([1, 2, 3],), (["a"],), ([9, 8],)]),
        S("last", "last(xs)", "returns the last item of xs", lambda xs: xs[-1],
          [([1, 2, 3],), (["a"],), ([9, 8],)]),
        S("second", "second(xs)", "returns the second item of xs", lambda xs: xs[1],
          [([1, 2, 3],), (["a", "b"],), ([9, 8, 7],)]),
        S("reverse_list", "reverse_list(xs)", "returns a new list with the items of xs in reverse order",
          lambda xs: xs[::-1], [([1, 2, 3],), ([],), ([9],)]),
        S("drop_first", "drop_first(xs)", "returns xs without its first item", lambda xs: xs[1:],
          [([1, 2, 3],), ([9],), ([4, 5],)]),
        S("count_evens", "count_evens(xs)", "returns how many items in xs are even",
          lambda xs: sum(1 for v in xs if v % 2 == 0), [([1, 2, 4],), ([1, 3],), ([],)]),
        S("count_positives", "count_positives(xs)", "returns how many items in xs are > 0",
          lambda xs: sum(1 for v in xs if v > 0), [([1, -2, 3],), ([],), ([-1, -2],)]),
        S("sum_positives", "sum_positives(xs)", "returns the sum of the items in xs that are > 0",
          lambda xs: sum(v for v in xs if v > 0), [([1, -2, 3],), ([],), ([-5, -1],)]),
        S("all_positive", "all_positive(xs)", "returns True iff every item in xs is > 0",
          lambda xs: all(v > 0 for v in xs), [([1, 2, 3],), ([1, -1],), ([],)]),
        S("any_negative", "any_negative(xs)", "returns True iff any item in xs is < 0",
          lambda xs: any(v < 0 for v in xs), [([1, -2, 3],), ([1, 2],), ([],)]),
        S("index_of", "index_of(xs, x)", "returns the index of the first occurrence of x in xs",
          lambda xs, x: xs.index(x), [([1, 2, 3], 2), ([5, 6], 6), ([9], 9)]),
        S("head_tail_sum", "head_tail_sum(xs)", "returns the sum of the first and last items of xs",
          lambda xs: xs[0] + xs[-1], [([1, 2, 3],), ([5, 5],), ([4],)]),
        # ---- math ----
        S("factorial", "factorial(n)", "returns n! (0! = 1)",
          lambda n: math.factorial(n), [(0,), (1,), (5,)]),
        S("gcd", "gcd(a, b)", "returns the greatest common divisor of a and b",
          lambda a, b: math.gcd(a, b), [(12, 8), (7, 3), (10, 5)]),
        S("abs_val", "abs_val(x)", "returns the absolute value of x without calling abs",
          lambda x: x if x >= 0 else -x, [(3,), (-4,), (0,)]),
        S("clamp", "clamp(x, lo, hi)", "returns x limited to the range [lo, hi]",
          lambda x, lo, hi: max(lo, min(x, hi)), [(5, 0, 10), (-1, 0, 10), (99, 0, 10)]),
        S("fib", "fib(n)", "returns the n-th Fibonacci number (fib(0)=0, fib(1)=1)",
          lambda n: (lambda f: f(f, n))(lambda f, k: k if k < 2 else f(f, k - 1) + f(f, k - 2)),
          [(0,), (1,), (7,)]),
        S("is_prime", "is_prime(n)", "returns True iff n is a prime number", _is_prime,
          [(7,), (4,), (1,)]),
        S("digit_sum", "digit_sum(n)", "returns the sum of the digits of the non-negative integer n",
          lambda n: sum(int(d) for d in str(n)), [(123,), (0,), (99,)]),
        S("count_digits", "count_digits(n)", "returns the number of digits in the non-negative integer n",
          lambda n: len(str(n)), [(123,), (0,), (9,)]),
        S("ceil_div", "ceil_div(a, b)", "returns a divided by b rounded up to the nearest integer",
          lambda a, b: -(-a // b), [(7, 3), (10, 5), (9, 2)]),
        S("power_of_two", "power_of_two(n)", "returns 2 raised to the power n", lambda n: 2 ** n,
          [(0,), (3,), (5,)]),
        S("is_leap", "is_leap(y)", "returns True iff year y is a leap year",
          lambda y: y % 4 == 0 and (y % 100 != 0 or y % 400 == 0), [(2000,), (1900,), (2024,)]),
        S("triangle_area", "triangle_area(b, h)", "returns the area of a triangle with base b and height h",
          lambda b, h: b * h / 2, [(4, 3), (2, 2), (10, 1)]),
        S("celsius_to_f", "celsius_to_f(c)", "converts a Celsius temperature c to Fahrenheit",
          lambda c: c * 9 / 5 + 32, [(0,), (100,), (-40,)]),
        # ---- dicts ----
        S("get_or_zero", "get_or_zero(d, k)", "returns d[k] if k is in d else 0",
          lambda d, k: d.get(k, 0), [({"a": 1}, "a"), ({}, "x"), ({"b": 5}, "b")]),
        S("has_key", "has_key(d, k)", "returns True iff k is a key of dict d", lambda d, k: k in d,
          [({"a": 1}, "a"), ({}, "x"), ({"b": 2}, "c")]),
        S("key_count", "key_count(d)", "returns the number of keys in dict d", lambda d: len(d),
          [({"a": 1, "b": 2},), ({},), ({"x": 9},)]),
        S("values_sum", "values_sum(d)", "returns the sum of the values in dict d",
          lambda d: sum(d.values()), [({"a": 1, "b": 2},), ({},), ({"x": 5},)]),
    ]


def parametric(specs: list[Spec]) -> list[Spec]:
    """Add small parametric families to grow volume (constant-varied structural twins)."""
    S = Spec
    extra: list[Spec] = []
    for k in (1, 2, 5, 7, 10, 100):
        extra.append(S(f"add_{k}", f"add_{k}(x)", f"returns x plus {k}",
                       (lambda k: (lambda x: x + k))(k), [(0,), (3,), (-2,)]))
    for k in (2, 3, 4, 6, 10):
        extra.append(S(f"times_{k}", f"times_{k}(x)", f"returns x multiplied by {k}",
                       (lambda k: (lambda x: x * k))(k), [(0,), (3,), (-1,)]))
    for k in (3, 5, 10):
        extra.append(S(f"is_multiple_{k}", f"is_multiple_{k}(n)",
                       f"returns True iff n is a multiple of {k}",
                       (lambda k: (lambda n: n % k == 0))(k), [(0,), (k,), (k + 1,)]))
    return specs + extra


def to_prompt_row(spec: Spec) -> dict:
    return {
        "id": f"gen-{spec.name}",
        "domain": "coding",
        "messages": [{"role": "user", "content": spec.prompt_text()}],
        "meta": {"task": spec.name},
    }


def to_eval_row(spec: Spec) -> dict:
    return {
        "prompt": spec.prompt_text(),
        "asserts": spec.asserts(),
        "metric": "code_exec",
        "meta": {"task": spec.name},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate coding-task train/eval sets.")
    ap.add_argument("--eval-frac", type=float, default=0.2, help="fraction of tasks held out for eval")
    ap.add_argument("--limit", type=int, default=0, help="cap total tasks (0 = all); useful for quick CPU runs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    specs = parametric(base_specs())
    # guard against accidental duplicate names
    names = [s.name for s in specs]
    assert len(names) == len(set(names)), f"duplicate task names: {sorted({n for n in names if names.count(n) > 1})}"

    rng = random.Random(args.seed)
    rng.shuffle(specs)
    if args.limit and args.limit < len(specs):
        specs = specs[: args.limit]

    n_eval = max(1, round(len(specs) * args.eval_frac))
    eval_specs = specs[:n_eval]
    train_specs = specs[n_eval:]

    train_path = REPO / "data" / "prompts" / "coding_train.jsonl"
    eval_path = REPO / "data" / "eval_general" / "coding.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    with train_path.open("w") as f:
        for s in train_specs:
            f.write(json.dumps(to_prompt_row(s)) + "\n")
    with eval_path.open("w") as f:
        for s in eval_specs:
            f.write(json.dumps(to_eval_row(s)) + "\n")

    print(f"total_specs={len(specs)} train={len(train_specs)} eval={len(eval_specs)}")
    print(f"wrote {train_path.relative_to(REPO)}")
    print(f"wrote {eval_path.relative_to(REPO)}")
    print("held-out eval tasks:", ", ".join(s.name for s in eval_specs))


if __name__ == "__main__":
    main()
