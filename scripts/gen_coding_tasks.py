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


def base_specs() -> list[Spec]:
    S = Spec
    specs: list[Spec] = [
        # arithmetic
        S("subtract", "subtract(a, b)", "returns a minus b", lambda a, b: a - b,
          [(5, 3), (0, 0), (-2, -5)]),
        S("power", "power(a, b)", "returns a raised to the power b", lambda a, b: a ** b,
          [(2, 3), (5, 0), (3, 2)]),
        S("modulo", "modulo(a, b)", "returns the remainder of a divided by b", lambda a, b: a % b,
          [(7, 3), (10, 5), (9, 2)]),
        S("average", "average(a, b)", "returns the average of a and b as a float", lambda a, b: (a + b) / 2,
          [(2, 4), (0, 0), (5, 5)]),
        S("negate", "negate(x)", "returns the negation of x", lambda x: -x,
          [(3,), (-4,), (0,)]),
        S("double", "double(x)", "returns x times two", lambda x: x * 2,
          [(3,), (0,), (-5,)]),
        S("square", "square(x)", "returns x squared", lambda x: x * x,
          [(3,), (0,), (-4,)]),
        # comparison / conditional
        S("min2", "min2(a, b)", "returns the smaller of a and b", lambda a, b: min(a, b),
          [(1, 2), (5, 5), (-3, -1)]),
        S("is_between", "is_between(x, lo, hi)", "returns True iff lo <= x <= hi",
          lambda x, lo, hi: lo <= x <= hi, [(5, 0, 10), (-1, 0, 10), (10, 0, 10)]),
        S("sign", "sign(x)", "returns -1, 0, or 1 for the sign of x",
          lambda x: (x > 0) - (x < 0), [(5,), (-3,), (0,)]),
        S("grade", "grade(score)", "returns 'pass' if score >= 50 else 'fail'",
          lambda s: "pass" if s >= 50 else "fail", [(75,), (49,), (50,)]),
        # boolean
        S("is_odd", "is_odd(n)", "returns True iff n is odd", lambda n: n % 2 == 1,
          [(3,), (4,), (0,)]),
        S("is_positive", "is_positive(x)", "returns True iff x is greater than zero", lambda x: x > 0,
          [(3,), (-1,), (0,)]),
        S("logical_xor", "logical_xor(a, b)", "returns the exclusive-or of booleans a and b",
          lambda a, b: a != b, [(True, False), (True, True), (False, False)]),
        # strings
        S("reverse_str", "reverse_str(s)", "returns the string s reversed", lambda s: s[::-1],
          [("abc",), ("",), ("racecar",)]),
        S("shout", "shout(s)", "returns s in upper case", lambda s: s.upper(),
          [("hi",), ("Abc",), ("",)]),
        S("count_char", "count_char(s, c)", "returns how many times c appears in s",
          lambda s, c: s.count(c), [("banana", "a"), ("abc", "z"), ("", "x")]),
        S("starts_with", "starts_with(s, p)", "returns True iff s starts with prefix p",
          lambda s, p: s.startswith(p), [("hello", "he"), ("hello", "wo"), ("", "")]),
        S("str_len", "str_len(s)", "returns the length of s without using len twice",
          lambda s: len(s), [("abc",), ("",), ("hello",)]),
        S("repeat", "repeat(s, n)", "returns s repeated n times", lambda s, n: s * n,
          [("ab", 3), ("x", 0), ("z", 1)]),
        # lists
        S("sum_list", "sum_list(xs)", "returns the sum of the numbers in xs", lambda xs: sum(xs),
          [([1, 2, 3],), ([],), ([-1, 1],)]),
        S("max_list", "max_list(xs)", "returns the largest number in xs", lambda xs: max(xs),
          [([1, 5, 3],), ([-2, -9],), ([7],)]),
        S("length", "length(xs)", "returns the number of items in xs", lambda xs: len(xs),
          [([1, 2, 3],), ([],), (["a", "b"],)]),
        S("contains", "contains(xs, x)", "returns True iff x is in xs", lambda xs, x: x in xs,
          [([1, 2, 3], 2), ([1, 2], 9), ([], 0)]),
        S("first", "first(xs)", "returns the first item of xs", lambda xs: xs[0],
          [([1, 2, 3],), (["a"],), ([9, 8],)]),
        S("last", "last(xs)", "returns the last item of xs", lambda xs: xs[-1],
          [([1, 2, 3],), (["a"],), ([9, 8],)]),
        S("count_positives", "count_positives(xs)", "returns how many items in xs are > 0",
          lambda xs: sum(1 for v in xs if v > 0), [([1, -2, 3],), ([],), ([-1, -2],)]),
        # math
        S("factorial", "factorial(n)", "returns n! (0! = 1)",
          lambda n: 1 if n == 0 else __import__("math").factorial(n), [(0,), (1,), (5,)]),
        S("gcd", "gcd(a, b)", "returns the greatest common divisor of a and b",
          lambda a, b: __import__("math").gcd(a, b), [(12, 8), (7, 3), (10, 5)]),
        S("abs_val", "abs_val(x)", "returns the absolute value of x without calling abs",
          lambda x: x if x >= 0 else -x, [(3,), (-4,), (0,)]),
        S("clamp", "clamp(x, lo, hi)", "returns x limited to the range [lo, hi]",
          lambda x, lo, hi: max(lo, min(x, hi)), [(5, 0, 10), (-1, 0, 10), (99, 0, 10)]),
        S("fib", "fib(n)", "returns the n-th Fibonacci number (fib(0)=0, fib(1)=1)",
          lambda n: (lambda f: f(f, n))(lambda f, k: k if k < 2 else f(f, k - 1) + f(f, k - 2)),
          [(0,), (1,), (7,)]),
    ]
    return specs


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
    ap.add_argument("--eval-frac", type=float, default=0.2, help="fraction held out for eval")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    specs = parametric(base_specs())
    rng = random.Random(args.seed)
    rng.shuffle(specs)

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
    print("train tasks:", ", ".join(s.name for s in train_specs[:8]), "...")
    print("held-out eval tasks:", ", ".join(s.name for s in eval_specs))


if __name__ == "__main__":
    main()
