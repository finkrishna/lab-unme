"""Pass/fail coverage for CodeVerifier, MathVerifier, ConsistencyVerifier."""

from __future__ import annotations

from unme.verify import CodeVerifier, ConsistencyVerifier, MathVerifier

# --- CodeVerifier -----------------------------------------------------------


def test_code_verifier_pass():
    prompt = (
        "Write add(a, b).\n\n"
        "## Tests\n"
        "assert add(2, 3) == 5\n"
        "assert add(0, 0) == 0\n"
    )
    output = "```python\ndef add(a, b):\n    return a + b\n```"
    ok, score = CodeVerifier().check(prompt, output)
    assert ok is True
    assert score == 1.0


def test_code_verifier_fail_wrong_impl():
    prompt = "## Tests\nassert add(2, 3) == 5\n"
    output = "```python\ndef add(a, b):\n    return a - b\n```"
    ok, score = CodeVerifier().check(prompt, output)
    assert ok is False
    assert score == 0.0


def test_code_verifier_fail_no_code():
    ok, score = CodeVerifier().check("## Tests\nassert True\n", "sorry I cannot")
    assert ok is False and score == 0.0


def test_code_verifier_rejects_banned_import():
    prompt = "## Tests\nassert add(1, 2) == 3\n"
    output = (
        "```python\n"
        "import os\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```"
    )
    ok, score = CodeVerifier().check(prompt, output)
    assert ok is False
    assert score == 0.0


def test_code_verifier_allows_allowlisted_import():
    prompt = "## Tests\nassert floor_plus(3.7, 1) == 4\n"
    output = (
        "```python\n"
        "import math\n"
        "def floor_plus(x, y):\n"
        "    return math.floor(x) + y\n"
        "```"
    )
    ok, score = CodeVerifier().check(prompt, output)
    assert ok is True
    assert score == 1.0


def test_code_verifier_rejects_dunder_attr():
    prompt = "## Tests\nassert f() == 1\n"
    output = (
        "```python\n"
        "def f():\n"
        "    return ().__class__.__bases__[0].__subclasses__\n"
        "```"
    )
    ok, _score = CodeVerifier().check(prompt, output)
    assert ok is False


def test_code_verifier_timeout_on_infinite_loop():
    prompt = "## Tests\nassert True\n"
    output = (
        "```python\n"
        "while True:\n"
        "    pass\n"
        "```"
    )
    # Short wall-clock timeout; rlimits also apply on POSIX.
    ok, score = CodeVerifier(timeout_s=0.3, cpu_seconds=1).check(prompt, output)
    assert ok is False
    assert score == 0.0


# --- MathVerifier -----------------------------------------------------------


def test_math_verifier_pass():
    prompt = "What is 17+25?\nANSWER: 42"
    output = "Adding gives 42.\nFinal answer: 42"
    ok, score = MathVerifier().check(prompt, output)
    assert ok is True
    assert score == 1.0


def test_math_verifier_pass_with_ctor_reference():
    ok, score = MathVerifier(reference=132).check("12*11?", "The answer is 132.")
    assert ok is True and score == 1.0


def test_math_verifier_fail_wrong_number():
    ok, score = MathVerifier(reference=42).check("sum?", "The answer is 41.")
    assert ok is False
    assert score == 0.0


def test_math_verifier_fail_no_number():
    ok, score = MathVerifier(reference=1).check("q", "I am not sure.")
    assert ok is False and score == 0.0


# --- ConsistencyVerifier ----------------------------------------------------


def test_consistency_verifier_pass_majority():
    peers = [
        "Final answer: 7",
        "I get 7",
        "maybe 9",
    ]
    # output agrees with majority 7 (3 of 4 after including output)
    ok, score = ConsistencyVerifier(peers=peers).check("x?", "The answer is 7")
    assert ok is True
    assert score >= 0.5


def test_consistency_verifier_fail_minority():
    peers = [
        "Final answer: 7",
        "Final answer: 7",
        "Final answer: 7",
    ]
    ok, score = ConsistencyVerifier(peers=peers).check("x?", "Final answer: 99")
    assert ok is False
    assert score == 0.0


def test_consistency_verifier_fail_no_majority():
    # 2 vs 2 after including output — not a strict majority for either
    peers = ["Final answer: 1", "Final answer: 2", "Final answer: 1"]
    ok, _score = ConsistencyVerifier(peers=peers).check("x?", "Final answer: 2")
    # pool: 1,2,1,2 → tie 2-2; majority_n == 2, len=4, 2 > 2 is False
    assert ok is False
