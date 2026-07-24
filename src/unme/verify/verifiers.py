"""Domain verifiers used as synth data filters (and later as RLVR rewards).

Protocol: ``check(prompt, output) -> (passed, score)``.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class Verifier(Protocol):
    """Binary + graded check over a (prompt, model output) pair."""

    def check(self, prompt: str, output: str) -> tuple[bool, float]:
        """Return (passed, score in [0, 1])."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CODE_FENCE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)
_ASSERT_LINE = re.compile(r"^\s*assert\b.*", re.MULTILINE)
_ANSWER_PATTERNS = [
    re.compile(r"####\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"),
    re.compile(r"\\boxed\{([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\}"),
    re.compile(r"final answer\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r"the answer is\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*ANSWER\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.IGNORECASE),
    re.compile(r"(?:^|\n)\s*reference\s*[:=]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", re.IGNORECASE),
]


def extract_python_code(output: str) -> str | None:
    """Prefer fenced python blocks; else whole output if it parses as module."""
    blocks = _CODE_FENCE.findall(output)
    if blocks:
        return max((b.strip() for b in blocks), key=len)
    text = output.strip()
    if not text:
        return None
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        return None


def extract_asserts(prompt: str) -> list[str]:
    """Pull assert statements from the prompt (tests provided to the model)."""
    # Explicit section
    m = re.search(
        r"(?:##\s*tests|asserts?\s*:)\s*\n([\s\S]+)",
        prompt,
        re.IGNORECASE,
    )
    section = m.group(1) if m else prompt
    return [ln.strip() for ln in section.splitlines() if _ASSERT_LINE.match(ln)]


def parse_numeric(text: str) -> float | None:
    """Best-effort final numeric answer from free text."""
    for pat in _ANSWER_PATTERNS:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    # last standalone number as weak fallback
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def _normalize_answer_text(text: str) -> str:
    n = parse_numeric(text)
    if n is not None:
        # stable string for majority vote on numbers
        if abs(n - round(n)) < 1e-9:
            return str(round(n))
        return f"{n:.6g}"
    # collapse whitespace for non-numeric answers
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------


class CodeVerifier:
    """Execute model code + prompt asserts in a subprocess with a hard timeout."""

    def __init__(self, timeout_s: float = 2.0) -> None:
        self.timeout_s = timeout_s

    def check(self, prompt: str, output: str) -> tuple[bool, float]:
        code = extract_python_code(output)
        if not code:
            return False, 0.0
        asserts = extract_asserts(prompt)
        if not asserts:
            return False, 0.0

        # Structural guard: reject obviously unsafe constructs before spawn.
        banned = ("import os", "import sys", "import subprocess", "__import__", "open(", "exec(", "eval(")
        low = code.lower()
        if any(b in low for b in banned):
            return False, 0.0

        script = code.rstrip() + "\n\n" + "\n".join(asserts) + "\n"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "solution.py"
                path.write_text(script, encoding="utf-8")
                proc = subprocess.run(
                    [sys.executable, str(path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
        except subprocess.TimeoutExpired:
            return False, 0.0
        except OSError:
            return False, 0.0

        if proc.returncode == 0:
            return True, 1.0
        return False, 0.0


class MathVerifier:
    """Parse a final numeric answer and compare to a reference.

    Reference resolution order:
      1. constructor ``reference`` if provided
      2. ``ANSWER:`` / ``reference:`` / ``####`` in the *prompt*
    """

    def __init__(self, reference: float | str | None = None, atol: float = 1e-6) -> None:
        if reference is None:
            self.reference: float | None = None
        else:
            self.reference = float(reference)
        self.atol = atol

    def check(self, prompt: str, output: str) -> tuple[bool, float]:
        ref = self.reference
        if ref is None:
            ref = parse_numeric(prompt)
        if ref is None:
            return False, 0.0
        pred = parse_numeric(output)
        if pred is None:
            return False, 0.0
        ok = abs(pred - ref) <= self.atol
        return ok, 1.0 if ok else 0.0


class ConsistencyVerifier:
    """Majority agreement of ``output`` against peer samples for the same prompt.

    Pass peer completions via ``peers``. Score is the fraction of the pool that
    matches the majority label; ``passed`` requires ``output`` equals majority
    and the majority size is strictly > half the pool.
    """

    def __init__(self, peers: Sequence[str] | None = None) -> None:
        self.peers = list(peers or [])

    def check(self, prompt: str, output: str) -> tuple[bool, float]:
        del prompt  # prompt identity is external; peers are already scoped
        pool = [*self.peers, output]
        if not pool:
            return False, 0.0
        labels = [_normalize_answer_text(s) for s in pool]
        # empty/degenerate answers kill consistency
        if any(not lab for lab in labels) and not _normalize_answer_text(output):
            return False, 0.0
        counts = Counter(labels)
        majority_label, majority_n = counts.most_common(1)[0]
        frac = majority_n / len(labels)
        out_label = _normalize_answer_text(output)
        agrees = out_label == majority_label
        # strict majority (> 50%)
        passed = agrees and majority_n > len(labels) / 2
        score = frac if agrees else 0.0
        return passed, float(score)
