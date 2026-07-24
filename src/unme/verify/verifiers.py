"""Domain verifiers used as synth data filters (and later as RLVR rewards).

Protocol: ``check(prompt, output) -> (passed, score)``.
"""

from __future__ import annotations

import ast
import os
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


# Modules allowed inside submitted student code (stdlib subset only).
_CODE_IMPORT_ALLOWLIST = frozenset(
    {
        "math",
        "typing",
        "itertools",
        "functools",
        "collections",
        # common submodules / re-exports used as `import collections.abc`
        "collections.abc",
    }
)


def _root_module(name: str | None) -> str:
    if not name:
        return ""
    return name.split(".", 1)[0]


def ast_code_is_safe(code: str) -> bool:
    """Reject non-allowlisted imports and dunder attribute access via AST.

    Returns True only if the module parses and passes static checks.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = _root_module(alias.name)
                # allow full dotted names that are explicitly listed (collections.abc)
                if alias.name not in _CODE_IMPORT_ALLOWLIST and root not in _CODE_IMPORT_ALLOWLIST:
                    return False
        elif isinstance(node, ast.ImportFrom):
            # relative imports not allowed
            if node.level and node.level > 0:
                return False
            mod = node.module or ""
            root = _root_module(mod)
            if mod not in _CODE_IMPORT_ALLOWLIST and root not in _CODE_IMPORT_ALLOWLIST:
                return False
        elif isinstance(node, ast.Attribute):
            if isinstance(node.attr, str) and node.attr.startswith("__") and node.attr.endswith("__"):
                return False
        elif isinstance(node, ast.Name):
            # bare dunder names like __builtins__
            if (
                node.id.startswith("__")
                and node.id.endswith("__")
                and node.id
                in {"__builtins__", "__globals__", "__dict__", "__class__", "__import__"}
            ):
                return False
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            return False
    return True


def _posix_rlimit_preexec(cpu_seconds: int, as_bytes: int):
    """Return a preexec_fn that sets RLIMIT_CPU and RLIMIT_AS, or None if unavailable."""

    def _apply() -> None:
        try:
            import resource
        except ImportError:
            return
        # CPU seconds (soft, hard)
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        except (ValueError, OSError):
            pass
        # Address space (virtual memory)
        try:
            if hasattr(resource, "RLIMIT_AS"):
                resource.setrlimit(resource.RLIMIT_AS, (as_bytes, as_bytes))
        except (ValueError, OSError):
            pass

    # Only attach on POSIX where resource + fork semantics exist.
    if not hasattr(os, "fork"):
        return None
    try:
        import resource  # noqa: F401
    except ImportError:
        return None
    return _apply


class CodeVerifier:
    """Execute model code + prompt asserts in a subprocess with a hard timeout.

    Static AST gate (import allowlist + no dunder attr access) runs before spawn.
    On POSIX, the child also gets RLIMIT_CPU / RLIMIT_AS via preexec_fn.
    """

    def __init__(
        self,
        timeout_s: float = 2.0,
        *,
        cpu_seconds: int = 2,
        as_mb: int = 256,
    ) -> None:
        self.timeout_s = timeout_s
        self.cpu_seconds = cpu_seconds
        self.as_bytes = int(as_mb) * 1024 * 1024

    def check(self, prompt: str, output: str) -> tuple[bool, float]:
        code = extract_python_code(output)
        if not code:
            return False, 0.0
        asserts = extract_asserts(prompt)
        if not asserts:
            return False, 0.0

        if not ast_code_is_safe(code):
            return False, 0.0

        script = code.rstrip() + "\n\n" + "\n".join(asserts) + "\n"
        preexec = _posix_rlimit_preexec(self.cpu_seconds, self.as_bytes)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "solution.py"
                path.write_text(script, encoding="utf-8")
                if preexec is not None:
                    proc = subprocess.run(
                        [sys.executable, str(path)],
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_s,
                        check=False,
                        preexec_fn=preexec,
                    )
                else:
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
