"""Regression-gate eval harness.

``evaluate(student, teacher_scores, suite_dir, floor)`` scores a candidate student on a
fixed per-domain eval suite, pairs each domain's score with the precomputed teacher
score, and assembles a ``unme.schemas.GateReport`` enforcing the regression floor.

Suite layout: ``suite_dir/<domain>.jsonl`` where each line is an eval item of the form::

    {"prompt": "...", "answer": "...", "meta": {...}}

The default metric is exact-match of the student's free-text answer against the
item's ``answer`` field (whitespace-trimmed, case-folded). A domain with no items
counts as score 0.0 for the student.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson

from unme.schemas import EvalResult, GateReport

# A "student" is anything callable on a prompt string that returns the candidate's
# answer text. We also tolerate objects exposing ``predict(prompt)`` or
# ``generate_one(prompt).text`` (the legacy ``TeacherClient`` shape) for convenience.
StudentFn = Callable[[str], str]


def _call_student(student: Any, prompt: str) -> str:
    if callable(student):
        return str(student(prompt))
    for attr in ("predict", "answer"):
        fn = getattr(student, attr, None)
        if callable(fn):
            return str(fn(prompt))
    gen = getattr(student, "generate_one", None)
    if callable(gen):
        # Legacy TeacherClient shape: generate_one(req) returns an object with .text.
        # We pass the raw prompt; callers passing that shape are responsible for the
        # expected call convention.
        out = gen(prompt)  # type: ignore[arg-type]
        return getattr(out, "text", str(out)) if out is not None else ""
    raise TypeError(
        f"student must be callable on a prompt string or expose .predict/.answer/.generate_one; "
        f"got {type(student).__name__}"
    )


def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase for exact-match comparison."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _load_suite(suite_dir: str | Path) -> dict[str, list[dict]]:
    root = Path(suite_dir)
    if not root.exists():
        raise FileNotFoundError(f"eval suite dir not found: {root}")
    by_domain: dict[str, list[dict]] = {}
    for fp in sorted(root.glob("*.jsonl")):
        domain = fp.stem
        items: list[dict] = []
        with fp.open("rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(orjson.loads(line))
        if items:
            by_domain[domain] = items
    if not by_domain:
        raise ValueError(f"no domain eval files (*.jsonl) found under {root}")
    return by_domain


def _score_domain(student: Any, items: list[dict]) -> float:
    if not items:
        return 0.0
    from unme.verify import CodeVerifier

    correct = 0
    seen = 0
    code_verifier = CodeVerifier()
    for item in items:
        prompt = item.get("prompt")
        if not isinstance(prompt, str):
            # Malformed eval item: skip it from the denominator.
            continue
        # Route by metric BEFORE exact_match answer-check.
        metric = item.get("metric", "exact_match")

        if metric == "code_exec":
            asserts = item.get("asserts")
            if not isinstance(asserts, list) or not asserts:
                continue
            seen += 1
            got = _call_student(student, prompt)
            # Reuse CodeVerifier: synthetic prompt carries only the assert block.
            check_prompt = "## tests\n" + "\n".join(str(a) for a in asserts)
            ok, _score = code_verifier.check(check_prompt, got)
            if ok:
                correct += 1
            continue

        expected = item.get("answer")
        if not isinstance(expected, str):
            continue
        seen += 1
        got = _call_student(student, prompt)
        if metric != "exact_match":
            # Unknown metrics count as incorrect (same as prior behavior).
            continue
        if _normalize(got) == _normalize(expected):
            correct += 1
    return correct / seen if seen else 0.0


def evaluate(
    student: Any,
    teacher_scores: dict[str, float],
    suite_dir: str | Path,
    floor: float,
    *,
    candidate: str = "student",
) -> GateReport:
    """Score ``student`` on every domain suite and return a ``GateReport``.

    Args:
        student: callable ``(prompt: str) -> str`` (or an object exposing
            ``.predict`` / ``.answer`` / ``.generate_one``).
        teacher_scores: ``{domain: teacher_score}`` precomputed baseline.
        suite_dir: directory containing ``<domain>.jsonl`` eval files.
        floor: regression floor; the gate passes only if every domain's
            ``student_score / teacher_score`` (or 0 if teacher_score==0) is ``>= floor``.
        candidate: name written into the GateReport (defaults to ``"student"``).

    Returns:
        ``unme.schemas.GateReport`` with one ``EvalResult`` per domain found on disk.
    """
    by_domain = _load_suite(suite_dir)
    results: list[EvalResult] = []
    for domain, items in by_domain.items():
        student_score = _score_domain(student, items)
        teacher_score = float(teacher_scores.get(domain, 0.0))
        results.append(
            EvalResult(
                domain=domain,
                metric="exact_match",
                student_score=student_score,
                teacher_score=teacher_score,
            )
        )
    return GateReport(candidate=candidate, results=results, regression_floor=float(floor))
