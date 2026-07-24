"""Regression gate: every domain >= floor passes; one domain below floor fails.

This tests both the ``unme.schemas.GateReport.passed`` property (pure) and the
``unme.eval.harness.evaluate`` harness that builds a report from a per-domain
suite directory + a callable student + precomputed teacher scores.
"""

from __future__ import annotations

from pathlib import Path

import orjson

from unme.eval.harness import evaluate
from unme.schemas import EvalResult, GateReport


def _write_suite(path: Path, domain: str, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for it in items:
            f.write(orjson.dumps(it))
            f.write(b"\n")


# --- GateReport.passed ------------------------------------------------------


def test_gate_passes_when_all_ratios_meet_floor():
    report = GateReport(
        candidate="c",
        results=[
            EvalResult(domain="math", metric="exact_match", student_score=0.99, teacher_score=1.0),
            EvalResult(domain="code", metric="exact_match", student_score=0.98, teacher_score=1.0),
        ],
        regression_floor=0.98,
    )
    assert report.passed is True
    assert all(r.ratio >= report.regression_floor for r in report.results)


def test_gate_fails_when_one_domain_below_floor():
    report = GateReport(
        candidate="c",
        results=[
            EvalResult(domain="math", metric="exact_match", student_score=1.00, teacher_score=1.0),
            EvalResult(domain="code", metric="exact_match", student_score=0.5, teacher_score=1.0),
        ],
        regression_floor=0.98,
    )
    assert report.passed is False
    ratios = {r.domain: r.ratio for r in report.results}
    assert ratios["code"] < 0.98 <= ratios["math"]


def test_gate_treats_zero_teacher_score_as_zero_ratio():
    # EvalResult.ratio is 0.0 when teacher_score==0, so a domain with teacher 0
    # always fails the floor unless floor<=0.
    report = GateReport(
        candidate="c",
        results=[
            EvalResult(domain="qa", metric="exact_match", student_score=1.0, teacher_score=0.0),
        ],
        regression_floor=0.01,
    )
    assert report.passed is False


# --- evaluate() harness -----------------------------------------------------


def _student_that_answers_correctly_on_math_but_not_code(seed_math: str, seed_code: str):
    def student(prompt: str) -> str:
        if prompt.startswith("math:"):
            return seed_math
        return seed_code

    return student


def test_evaluate_builds_passing_report(tmp_path: Path):
    # Math suite: every item has answer "42"; student returns "42".
    _write_suite(
        tmp_path / "math.jsonl",
        "math",
        [
            {"prompt": "math: 6*7?", "answer": "42", "metric": "exact_match"},
            {"prompt": "math: 40+2?", "answer": "42", "metric": "exact_match"},
        ],
    )
    _write_suite(
        tmp_path / "code.jsonl",
        "code",
        [{"prompt": "code: add", "answer": "ok", "metric": "exact_match"}],
    )
    student = _student_that_answers_correctly_on_math_but_not_code("42", "ok")
    teacher_scores = {"math": 1.0, "code": 1.0}
    report = evaluate(student, teacher_scores, tmp_path, floor=1.0, candidate="cand-alpha")

    assert report.candidate == "cand-alpha"
    assert report.regression_floor == 1.0
    by_domain = {r.domain: r for r in report.results}
    assert by_domain["math"].student_score == 1.0
    assert by_domain["code"].student_score == 1.0
    assert all(r.metric == "exact_match" for r in report.results)
    assert report.passed is True


def test_evaluate_builds_failing_report_when_one_domain_regresses(tmp_path: Path):
    _write_suite(
        tmp_path / "math.jsonl",
        "math",
        [
            {"prompt": "math: a", "answer": "42", "metric": "exact_match"},
            {"prompt": "math: b", "answer": "42", "metric": "exact_match"},
        ],
    )
    _write_suite(
        tmp_path / "code.jsonl",
        "code",
        [
            {"prompt": "code: a", "answer": "ok", "metric": "exact_match"},
            {"prompt": "code: b", "answer": "ok", "metric": "exact_match"},
        ],
    )
    # Student gets every math answer right but every code answer wrong.
    student = _student_that_answers_correctly_on_math_but_not_code("42", "WRONG")
    teacher_scores = {"math": 1.0, "code": 1.0}

    report = evaluate(student, teacher_scores, tmp_path, floor=0.98, candidate="cand-beta")
    by_domain = {r.domain: r for r in report.results}
    assert by_domain["math"].student_score == 1.0
    assert by_domain["code"].student_score == 0.0
    assert by_domain["math"].teacher_score == 1.0
    assert by_domain["code"].teacher_score == 1.0
    assert report.passed is False


def test_evaluate_missing_suite_dir_raises(tmp_path: Path):
    import pytest

    with pytest.raises(FileNotFoundError):
        evaluate(lambda p: "", {}, tmp_path / "nope", 0.5)


def test_evaluate_empty_suite_dir_raises(tmp_path: Path):
    import pytest

    tmp_path.mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError):
        evaluate(lambda p: "", {}, tmp_path, 0.5)
