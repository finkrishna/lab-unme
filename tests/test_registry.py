"""Registry promotion: blocked on failed gate, allowed on pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from unme.registry import PromotionError, Registry
from unme.schemas import EvalResult, GateReport


def _report(candidate: str, student: float, teacher: float, floor: float = 0.98) -> GateReport:
    return GateReport(
        candidate=candidate,
        results=[
            EvalResult(
                domain="math",
                metric="holdout",
                student_score=student,
                teacher_score=teacher,
            )
        ],
        regression_floor=floor,
    )


def test_promote_blocked_when_gate_fails(tmp_path: Path):
    reg = Registry(tmp_path)
    # 0.5 / 1.0 = 0.5 < 0.98
    reg.record("cand-bad", _report("cand-bad", student=0.5, teacher=1.0))
    assert reg.load_report("cand-bad").passed is False
    with pytest.raises(PromotionError, match="Refusing to promote"):
        reg.promote("cand-bad")
    assert not (tmp_path / "promoted.json").exists()


def test_promote_allowed_when_gate_passes(tmp_path: Path):
    reg = Registry(tmp_path)
    # 0.99 / 1.0 >= 0.98
    reg.record("cand-good", _report("cand-good", student=0.99, teacher=1.0))
    assert reg.load_report("cand-good").passed is True
    path = reg.promote("cand-good")
    assert path.exists()
    assert reg.current() == "cand-good"
    assert (tmp_path / "candidates" / "cand-good" / "PROMOTED").exists()


def test_promote_unknown_candidate(tmp_path: Path):
    reg = Registry(tmp_path)
    with pytest.raises(PromotionError):
        reg.promote("missing")


def test_multi_domain_any_fail_blocks(tmp_path: Path):
    reg = Registry(tmp_path)
    report = GateReport(
        candidate="mixed",
        results=[
            EvalResult(domain="math", metric="m", student_score=1.0, teacher_score=1.0),
            EvalResult(domain="code", metric="m", student_score=0.5, teacher_score=1.0),
        ],
        regression_floor=0.98,
    )
    reg.record("mixed", report)
    with pytest.raises(PromotionError):
        reg.promote("mixed")
