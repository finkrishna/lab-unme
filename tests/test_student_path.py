"""train and eval must resolve the same student checkpoint directory."""

from __future__ import annotations

from pathlib import Path

from unme.cli import _resolve_student_dir


def test_resolve_student_dir_config_wins_when_no_override():
    cfg = {"output_dir": "outputs/student-local"}
    assert _resolve_student_dir(cfg, None) == Path("outputs/student-local")
    assert _resolve_student_dir({}, None) == Path("outputs/student")


def test_resolve_student_dir_override_wins():
    cfg = {"output_dir": "outputs/student-local"}
    assert _resolve_student_dir(cfg, Path("outputs/override")) == Path("outputs/override")


def test_train_and_eval_resolve_same_dir_from_config():
    """What `unme run` (train without --out) and `unme eval` both use."""
    cfg = {"output_dir": "outputs/student-local"}
    # train_cmd(..., out=None) and eval_cmd(..., student_path=None)
    train_dir = _resolve_student_dir(cfg, None)
    eval_dir = _resolve_student_dir(cfg, None)
    assert train_dir == eval_dir == Path("outputs/student-local")
