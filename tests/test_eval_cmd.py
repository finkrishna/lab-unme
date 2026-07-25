"""eval_cmd records real GateReport ratios via harness.evaluate (stub student)."""

from __future__ import annotations

from pathlib import Path

import orjson
import yaml
from typer.testing import CliRunner

from unme.cli import app
from unme.registry import Registry
from unme.schemas import GateReport

_RUNNER = CliRunner()


def test_eval_cmd_uses_real_ratios_with_stub_student(tmp_path: Path, monkeypatch) -> None:
    suite = tmp_path / "eval"
    suite.mkdir()
    # Two items in domain cs; stub answers both correctly.
    with (suite / "cs.jsonl").open("wb") as f:
        for row in (
            {"prompt": "q1", "answer": "yes", "metric": "exact_match"},
            {"prompt": "q2", "answer": "42", "metric": "exact_match"},
        ):
            f.write(orjson.dumps(row))
            f.write(b"\n")

    def stub_student(prompt: str) -> str:
        return {"q1": "yes", "q2": "42"}.get(prompt, "nope")

    # Avoid loading a real HF checkpoint.
    monkeypatch.setattr("unme.cli.load_student_callable", lambda *a, **k: stub_student)

    cfg_path = tmp_path / "distill.yaml"
    reg_dir = tmp_path / "registry"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "output_dir": str(tmp_path / "student"),
                "eval": {
                    "suite": str(suite),
                    "regression_floor": 0.98,
                    "teacher_scores": {"cs": 1.0},
                },
            }
        )
    )
    (tmp_path / "student").mkdir()

    result = _RUNNER.invoke(
        app,
        [
            "eval",
            "stub-cand",
            "--config",
            str(cfg_path),
            "--registry",
            str(reg_dir),
            "--student",
            str(tmp_path / "student"),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "student=1.0000" in result.output or "student=1.0" in result.output
    assert "ratio=" in result.output
    # Not the old placeholder domain=math / 0.0
    assert "domain=cs" in result.output
    assert "placeholder" not in result.output.lower()

    report = Registry(reg_dir).load_report("stub-cand")
    assert isinstance(report, GateReport)
    assert report.results[0].domain == "cs"
    assert report.results[0].student_score == 1.0
    assert report.results[0].teacher_score == 1.0
    assert report.passed is True  # 1.0 >= 0.98


def test_eval_cmd_gate_may_refuse_on_low_score(tmp_path: Path, monkeypatch) -> None:
    suite = tmp_path / "eval"
    suite.mkdir()
    with (suite / "cs.jsonl").open("wb") as f:
        f.write(orjson.dumps({"prompt": "q", "answer": "right", "metric": "exact_match"}))
        f.write(b"\n")

    monkeypatch.setattr(
        "unme.cli.load_student_callable",
        lambda *a, **k: (lambda prompt: "wrong"),
    )

    cfg_path = tmp_path / "distill.yaml"
    reg_dir = tmp_path / "registry"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "eval": {
                    "suite": str(suite),
                    "regression_floor": 0.98,
                    "teacher_scores": {"cs": 1.0},
                },
            }
        )
    )
    result = _RUNNER.invoke(
        app,
        ["eval", "low", "--config", str(cfg_path), "--registry", str(reg_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "passed=False" in result.output or "passed=False" in result.output.replace(" ", "")
    report = Registry(reg_dir).load_report("low")
    assert report.results[0].student_score == 0.0
    assert report.passed is False
