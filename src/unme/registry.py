"""Candidate checkpoint registry with regression-gate enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import orjson

from unme.schemas import EvalResult, GateReport


class PromotionError(RuntimeError):
    """Raised when promote() is called on a failing or unknown candidate."""


class Registry:
    """Persist candidates + GateReports as JSON under ``root``.

    Layout
    ------
    root/
      candidates/<name>/gate_report.json
      promoted.json          # single pointer to current promoted candidate
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.candidates_dir = self.root / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

    def _candidate_dir(self, candidate: str) -> Path:
        # prevent path escape
        name = Path(candidate).name
        return self.candidates_dir / name

    def record(self, candidate: str, report: GateReport) -> Path:
        """Write / overwrite the gate report for ``candidate``."""
        if report.candidate != candidate:
            # keep disk name and report in sync
            report = GateReport(
                candidate=candidate,
                results=report.results,
                regression_floor=report.regression_floor,
            )
        cdir = self._candidate_dir(candidate)
        cdir.mkdir(parents=True, exist_ok=True)
        path = cdir / "gate_report.json"
        path.write_bytes(orjson.dumps(report.model_dump(mode="json"), option=orjson.OPT_INDENT_2))
        return path

    def load_report(self, candidate: str) -> GateReport:
        path = self._candidate_dir(candidate) / "gate_report.json"
        if not path.exists():
            raise FileNotFoundError(f"No gate report for candidate {candidate!r} at {path}")
        data = orjson.loads(path.read_bytes())
        # rebuild EvalResult objects
        results = [EvalResult.model_validate(r) for r in data.get("results", [])]
        return GateReport(
            candidate=data["candidate"],
            results=results,
            regression_floor=float(data["regression_floor"]),
        )

    def promote(self, candidate: str) -> Path:
        """Promote ``candidate`` only if its GateReport.passed is True.

        Returns path to ``promoted.json``. Raises ``PromotionError`` otherwise.
        """
        try:
            report = self.load_report(candidate)
        except FileNotFoundError as e:
            raise PromotionError(str(e)) from e

        if not report.passed:
            raise PromotionError(
                f"Refusing to promote {candidate!r}: gate failed "
                f"(floor={report.regression_floor}, "
                f"ratios={[round(r.ratio, 4) for r in report.results]})"
            )

        pointer = {
            "candidate": candidate,
            "gate_report": report.model_dump(mode="json"),
        }
        out = self.root / "promoted.json"
        out.write_text(json.dumps(pointer, indent=2))
        # also stamp the candidate dir
        (self._candidate_dir(candidate) / "PROMOTED").write_text("ok\n")
        return out

    def current(self) -> str | None:
        path = self.root / "promoted.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return data.get("candidate")
