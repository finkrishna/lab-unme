"""CLI train_cmd must actually run distillation (not just load the dataset)."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest
import yaml
from typer.testing import CliRunner

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from unme.cli import app
from unme.schemas import StepLogits, Trace

_TINY = "hf-internal-testing/tiny-random-gpt2"
_RUNNER = CliRunner()


def _make_trace(prompt_id: str, n_in: int, n_out: int, k: int, vocab: int) -> dict:
    input_ids = [(i % (vocab - 1)) + 1 for i in range(n_in)]
    output_ids = [((n_in + i) % (vocab - 1)) + 1 for i in range(n_out)]
    steps = []
    for oid in output_ids:
        token_ids = [oid, (oid + 1) % vocab, (oid + 2) % vocab, (oid + 3) % vocab][:k]
        logprobs = [0.0, -1.5, -2.5, -4.0][:k]
        steps.append(StepLogits(token_ids=token_ids, logprobs=logprobs).model_dump())
    row = {
        "prompt_id": prompt_id,
        "domain": "math",
        "teacher_model": _TINY,
        "input_ids": input_ids,
        "output_ids": output_ids,
        "steps": steps,
        "hidden_path": None,
        "topk": k,
        "temperature": 1.0,
        "meta": {},
    }
    Trace.model_validate(row)
    return row


def test_train_cmd_runs_real_distillation(tmp_path: Path) -> None:
    """train_cmd → unme.train.distill.train; loss_history non-empty."""
    vocab = 1000
    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()
    # Sibling verdicts.jsonl must NOT be consumed as traces.
    (filtered_dir / "verdicts.jsonl").write_text(
        '{"prompt_id":"x","keep":true,"reasons":[],"scores":{}}\n'
    )
    kept = filtered_dir / "kept.jsonl"
    rows = [
        _make_trace("p1", n_in=8, n_out=4, k=4, vocab=vocab),
        _make_trace("p2", n_in=6, n_out=4, k=4, vocab=vocab),
    ]
    with kept.open("wb") as f:
        for r in rows:
            f.write(orjson.dumps(r))
            f.write(b"\n")

    cfg_path = tmp_path / "distill.yaml"
    out_dir = tmp_path / "student"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "student": {"model": _TINY, "dtype": "float32"},
                "data": {"filtered": str(filtered_dir)},
                "distill": {
                    "temperature": 2.0,
                    "alpha_kl": 1.0,
                    "alpha_hidden": 0.0,
                    "alpha_ce": 0.1,
                    "hidden_layer_map": {},
                    "lr": 2.0e-4,
                    "batch_size": 1,
                    "epochs": 1,
                    "grad_clip": 1.0,
                },
                "output_dir": str(out_dir),
            }
        )
    )

    result = _RUNNER.invoke(
        app,
        [
            "train",
            "--config",
            str(cfg_path),
            "--data",
            str(filtered_dir),
            "--out",
            str(out_dir),
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "train done" in result.output
    assert "n_steps=" in result.output
    # Parse n_steps from output
    n_steps = None
    for part in result.output.replace("\n", " ").split():
        if part.startswith("n_steps="):
            n_steps = int(part.split("=", 1)[1])
            break
    assert n_steps is not None and n_steps >= 1
    # first/last loss printed
    assert "loss_first=" in result.output and "loss_last=" in result.output
    # checkpoint written
    assert (out_dir / "config.json").exists() or any(out_dir.iterdir())
