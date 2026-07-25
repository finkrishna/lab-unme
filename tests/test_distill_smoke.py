"""Smoke test for the distillation training loop.

Runs TWO training steps on ``hf-internal-testing/tiny-random-gpt2`` over a tiny
synthetic Trace JSONL fixture and asserts every step loss is finite. Skips cleanly
when torch/transformers are not installed (the ``[train]`` extra) so the rest of the
suite still runs in a fast CI without the heavy stack.
"""

from __future__ import annotations

import math
from pathlib import Path

import orjson
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from unme.schemas import StepLogits, Trace
from unme.train.distill import train


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
        "teacher_model": "hf-internal-testing/tiny-random-gpt2",
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


def test_distill_smoke_two_steps(tmp_path: Path) -> None:
    model = "hf-internal-testing/tiny-random-gpt2"
    # vocab/n_layer read from the model (smoke uses small values below the limits).
    vocab = 1000

    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()
    kept = filtered_dir / "kept.jsonl"
    rows = [
        _make_trace("p1", n_in=8, n_out=6, k=4, vocab=vocab),
        _make_trace("p2", n_in=10, n_out=8, k=4, vocab=vocab),
    ]
    with kept.open("wb") as f:
        for r in rows:
            f.write(orjson.dumps(r))
            f.write(b"\n")

    cfg_path = tmp_path / "distill.yaml"
    import yaml

    cfg = {
        "teacher": {"model": model, "dtype": "float32", "topk": 4, "emit_hidden_states": False},
        "student": {"model": model, "dtype": "float32"},
        "data": {
            "prompts": "data/prompts/pilot.jsonl",
            "traces": str(tmp_path / "traces"),
            "filtered": str(filtered_dir),
            "max_seq_len": 256,
        },
        "distill": {
            "temperature": 2.0,
            "alpha_kl": 1.0,
            "alpha_hidden": 0.5,
            "alpha_ce": 0.1,
            "hidden_layer_map": {1: 3},  # tiny model has 5 layers; keep student idxs in range
            "lr": 2.0e-4,
            "batch_size": 1,
            "epochs": 1,
            "grad_clip": 1.0,
        },
        "output_dir": str(tmp_path / "student"),
    }
    cfg_path.write_text(yaml.safe_dump(cfg))

    summary = train(cfg_path)

    assert summary["n_steps"] >= 2
    assert len(summary["loss_history"]) == summary["n_steps"]
    for loss in summary["loss_history"]:
        assert isinstance(loss, float)
        # finite (NaN/Inf guarded inside train() but assert defensively).
        assert math.isfinite(loss)
    # Default smoke config uses epochs: 1 → one epoch mean.
    assert "epoch_losses" in summary
    assert len(summary["epoch_losses"]) == 1
    assert math.isfinite(summary["epoch_losses"][0])


def test_select_device_honors_config_force_cpu() -> None:
    from unme.train.distill import _select_device

    dev = _select_device({"device": "cpu"})
    assert str(dev) == "cpu"


def test_distill_device_cpu_forced_in_summary(tmp_path: Path) -> None:
    """Tiny run with distill.device=cpu stays on CPU and reports it."""
    model = "hf-internal-testing/tiny-random-gpt2"
    vocab = 1000
    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()
    kept = filtered_dir / "kept.jsonl"
    rows = [_make_trace("p1", n_in=6, n_out=4, k=4, vocab=vocab)]
    with kept.open("wb") as f:
        for r in rows:
            f.write(orjson.dumps(r))
            f.write(b"\n")
    cfg_path = tmp_path / "distill.yaml"
    import yaml

    cfg_path.write_text(
        yaml.safe_dump(
            {
                "student": {"model": model},
                "data": {"filtered": str(filtered_dir)},
                "distill": {
                    "device": "cpu",
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
                "output_dir": str(tmp_path / "student"),
            }
        )
    )
    summary = train(cfg_path)
    assert summary.get("device") == "cpu"
    assert summary["n_steps"] >= 1


def test_distill_epoch_losses_length_matches_epochs(tmp_path: Path) -> None:
    """summary['epoch_losses'] has one mean per epoch (tiny 2-epoch run)."""
    model = "hf-internal-testing/tiny-random-gpt2"
    vocab = 1000
    filtered_dir = tmp_path / "filtered"
    filtered_dir.mkdir()
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
    import yaml

    n_epochs = 2
    cfg = {
        "student": {"model": model, "dtype": "float32"},
        "data": {"filtered": str(filtered_dir)},
        "distill": {
            "temperature": 2.0,
            "alpha_kl": 1.0,
            "alpha_hidden": 0.0,
            "alpha_ce": 0.1,
            "hidden_layer_map": {},
            "lr": 2.0e-4,
            "batch_size": 1,
            "epochs": n_epochs,
            "grad_clip": 1.0,
        },
        "output_dir": str(tmp_path / "student"),
    }
    cfg_path.write_text(yaml.safe_dump(cfg))

    summary = train(cfg_path)
    assert len(summary["epoch_losses"]) == n_epochs
    assert summary.get("n_epochs") == n_epochs
    for el in summary["epoch_losses"]:
        assert isinstance(el, float) and math.isfinite(el)
    # loss_history still has every step
    assert len(summary["loss_history"]) == summary["n_steps"]
    assert summary["n_steps"] >= n_epochs  # at least one step per epoch
