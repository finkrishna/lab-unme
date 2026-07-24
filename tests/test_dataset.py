"""DistillDataset + collate over a hand-written 2-line Trace JSONL fixture."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

torch = pytest.importorskip("torch")

from unme.data.dataset import DistillDataset, collate
from unme.schemas import StepLogits, Trace


def _make_trace(
    prompt_id: str,
    input_ids: list[int],
    output_ids: list[int],
    k: int = 3,
) -> dict:
    steps = []
    for oid in output_ids:
        token_ids = [oid] + [(oid + j + 1) % 50 for j in range(k - 1)]
        # logprobs already log-softmax-ish
        logprobs = [0.0] + [-1.5] * (k - 1)
        steps.append(StepLogits(token_ids=token_ids, logprobs=logprobs).model_dump())
    tr = {
        "prompt_id": prompt_id,
        "domain": "math",
        "teacher_model": "teacher-x",
        "input_ids": input_ids,
        "output_ids": output_ids,
        "steps": steps,
        "hidden_path": None,
        "topk": k,
        "temperature": 1.0,
        "meta": {},
    }
    # validate against schema
    Trace.model_validate(tr)
    return tr


def test_dataset_two_line_fixture(tmp_path: Path):
    rows = [
        _make_trace("a", input_ids=[1, 2, 3], output_ids=[10, 11]),
        _make_trace("b", input_ids=[4, 5], output_ids=[20, 21, 22]),
    ]
    path = tmp_path / "kept.jsonl"
    with path.open("wb") as f:
        for r in rows:
            f.write(orjson.dumps(r))
            f.write(b"\n")

    ds = DistillDataset(path)
    assert len(ds) == 2

    item0 = ds[0]
    assert item0["input_ids"].tolist() == [1, 2, 3]
    assert item0["output_ids"].tolist() == [10, 11]
    assert item0["teacher_top_ids"].shape == (2, 3)
    assert item0["teacher_top_logprobs"].shape == (2, 3)
    assert item0["attention_mask"].tolist() == [1, 1, 1]
    assert item0["teacher_top_ids"][0, 0].item() == 10  # chosen token first in top-k

    item1 = ds[1]
    assert item1["output_ids"].shape[0] == 3

    batch = collate([item0, item1])
    assert batch["input_ids"].shape == (2, 3)  # padded to max_in=3
    assert batch["output_ids"].shape == (2, 3)  # padded to max_t=3
    assert batch["teacher_top_ids"].shape == (2, 3, 3)
    assert batch["teacher_top_logprobs"].shape == (2, 3, 3)
    assert batch["mask"].shape == (2, 3)
    # first example has 2 real output tokens
    assert batch["mask"][0].tolist() == [1.0, 1.0, 0.0]
    # second has 3
    assert batch["mask"][1].tolist() == [1.0, 1.0, 1.0]
    # attention: first full, second padded
    assert batch["attention_mask"][0].tolist() == [1, 1, 1]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0]
    assert batch["hidden_states"] is None


def test_collate_empty_raises():
    with pytest.raises(ValueError):
        collate([])
