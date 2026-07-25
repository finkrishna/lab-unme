"""inspect: pretty-print teacher top-k from a tiny Trace fixture."""

from __future__ import annotations

import math
from pathlib import Path

import orjson

from unme.inspect import format_topk_table, inspect_traces, load_first_trace
from unme.schemas import StepLogits, Trace


def _fixture_trace() -> Trace:
    # Two positions; top-2 each. Sampled ids 10 then 20.
    steps = [
        StepLogits(
            token_ids=[10, 11],
            logprobs=[math.log(0.7), math.log(0.3)],
        ),
        StepLogits(
            token_ids=[20, 21],
            logprobs=[math.log(0.6), math.log(0.4)],
        ),
    ]
    return Trace(
        prompt_id="inspect-1",
        domain="cs",
        teacher_model="mock-teacher",
        input_ids=[1, 2, 3],
        output_ids=[10, 20],
        steps=steps,
        topk=2,
        temperature=1.0,
        meta={"prompt": "hi"},
    )


def test_format_topk_table_shows_sampled_and_probs():
    tr = _fixture_trace()
    text = format_topk_table(tr)
    assert "inspect-1" in text
    assert "sampled" in text.lower() or "10" in text
    assert "10=" in text  # id=prob cells
    assert "0.7000" in text or "0.7" in text
    assert "20=" in text


def test_inspect_traces_reads_jsonl(tmp_path: Path):
    tr = _fixture_trace()
    path = tmp_path / "cs.jsonl"
    with path.open("wb") as f:
        f.write(orjson.dumps(tr.model_dump(mode="json")))
        f.write(b"\n")
    out = inspect_traces(path)
    assert "inspect-1" in out
    loaded = load_first_trace(tmp_path)  # dir form
    assert loaded.prompt_id == "inspect-1"
