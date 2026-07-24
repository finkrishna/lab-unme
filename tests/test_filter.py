"""Strict filter: keep good traces, drop bad/dup/degenerate."""

from __future__ import annotations

from pathlib import Path

import orjson

from unme.schemas import StepLogits, Trace
from unme.synth.filter import filter_traces
from unme.verify import CodeVerifier, MathVerifier


def _step(tid: int, k: int = 2) -> StepLogits:
    ids = [tid] + [tid + 1 + i for i in range(k - 1)]
    # crude log-mass
    logprobs = [0.0] + [-2.0] * (k - 1)
    return StepLogits(token_ids=ids[:k], logprobs=logprobs[:k])


def _trace(
    prompt_id: str,
    *,
    domain: str,
    output_ids: list[int],
    meta: dict[str, str],
    input_ids: list[int] | None = None,
) -> Trace:
    inp = input_ids or [1, 2, 3]
    steps = [_step(t) for t in output_ids]
    return Trace(
        prompt_id=prompt_id,
        domain=domain,
        teacher_model="mock-teacher",
        input_ids=inp,
        output_ids=output_ids,
        steps=steps,
        topk=2,
        temperature=1.0,
        meta=meta,
    )


def _write_jsonl(path: Path, traces: list[Trace]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        for t in traces:
            f.write(orjson.dumps(t.model_dump(mode="json")))
            f.write(b"\n")


def test_filter_keeps_math_pass_drops_fail(tmp_path: Path):
    good = _trace(
        "p-math-1",
        domain="math",
        output_ids=[10, 11, 12],
        meta={"prompt": "What is 2+2?\nANSWER: 4", "output": "Final answer: 4"},
    )
    bad = _trace(
        "p-math-2",
        domain="math",
        output_ids=[20, 21, 22],
        meta={"prompt": "What is 2+2?\nANSWER: 4", "output": "Final answer: 7"},
    )
    src = tmp_path / "traces"
    src.mkdir()
    _write_jsonl(src / "batch.jsonl", [good, bad])
    out = tmp_path / "filtered"
    verdicts = filter_traces(src, out, verifiers=[MathVerifier()])

    kept_path = out / "kept.jsonl"
    kept = [orjson.loads(line) for line in kept_path.read_bytes().splitlines() if line]
    assert len(kept) == 1
    assert kept[0]["prompt_id"] == "p-math-1"
    assert any(v.prompt_id == "p-math-2" and not v.keep for v in verdicts)


def test_filter_dedup_prompt_id(tmp_path: Path):
    a = _trace(
        "same",
        domain="math",
        output_ids=[1, 2, 3],
        meta={"prompt": "1+1?\nANSWER: 2", "output": "Final answer: 2"},
    )
    b = _trace(
        "same",
        domain="math",
        output_ids=[1, 2, 3, 4, 5],
        meta={"prompt": "1+1?\nANSWER: 2", "output": "Final answer: 2"},
    )
    src = tmp_path / "t"
    src.mkdir()
    _write_jsonl(src / "x.jsonl", [a, b])
    out = tmp_path / "o"
    verdicts = filter_traces(src, out, verifiers=[MathVerifier()])
    kept = list((out / "kept.jsonl").read_bytes().splitlines())
    assert len(kept) == 1
    assert sum(1 for v in verdicts if v.keep) == 1
    assert any("dedup_prompt_id" in v.reasons for v in verdicts)


def test_filter_drops_repetitive(tmp_path: Path):
    # 40 identical tokens → repetitive
    ids = [7] * 40
    rep = _trace(
        "rep",
        domain="general",
        output_ids=ids,
        meta={"prompt": "hi", "output": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    )
    src = tmp_path / "t"
    src.mkdir()
    _write_jsonl(src / "x.jsonl", [rep])
    out = tmp_path / "o"
    verdicts = filter_traces(src, out, verifiers=[])
    assert all(not v.keep for v in verdicts)
    assert (out / "kept.jsonl").read_bytes().strip() == b""


def test_filter_code_verifier_integration(tmp_path: Path):
    good = _trace(
        "code-1",
        domain="code",
        output_ids=[1, 2, 3, 4],
        meta={
            "prompt": "add\n## Tests\nassert add(1,2)==3\n",
            "output": "```python\ndef add(a,b):\n    return a+b\n```",
        },
    )
    src = tmp_path / "t"
    src.mkdir()
    _write_jsonl(src / "x.jsonl", [good])
    out = tmp_path / "o"
    verdicts = filter_traces(src, out, verifiers=[CodeVerifier()])
    assert any(v.keep for v in verdicts)
    assert len((out / "kept.jsonl").read_bytes().splitlines()) == 1
