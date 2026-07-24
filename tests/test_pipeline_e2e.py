"""End-to-end smoke: generate (mocked teacher) → filter → DistillDataset/collate.

Does not assert training quality — only that stages connect and batch shapes/keys are sane.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import orjson
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from unme.data.dataset import DistillDataset, collate
from unme.synth.filter import filter_traces
from unme.teacher.generate import generate
from unme.verify import CodeVerifier

# Tiny HF model used only as tokenizer name (and teacher.model id in traces).
_TINY = "hf-internal-testing/tiny-random-gpt2"

# Correct solutions keyed by prompt needles (first match wins).
_SOLUTIONS: list[tuple[str, str]] = [
    ("add(a, b)", "```python\ndef add(a, b):\n    return a + b\n```\n"),
    ("mul(a, b)", "```python\ndef mul(a, b):\n    return a * b\n```\n"),
    ("my_abs(x)", "```python\ndef my_abs(x):\n    return x if x >= 0 else -x\n```\n"),
    (
        "clamp(x, lo, hi)",
        (
            "```python\ndef clamp(x, lo, hi):\n    if x < lo:\n        return lo\n"
            "    if x > hi:\n        return hi\n    return x\n```\n"
        ),
    ),
    ("is_even(n)", "```python\ndef is_even(n):\n    return n % 2 == 0\n```\n"),
    (
        "factorial(n)",
        (
            "```python\ndef factorial(n):\n    r = 1\n    for i in range(1, n + 1):\n"
            "        r *= i\n    return r\n```\n"
        ),
    ),
    ("max2(a, b)", "```python\ndef max2(a, b):\n    return a if a >= b else b\n```\n"),
    (
        "sum_list(xs)",
        (
            "```python\ndef sum_list(xs):\n    s = 0\n    for x in xs:\n        s += x\n"
            "    return s\n```\n"
        ),
    ),
]


def _solution_for(prompt: str) -> str:
    for needle, code in _SOLUTIONS:
        if needle in prompt:
            return code
    return "```python\ndef solve():\n    return 0\n```\n"


def _logprobs_for_text(text: str, topk: int = 4) -> list[dict]:
    """Turn text into a short list of OpenAI-style per-token logprob steps."""
    # Prefer word-ish chunks so we get several distinct token ids after tokenization.
    parts = [p for p in re.split(r"(\s+)", text) if p != ""]
    if len(parts) > 24:
        # keep sequence short for e2e speed / uniqueness heuristics
        parts = parts[:24]
    content = []
    for i, tok in enumerate(parts):
        alts = [{"token": tok, "logprob": -0.05 - 0.01 * i}]
        for j in range(1, topk):
            alts.append({"token": f"alt{j}", "logprob": -2.0 - j})
        content.append({"token": tok, "logprob": alts[0]["logprob"], "top_logprobs": alts})
    return content


def _mock_transport(model: str, topk: int = 4) -> httpx.MockTransport:
    """Same pattern as tests/test_teacher.py, but returns correct code per prompt."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/completions")
        body = json.loads(request.content)
        assert body["model"] == model
        prompt = body.get("prompt") or ""
        text = _solution_for(prompt)
        payload = {
            "id": "cmpl-e2e",
            "model": model,
            "choices": [
                {
                    "text": text,
                    "finish_reason": "stop",
                    "logprobs": {"content": _logprobs_for_text(text, topk=topk)},
                }
            ],
            "usage": {
                "prompt_tokens": 8,
                "completion_tokens": len(_logprobs_for_text(text, topk=topk)),
            },
        }
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def test_pipeline_generate_filter_dataset(tmp_path: Path):
    """generate → filter (CodeVerifier) → DistillDataset → collate keys/shapes."""
    repo = Path(__file__).resolve().parents[1]
    pilot = repo / "data" / "prompts" / "pilot.jsonl"
    assert pilot.exists(), "missing data/prompts/pilot.jsonl fixtures"

    # Use a 2-prompt slice for speed, but still real pilot schema rows.
    rows = []
    with pilot.open("rb") as f:
        for line in f:
            if line.strip():
                rows.append(orjson.loads(line))
            if len(rows) >= 2:
                break
    prompts_path = tmp_path / "prompts.jsonl"
    with prompts_path.open("wb") as f:
        for r in rows:
            f.write(orjson.dumps(r))
            f.write(b"\n")

    topk = 4
    config = {
        "teacher": {
            "model": _TINY,
            "tokenizer": _TINY,
            "base_url": "http://localhost:7999/v1",
            "api_key": "sk-test",
            "topk": topk,
            "temperature": 1.0,
            "emit_hidden_states": False,
            "transport_value": _mock_transport(_TINY, topk=topk),
        }
    }

    traces_dir = tmp_path / "traces"
    traces = generate(prompts_path, traces_dir, config)
    assert len(traces) >= 1
    assert (traces_dir / "cs.jsonl").exists()
    # meta.prompt + meta.completion present for CodeVerifier
    assert traces[0].meta.get("prompt")
    assert traces[0].meta.get("completion")

    filtered_dir = tmp_path / "filtered"
    verdicts = filter_traces(traces_dir, filtered_dir, verifiers=[CodeVerifier()])
    kept_path = filtered_dir / "kept.jsonl"
    assert kept_path.exists()
    kept_lines = [ln for ln in kept_path.read_bytes().splitlines() if ln]
    assert len(kept_lines) >= 1, (
        f"expected ≥1 kept trace after CodeVerifier; verdicts="
        f"{[v.model_dump() for v in verdicts]}"
    )

    ds = DistillDataset(kept_path)
    assert len(ds) >= 1
    items = [ds[i] for i in range(len(ds))]
    batch = collate(items)

    expected_keys = {
        "input_ids",
        "output_ids",
        "teacher_top_ids",
        "teacher_top_logprobs",
        "mask",
    }
    assert expected_keys.issubset(batch.keys())

    b = batch["input_ids"].shape[0]
    assert b == len(ds) and b >= 1
    t = batch["output_ids"].shape[1]
    k = batch["teacher_top_ids"].shape[-1]
    assert batch["output_ids"].shape == (b, t)
    assert batch["teacher_top_ids"].shape == (b, t, k)
    assert batch["teacher_top_logprobs"].shape == (b, t, k)
    assert batch["mask"].shape == (b, t)
    assert batch["attention_mask"].shape[0] == b
    # mask has at least one real token somewhere
    assert float(batch["mask"].sum()) >= 1.0
    # teacher top-k width is positive
    assert k >= 1
