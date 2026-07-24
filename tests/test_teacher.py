"""Teacher client + Trace generation via a mocked httpx transport (no live calls).

Uses ``httpx.MockTransport`` to return a canned OpenAI-style ``/completions`` response
with per-token top-k logprobs and asserts a valid ``unme.schemas.Trace`` is produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import orjson
import pytest

pytest.importorskip("transformers")

from unme.schemas import Prompt, Trace
from unme.teacher.client import TeacherLogitsClient
from unme.teacher.generate import generate


def _completions_response(model: str) -> dict:
    return {
        "id": "cmpl-test",
        "model": model,
        "choices": [
            {
                "text": "x = 1 + 2\n",
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {"token": "x", "logprob": -0.1, "top_logprobs": [
                            {"token": "x", "logprob": -0.1},
                            {"token": "y", "logprob": -2.3},
                            {"token": "z", "logprob": -4.5},
                        ]},
                        {"token": " ", "logprob": -0.2, "top_logprobs": [
                            {"token": " ", "logprob": -0.2},
                            {"token": "=", "logprob": -3.0},
                        ]},
                        {"token": "=", "logprob": -0.05, "top_logprobs": [
                            {"token": "=", "logprob": -0.05},
                            {"token": ">", "logprob": -5.0},
                        ]},
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3},
    }


def _mock_transport(model: str):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/completions")
        body = json.loads(request.content)
        assert body["model"] == model
        assert "logprobs" in body
        return httpx.Response(200, json=_completions_response(model))

    return httpx.MockTransport(handler)


def _request_count_model() -> str:
    return "hf-internal-testing/tiny-random-gpt2"


def test_client_completions_returns_topk_logprobs():
    model = _request_count_model()
    with TeacherLogitsClient(
        base_url="http://localhost:7999/v1",
        model=model,
        topk=3,
        temperature=1.0,
        api_key="sk-test",
        transport=_mock_transport(model),
    ) as client:
        comp = client.complete("hello")
    assert comp.model == model
    assert len(comp.steps) == 3
    assert comp.steps[0].text == "x"
    assert comp.steps[0].top_logprobs[0].token == "x"
    assert len(comp.steps[0].top_logprobs) <= 3
    assert comp.usage["completion_tokens"] == 3
    assert comp.finish_reason == "stop"


def test_generate_emits_valid_trace(tmp_path):
    model = _request_count_model()
    prompts_path = tmp_path / "prompts.jsonl"
    prompt = Prompt(
        id="p-1",
        domain="math",
        messages=[{"role": "user", "content": "compute 1+2"}],
        meta={"seed": "1"},
    )
    with prompts_path.open("wb") as f:
        f.write(orjson.dumps(prompt.model_dump(mode="json")))
        f.write(b"\n")
    config = {
        "teacher": {
            "model": model,
            "base_url": "http://localhost:7999/v1",
            "api_key": "sk-test",
            "topk": 4,
            "temperature": 1.0,
            "emit_hidden_states": True,
            "transport_value": _mock_transport(model),
        }
    }
    out_dir = tmp_path / "traces"
    traces = generate(prompts_path, out_dir, config)

    assert len(traces) == 1
    tr = traces[0]
    assert isinstance(tr, Trace)
    assert tr.prompt_id == "p-1"
    assert tr.domain == "math"
    assert tr.teacher_model == model
    assert len(tr.input_ids) > 0
    assert len(tr.output_ids) == len(tr.steps)  # schema 1:1
    assert tr.topk == 4
    # top-k support length <= topk
    assert all(len(s.token_ids) <= tr.topk for s in tr.steps)
    assert all(len(s.token_ids) == len(s.logprobs) for s in tr.steps)
    assert (out_dir / "math.jsonl").exists()
    # hidden path written when emit_hidden_states=True
    assert tr.hidden_path is not None and Path(tr.hidden_path).exists()


# helpers ---------------------------------------------------------------------
