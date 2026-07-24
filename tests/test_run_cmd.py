"""`unme run --skip-train`: generate → filter → DistillDataset with mock teacher."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import orjson
import pytest
import yaml
from typer.testing import CliRunner

pytest.importorskip("torch")
pytest.importorskip("transformers")

from unme.cli import app
from unme.data.dataset import DistillDataset

_TINY = "hf-internal-testing/tiny-random-gpt2"
_RUNNER = CliRunner()


def _mock_transport(model: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == model
        text = "```python\ndef add(a, b):\n    return a + b\n```\n"
        content = [
            {
                "token": "def",
                "logprob": -0.1,
                "top_logprobs": [
                    {"token": "def", "logprob": -0.1},
                    {"token": "x", "logprob": -2.0},
                ],
            },
            {
                "token": " add",
                "logprob": -0.2,
                "top_logprobs": [
                    {"token": " add", "logprob": -0.2},
                    {"token": " y", "logprob": -3.0},
                ],
            },
            {
                "token": "\n",
                "logprob": -0.05,
                "top_logprobs": [
                    {"token": "\n", "logprob": -0.05},
                    {"token": " ", "logprob": -4.0},
                ],
            },
        ]
        return httpx.Response(
            200,
            json={
                "id": "cmpl-run",
                "model": model,
                "choices": [
                    {
                        "text": text,
                        "finish_reason": "stop",
                        "logprobs": {"content": content},
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3},
            },
        )

    return httpx.MockTransport(handler)


def test_run_skip_train_generate_filter_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Chain generate → filter → dataset-load without calling heavy train."""
    # Minimal pilot-shaped prompt CodeVerifier can pass against the mock solution.
    prompts_path = tmp_path / "prompts.jsonl"
    prompt = {
        "id": "cs-add-run",
        "domain": "cs",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Write add(a, b).\n\n## tests\n"
                    "assert add(2, 3) == 5\n"
                    "assert add(0, 0) == 0\n"
                ),
            }
        ],
        "meta": {"task": "add"},
    }
    with prompts_path.open("wb") as f:
        f.write(orjson.dumps(prompt))
        f.write(b"\n")

    traces_dir = tmp_path / "traces"
    filtered_dir = tmp_path / "filtered"
    registry_dir = tmp_path / "registry"
    config_path = tmp_path / "distill.yaml"

    cfg = {
        "teacher": {
            "model": _TINY,
            "tokenizer": _TINY,
            "base_url": "http://localhost:7999/v1",
            "api_key": "sk-test",
            "topk": 4,
            "temperature": 1.0,
            "emit_hidden_states": False,
            # Injected after YAML load (objects cannot round-trip through YAML).
            "transport_value": None,
        },
        "data": {
            "prompts": str(prompts_path),
            "traces": str(traces_dir),
            "filtered": str(filtered_dir),
        },
        "eval": {"regression_floor": 0.98},
    }
    config_path.write_text(yaml.safe_dump({k: v for k, v in cfg.items() if k != "teacher"} | {
        "teacher": {k: v for k, v in cfg["teacher"].items() if k != "transport_value"},
    }))

    transport = _mock_transport(_TINY)

    # Patch YAML loader used by cli so generate sees MockTransport.
    import unme.cli as cli_mod

    raw = cli_mod._load_yaml

    def _load_with_transport(path: Path) -> dict:
        data = raw(path)
        t = data.setdefault("teacher", {})
        t["model"] = _TINY
        t["tokenizer"] = _TINY
        t["base_url"] = "http://localhost:7999/v1"
        t["api_key"] = "sk-test"
        t["topk"] = 4
        t["temperature"] = 1.0
        t["emit_hidden_states"] = False
        t["transport_value"] = transport
        data.setdefault("data", {})
        data["data"]["prompts"] = str(prompts_path)
        data["data"]["traces"] = str(traces_dir)
        data["data"]["filtered"] = str(filtered_dir)
        return data

    monkeypatch.setattr(cli_mod, "_load_yaml", _load_with_transport)

    result = _RUNNER.invoke(
        app,
        [
            "run",
            "--config",
            str(config_path),
            "--skip-train",
            "--candidate",
            "run-smoke",
            "--registry",
            str(registry_dir),
            "--domain",
            "cs",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert (traces_dir / "cs.jsonl").exists(), result.output
    kept = filtered_dir / "kept.jsonl"
    assert kept.exists(), result.output
    kept_lines = [ln for ln in kept.read_bytes().splitlines() if ln]
    assert len(kept_lines) >= 1, result.output

    ds = DistillDataset(kept)
    assert len(ds) >= 1
    item = ds[0]
    for key in ("input_ids", "output_ids", "teacher_top_ids", "teacher_top_logprobs"):
        assert key in item
    assert "dataset" in result.output.lower() or "loaded" in result.output.lower()
