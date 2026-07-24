"""Generate teacher ``Trace`` records (with per-token top-k logprobs) from a Prompt JSONL.

``generate(prompts_path, out_dir, config)`` reads ``unme.schemas.Prompt`` JSONL,
calls ``unme.teacher.client.TeacherLogitsClient``, tokenizes both the prompt and
the teacher's per-token top-k strings into ids via the configured HF tokenizer,
assembles a ``unme.schemas.Trace`` per prompt, and writes one JSONL file named by
domain into ``out_dir``.

Hidden states: the OpenAI ``/completions`` endpoint does not expose teacher hidden
states, so a weights-pass teacher backend is required to populate them. While that
backend is pending, ``teacher.emit_hidden_states``:

  * ``false`` (new default) — NO ``.npz`` is written; ``Trace.hidden_path`` stays
    ``None``. The hidden-match distillation term is skipped downstream.
  * ``true`` — a placeholder ``.npz`` of zeros (shape ``(n_steps, hidden_dim)``) is
    persisted and ``Trace.hidden_path`` is set, so the pipeline can exercise the
    path. ``distill.py`` separately treats an all-zero teacher hidden tensor as
    ABSENT, so this placeholder alone does NOT create a hidden-match signal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from unme.schemas import Prompt, StepLogits, Trace
from unme.teacher.client import TeacherCompletion, TeacherLogitsClient


def _read_prompts(prompts_path: str | Path) -> list[Prompt]:
    path = Path(prompts_path)
    if not path.exists():
        raise FileNotFoundError(f"prompts file not found: {path}")
    prompts: list[Prompt] = []
    with path.open("rb") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            prompts.append(Prompt.model_validate_json(line))
    if not prompts:
        raise ValueError(f"no prompts found at {path}")
    return prompts


def _build_client(config: dict) -> TeacherLogitsClient:
    tcfg = config.get("teacher") or {}
    base_url = tcfg.get("base_url")
    if not base_url:
        raise ValueError("config.teacher.base_url is required for the OpenAI-compat client")
    return TeacherLogitsClient(
        base_url=base_url,
        model=tcfg.get("model") or "default",
        topk=int(tcfg.get("topk", 20)),
        temperature=float(tcfg.get("temperature", 1.0)),
        api_key=tcfg.get("api_key") or "",
        max_tokens=int(tcfg.get("max_tokens", 1024)),
        timeout=float(tcfg.get("timeout", 120.0)),
        transport=_maybe_transport(tcfg.get("transport_value")),
    )


def _maybe_transport(transport: Any) -> Any:
    """Allow a constructed ``httpx.MockTransport`` to be injected via the config dict.

    The config is a plain dict (loaded from YAML in real runs), so tests pass a real
    transport object under the key ``teacher.transport_value``.
    """
    if transport is None or isinstance(transport, str):
        return None
    return transport


def _load_tokenizer(config: dict):
    from transformers import AutoTokenizer

    tcfg = config.get("teacher") or {}
    name = tcfg.get("tokenizer") or tcfg.get("model")
    if not name:
        raise ValueError("config.teacher.tokenizer (or teacher.model) is required for token ids")
    return AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def _prompt_text(prompt: Prompt) -> str:
    parts = [m.content for m in prompt.messages if m.role.value != "assistant"]
    system = [m.content for m in prompt.messages if m.role.value == "system"]
    body = "\n\n".join(parts) if parts else prompt.id
    if system:
        return "\n\n".join(system) + "\n\n" + body
    return body


def _token_to_ids(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if ids:
        return int(ids[0])
    # Fallback: encode the surrounding context-free repr (rare for real tokens).
    enc = tokenizer(text, add_special_tokens=False)
    return int(enc["input_ids"][0]) if enc["input_ids"] else 0


def _completion_output_ids(tokenizer, completion: TeacherCompletion) -> list[int]:
    out: list[int] = []
    for step in completion.steps:
        out.append(_token_to_ids(tokenizer, step.text))
    if not out and completion.text:
        # Older endpoint may have collapsed per-token logprobs into plain text.
        out = [int(x) for x in tokenizer.encode(completion.text, add_special_tokens=False)]
    return out


def _steps_to_logits(tokenizer, completion: TeacherCompletion, topk: int) -> list[StepLogits]:
    out: list[StepLogits] = []
    for step in completion.steps:
        ids = [_token_to_ids(tokenizer, lp.token) for lp in step.top_logprobs]
        lps = [float(lp.logprob) for lp in step.top_logprobs]
        if len(ids) > topk:
            ids, lps = ids[:topk], lps[:topk]
        out.append(StepLogits(token_ids=ids, logprobs=lps))
    return out


def _write_hidden(out_dir: Path, prompt_id: str, n_steps: int, hidden_dim: int) -> str:
    """Persist the teacher-hidden-state placeholder ``.npz`` (only when emit_hidden=True).

    A zero array is a PLACEHOLDER: ``train/distill.py`` treats an all-zero teacher
    hidden tensor as ABSENT and skips the hidden-match term, so this file alone
    produces no hidden-match gradient. Replace it with real teacher hidden states
    once a weights-pass teacher backend is wired.
    """
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    d = int(hidden_dim) if hidden_dim and hidden_dim > 0 else 1
    arr = np.zeros((max(n_steps, 1), d), dtype=np.float32)
    path = out_dir / f"{prompt_id}.npz"
    np.savez(str(path), hidden=arr)
    return str(path)


def generate(prompts_path: str | Path, out_dir: str | Path, config: dict) -> list[Trace]:
    """Read Prompt JSONL, call the teacher for each, emit ``Trace`` JSONL via orjson.

    Writes one JSONL file per domain into ``out_dir`` (e.g. ``out_dir/math.jsonl``).
    Returns the list of constructed ``Trace`` objects (also serialized to disk).
    """
    tcfg = config.get("teacher") or {}
    topk = int(tcfg.get("topk", 20))
    temperature = float(tcfg.get("temperature", 1.0))
    emit_hidden = bool(tcfg.get("emit_hidden_states", False))
    teacher_model = tcfg.get("model") or "teacher"

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = _load_tokenizer(config)
    hidden_dim = int(getattr(getattr(tokenizer, "model_input_names", None), "hidden_size", 0) or 0) or _peek_hidden_dim(tcfg.get("model"))

    prompts = _read_prompts(prompts_path)
    # group writes per domain; keep an in-memory list to return.
    by_domain: dict[str, list[bytes]] = {}
    traces: list[Trace] = []

    with _build_client(config) as client:
        for prompt in prompts:
            text = _prompt_text(prompt)
            completion = client.complete(text)
            input_ids = [int(x) for x in tokenizer.encode(text, add_special_tokens=True)]
            output_ids = _completion_output_ids(tokenizer, completion)
            steps = _steps_to_logits(tokenizer, completion, topk)
            if not steps and output_ids:
                # No per-token logprobs returned; emit degenerate single-token steps
                # so Trace's 1:1 validator still passes.
                steps = [StepLogits(token_ids=[oid], logprobs=[0.0]) for oid in output_ids]
            tr = Trace(
                prompt_id=prompt.id,
                domain=prompt.domain,
                teacher_model=teacher_model,
                input_ids=input_ids,
                output_ids=output_ids,
                steps=steps,
                topk=topk,
                temperature=temperature,
                meta={"prompt": text, "completion": completion.text} | {k: v for k, v in prompt.meta.items()},
            )
            if emit_hidden:
                tr.hidden_path = _write_hidden(out, prompt.id, len(output_ids), hidden_dim)
            by_domain.setdefault(prompt.domain, []).append(orjson.dumps(tr.model_dump(mode="json")))
            traces.append(tr)

    for domain, rows in by_domain.items():
        with (out / f"{domain}.jsonl").open("wb") as f:
            for r in rows:
                f.write(r)
                f.write(b"\n")
    return traces


def _peek_hidden_dim(model_name: str | None) -> int:
    if not model_name:
        return 0
    try:
        from transformers import AutoConfig
    except ImportError:
        return 0
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:  # noqa: BLE001  (offline / unknown model: fall back to 0)
        return 0
    return int(getattr(cfg, "hidden_size", 0) or 0)
