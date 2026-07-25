"""Pretty-print teacher top-k distributions from Trace JSONL.

Traces store token *ids* and logprobs. When a tokenizer is available we decode
ids to surface strings; otherwise we fall back to raw ids.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import orjson

from unme.schemas import Trace


def load_first_trace(path: str | Path) -> Trace:
    """Load the first Trace from a JSONL file or the first ``*.jsonl`` under a dir."""
    p = Path(path)
    files: list[Path]
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.glob("*.jsonl"))
    else:
        raise FileNotFoundError(f"traces path not found: {p}")
    if not files:
        raise FileNotFoundError(f"no Trace JSONL under {p}")
    with files[0].open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return Trace.model_validate(orjson.loads(line))
    raise ValueError(f"no Trace rows in {files[0]}")


def try_load_tokenizer(name: str | None) -> Any | None:
    """Load an HF tokenizer by name/path; return None on failure (offline/missing)."""
    if not name or not str(name).strip():
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained(str(name).strip(), trust_remote_code=True)
    except (OSError, ValueError, RuntimeError, ImportError, AttributeError, TypeError):
        return None


def decode_token_id(tokenizer: Any | None, token_id: int) -> str:
    """Decode one token id to a display string; fall back to the numeric id."""
    if tokenizer is None:
        return str(token_id)
    try:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except (TypeError, ValueError, KeyError, IndexError, RuntimeError, OSError):
        return str(token_id)
    if text is None or text == "":
        return str(token_id)
    # Escape whitespace for single-line table cells
    return repr(text) if (text.strip() == "" or "\n" in text or "\t" in text) else text


def format_topk_table(
    trace: Trace,
    *,
    max_positions: int | None = None,
    tokenizer: Any | None = None,
) -> str:
    """Build a readable multi-line table for one Trace's teacher top-k.

    Columns: pos | sampled (decoded or id) | top-k as ``token=p`` (p = exp(logprob)).
    """
    n = len(trace.output_ids)
    if max_positions is not None:
        n = min(n, max_positions)

    tok_note = "decoded" if tokenizer is not None else "ids-only (no tokenizer)"
    header = (
        f"trace prompt_id={trace.prompt_id!r} domain={trace.domain!r} "
        f"teacher={trace.teacher_model!r} topk={trace.topk} n_out={len(trace.output_ids)} "
        f"[{tok_note}]"
    )
    lines = [header, "-" * min(100, max(len(header), 40))]
    lines.append(f"{'pos':>4}  {'sampled':>16}  top-k candidates (token=prob)")
    lines.append(f"{'----':>4}  {'----------------':>16}  " + "-" * 48)

    for i in range(n):
        oid = trace.output_ids[i]
        sampled_disp = decode_token_id(tokenizer, oid)
        step = trace.steps[i] if i < len(trace.steps) else None
        if step is None:
            lines.append(f"{i:>4}  {sampled_disp:>16}  (missing step)")
            continue
        pairs: list[tuple[int, float]] = []
        for tid, lp in zip(step.token_ids, step.logprobs, strict=False):
            try:
                p = math.exp(float(lp))
            except (OverflowError, ValueError):
                p = 0.0
            pairs.append((int(tid), p))
        pairs.sort(key=lambda x: x[1], reverse=True)
        cells = "  ".join(
            f"{decode_token_id(tokenizer, tid)}={p:.4f}" for tid, p in pairs
        )
        mark = "★" if any(tid == oid for tid, _ in pairs) else " "
        lines.append(f"{i:>4}  {sampled_disp:>16}{mark} {cells}")

    if max_positions is not None and len(trace.output_ids) > max_positions:
        lines.append(f"... ({len(trace.output_ids) - max_positions} more positions omitted)")
    return "\n".join(lines)


def inspect_traces(
    path: str | Path,
    *,
    max_positions: int | None = 64,
    tokenizer_name: str | None = None,
    tokenizer: Any | None = None,
) -> str:
    """Load first trace under ``path`` and return the pretty table string.

    Prefer an explicit ``tokenizer`` object; else try loading ``tokenizer_name``.
    """
    tr = load_first_trace(path)
    tok = tokenizer if tokenizer is not None else try_load_tokenizer(tokenizer_name)
    return format_topk_table(tr, max_positions=max_positions, tokenizer=tok)
