"""Pretty-print teacher top-k distributions from Trace JSONL.

Traces store token *ids* and logprobs (not surface strings). We show each sampled
output token id and its top-k candidates with linear probabilities ``exp(logprob)``.
"""

from __future__ import annotations

import math
from pathlib import Path

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


def format_topk_table(trace: Trace, *, max_positions: int | None = None) -> str:
    """Build a readable multi-line table for one Trace's teacher top-k.

    Columns: pos | sampled_id | top-k as ``id=p`` (p = exp(logprob)), ranked.
    """
    n = len(trace.output_ids)
    if max_positions is not None:
        n = min(n, max_positions)

    header = (
        f"trace prompt_id={trace.prompt_id!r} domain={trace.domain!r} "
        f"teacher={trace.teacher_model!r} topk={trace.topk} n_out={len(trace.output_ids)}"
    )
    lines = [header, "-" * min(100, max(len(header), 40))]
    lines.append(f"{'pos':>4}  {'sampled':>10}  top-k candidates (id=prob)")
    lines.append(f"{'----':>4}  {'----------':>10}  " + "-" * 48)

    for i in range(n):
        oid = trace.output_ids[i]
        step = trace.steps[i] if i < len(trace.steps) else None
        if step is None:
            lines.append(f"{i:>4}  {oid:>10}  (missing step)")
            continue
        pairs: list[tuple[int, float]] = []
        for tid, lp in zip(step.token_ids, step.logprobs, strict=False):
            try:
                p = math.exp(float(lp))
            except (OverflowError, ValueError):
                p = 0.0
            pairs.append((int(tid), p))
        # Rank by prob descending for display
        pairs.sort(key=lambda x: x[1], reverse=True)
        cells = "  ".join(f"{tid}={p:.4f}" for tid, p in pairs)
        # Mark sampled token
        mark = "★" if any(tid == oid for tid, _ in pairs) else " "
        lines.append(f"{i:>4}  {oid:>10}{mark} {cells}")

    if max_positions is not None and len(trace.output_ids) > max_positions:
        lines.append(f"... ({len(trace.output_ids) - max_positions} more positions omitted)")
    return "\n".join(lines)


def inspect_traces(
    path: str | Path,
    *,
    max_positions: int | None = 64,
) -> str:
    """Load first trace under ``path`` and return the pretty table string."""
    tr = load_first_trace(path)
    return format_topk_table(tr, max_positions=max_positions)
