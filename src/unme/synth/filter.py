"""Strict synthetic-trace filter. Quality here caps final student quality."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import orjson

from unme.schemas import FilterVerdict, Trace
from unme.verify.verifiers import Verifier

# Heuristic knobs (strict defaults)
MIN_OUTPUT_TOKENS = 1
MAX_OUTPUT_TOKENS = 8192
MIN_INPUT_TOKENS = 1
MAX_INPUT_TOKENS = 32768
# Reject if any token id repeats consecutively more than this many times
MAX_CONSECUTIVE_REPEAT = 32
# Reject if unique token fraction among outputs is below this
MIN_UNIQUE_RATIO = 0.15


def _read_traces(traces_dir: str | Path) -> list[tuple[Path, Trace]]:
    root = Path(traces_dir)
    if root.is_file():
        files = [root]
    else:
        files = sorted(root.glob("*.jsonl"))
    out: list[tuple[Path, Trace]] = []
    for fp in files:
        with fp.open("rb") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                    out.append((fp, Trace.model_validate(data)))
                except (orjson.JSONDecodeError, ValueError, TypeError):
                    # Corrupt / non-schema rows are dropped silently at read time.
                    continue
    return out


def _prompt_text(trace: Trace) -> str:
    return (
        trace.meta.get("prompt")
        or trace.meta.get("prompt_text")
        or ""
    )


def _output_text(trace: Trace) -> str:
    return (
        trace.meta.get("output")
        or trace.meta.get("output_text")
        or trace.meta.get("completion")
        or ""
    )


def _is_repetitive(ids: Sequence[int]) -> bool:
    if not ids:
        return True
    # consecutive run
    run = 1
    for i in range(1, len(ids)):
        if ids[i] == ids[i - 1]:
            run += 1
            if run > MAX_CONSECUTIVE_REPEAT:
                return True
        else:
            run = 1
    # low uniqueness
    uniq = len(set(ids))
    return uniq / max(len(ids), 1) < MIN_UNIQUE_RATIO and len(ids) >= 20


def _degenerate_output_text(text: str) -> bool:
    if not text or not text.strip():
        # empty text is ok if we only have token ids — checked separately
        return False
    t = text.strip()
    if len(set(t)) <= 2 and len(t) >= 20:
        return True
    # same character block
    return len(t) >= 40 and t[:20] * (len(t) // 20) in t


def _heuristic_reasons(trace: Trace) -> list[str]:
    reasons: list[str] = []
    n_out = len(trace.output_ids)
    n_in = len(trace.input_ids)
    if n_out < MIN_OUTPUT_TOKENS:
        reasons.append("output_too_short")
    if n_out > MAX_OUTPUT_TOKENS:
        reasons.append("output_too_long")
    if n_in < MIN_INPUT_TOKENS:
        reasons.append("input_too_short")
    if n_in > MAX_INPUT_TOKENS:
        reasons.append("input_too_long")
    if len(trace.steps) != n_out:
        reasons.append("steps_misaligned")  # schema usually prevents this
    if _is_repetitive(trace.output_ids):
        reasons.append("repetitive_tokens")
    if _degenerate_output_text(_output_text(trace)):
        reasons.append("degenerate_text")
    if n_out > 0 and len(set(trace.output_ids)) == 1:
        reasons.append("single_token_collapse")
    return reasons


def _apply_verifiers(
    trace: Trace,
    verifiers: Sequence[Verifier],
) -> tuple[list[str], dict[str, float]]:
    reasons: list[str] = []
    scores: dict[str, float] = {}
    prompt = _prompt_text(trace)
    output = _output_text(trace)
    if not verifiers:
        return reasons, scores
    if not prompt and not output:
        # token-only traces: skip text verifiers rather than auto-fail hard domains
        # unless a verifier is present — then we cannot grade → drop
        reasons.append("missing_text_for_verifiers")
        return reasons, scores

    for v in verifiers:
        name = type(v).__name__
        # Domain routing: only run matching verifier when domain is known
        domain = (trace.domain or "").lower()
        if name == "CodeVerifier" and domain and domain not in {"cs", "code", "coding"}:
            continue
        if name == "MathVerifier" and domain and domain not in {"math", "stem"}:
            continue
        try:
            ok, score = v.check(prompt, output)
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            ok, score = False, 0.0
            reasons.append(f"{name}_error:{type(exc).__name__}")
            scores[name] = 0.0
            continue
        scores[name] = float(score)
        if not ok:
            reasons.append(f"{name}_fail")
    return reasons, scores


def filter_traces(
    traces_dir: str | Path,
    out_dir: str | Path,
    verifiers: Sequence[Verifier] | None = None,
    *,
    verdicts_name: str = "verdicts.jsonl",
    kept_name: str = "kept.jsonl",
) -> list[FilterVerdict]:
    """Read Trace JSONL from ``traces_dir``, filter strictly, write kept traces.

    Writes:
      - ``out_dir/kept.jsonl`` — kept Trace rows
      - ``out_dir/verdicts.jsonl`` — FilterVerdict per considered trace

    Dedup: at most one kept trace per ``prompt_id`` (highest mean verifier score,
    then longest non-repetitive output).
    """
    verifiers = list(verifiers or [])
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    loaded = _read_traces(traces_dir)
    # score all
    scored: list[tuple[Trace, FilterVerdict, float]] = []
    for _fp, trace in loaded:
        reasons = _heuristic_reasons(trace)
        v_reasons, scores = _apply_verifiers(trace, verifiers)
        reasons.extend(v_reasons)
        keep = len(reasons) == 0
        mean_score = sum(scores.values()) / len(scores) if scores else (1.0 if keep else 0.0)
        verdict = FilterVerdict(
            prompt_id=trace.prompt_id,
            keep=keep,
            reasons=reasons,
            scores=scores,
        )
        scored.append((trace, verdict, mean_score))

    # Dedup by prompt_id among keepers only; also mark losers
    by_pid: dict[str, list[tuple[Trace, FilterVerdict, float]]] = defaultdict(list)
    for item in scored:
        by_pid[item[0].prompt_id].append(item)

    final_verdicts: list[FilterVerdict] = []
    kept_traces: list[Trace] = []

    for items in by_pid.values():
        keepers = [(t, v, s) for t, v, s in items if v.keep]
        if not keepers:
            for _t, v, _s in items:
                final_verdicts.append(v)
            continue
        # rank: score desc, then len(output_ids) desc
        keepers.sort(key=lambda x: (x[2], len(x[0].output_ids)), reverse=True)
        winner_t, winner_v, _ = keepers[0]
        kept_traces.append(winner_t)
        final_verdicts.append(winner_v)
        # demote duplicates
        for t, v, _s in keepers[1:]:
            final_verdicts.append(
                FilterVerdict(
                    prompt_id=t.prompt_id,
                    keep=False,
                    reasons=[*v.reasons, "dedup_prompt_id"],
                    scores=v.scores,
                )
            )
        for t, v, _s in items:
            if not v.keep:
                final_verdicts.append(v)

    verdicts_path = out_path / verdicts_name
    kept_path = out_path / kept_name
    with verdicts_path.open("wb") as vf:
        for v in final_verdicts:
            vf.write(orjson.dumps(v.model_dump(mode="json")))
            vf.write(b"\n")
    with kept_path.open("wb") as kf:
        for t in kept_traces:
            kf.write(orjson.dumps(t.model_dump(mode="json")))
            kf.write(b"\n")

    return final_verdicts


def iter_kept(out_dir: str | Path, kept_name: str = "kept.jsonl") -> Iterable[Trace]:
    path = Path(out_dir) / kept_name
    if not path.exists():
        return
        yield  # pragma: no cover
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Trace.model_validate(orjson.loads(line))
