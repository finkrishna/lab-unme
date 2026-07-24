#!/usr/bin/env python3
"""Probe a teacher /completions endpoint for per-token top-k logprobs.

Uses ``unme.teacher.client.TeacherLogitsClient`` (no reimplemented HTTP).
Exit 0 on PASS, non-zero on FAIL. Never prints API keys.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open() as f:
        return yaml.safe_load(f) or {}


def _redact(obj: object) -> object:
    """Drop anything that looks like a secret before dumping."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if any(s in lk for s in ("api_key", "authorization", "token", "password", "secret")):
                out[k] = "***"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj]
    return obj


def main(argv: list[str] | None = None) -> int:
    # Ensure repo src is importable when run as scripts/probe_teacher.py
    root = _repo_root()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    parser = argparse.ArgumentParser(
        description="Probe teacher completions for top-k logprobs (one short call)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "distill.probe.yaml",
        help="YAML config (default: configs/distill.probe.yaml)",
    )
    parser.add_argument("--base-url", type=str, default=None, help="Override teacher.base_url")
    parser.add_argument("--model", type=str, default=None, help="Override teacher.model")
    parser.add_argument("--api-key", type=str, default=None, help="Override teacher.api_key (not logged)")
    parser.add_argument("--topk", type=int, default=None, help="Override teacher.topk")
    parser.add_argument(
        "--prompt",
        type=str,
        default="Say hello in one short word.",
        help="Prompt text for the single probe completion",
    )
    parser.add_argument("--max-tokens", type=int, default=5, help="max_tokens for the probe call")
    parser.add_argument("--timeout", type=float, default=None, help="HTTP timeout seconds")
    args = parser.parse_args(argv)

    cfg: dict = {}
    if args.config.exists():
        cfg = _load_yaml(args.config)
    tcfg = cfg.get("teacher") or {}

    base_url = (args.base_url or tcfg.get("base_url") or "").strip()
    model = (args.model or tcfg.get("model") or "").strip()
    topk = int(args.topk if args.topk is not None else tcfg.get("topk") or 20)
    temperature = float(tcfg.get("temperature") or 1.0)
    timeout = float(args.timeout if args.timeout is not None else tcfg.get("timeout") or 60.0)
    api_key = (
        args.api_key
        if args.api_key is not None
        else (tcfg.get("api_key") or os.environ.get("MOONSHOT_API_KEY") or os.environ.get("TEACHER_API_KEY") or "")
    )

    if not base_url:
        print("FAIL: teacher.base_url is missing (set in config or --base-url)", file=sys.stderr)
        return 2
    if not model:
        print("FAIL: teacher.model is missing (set in config or --model)", file=sys.stderr)
        return 2

    from unme.teacher.client import TeacherLogitsClient

    print(f"probe base_url={base_url!r} model={model!r} topk={topk} max_tokens={args.max_tokens}")
    print("(api_key: set)" if api_key else "(api_key: empty)")

    try:
        with TeacherLogitsClient(
            base_url=base_url,
            model=model,
            topk=topk,
            temperature=temperature,
            api_key=api_key or "",
            max_tokens=args.max_tokens,
            timeout=timeout,
        ) as client:
            completion = client.complete(args.prompt, max_tokens=args.max_tokens)
    except Exception as exc:  # noqa: BLE001 — operator tool: surface any transport error
        print(f"FAIL: request error: {type(exc).__name__}: {exc}")
        return 1

    steps = completion.steps or []
    counts = [len(s.top_logprobs) for s in steps]
    n_steps = len(steps)
    has_topk = n_steps > 0 and all(c >= 1 for c in counts)

    if has_topk:
        print("PASS: top_logprobs present on every generated position")
        print(f"  positions={n_steps} top_logprobs_per_position={counts}")
        print(f"  requested_topk={topk} min_returned={min(counts)} max_returned={max(counts)}")
        # sample first position only (no secrets)
        first = steps[0]
        sample = [(t.token[:40], round(t.logprob, 4)) for t in first.top_logprobs[: min(5, len(first.top_logprobs))]]
        print(f"  first_position_sample={sample}")
        return 0

    print("FAIL: top_logprobs missing or empty")
    print(f"  positions={n_steps} top_logprobs_per_position={counts}")
    print("  raw choice (redacted):")
    print(json.dumps(_redact(completion.raw), indent=2, default=str)[:8000])
    if completion.text:
        text_preview = completion.text[:200].replace("\n", "\\n")
        print(f"  completion_text_preview={text_preview!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
