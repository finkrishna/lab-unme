"""OpenAI-compatible teacher client with per-token top-k logprobs.

This module is separate from the legacy ``unme.teacher`` harness (which targets
chat-style generation). The distillation ``Trace`` schema (``unme.schemas.Trace``)
needs the teacher's **per-token top-k logprobs**, so we hit the OpenAI-style
``/completions`` endpoint with ``logprobs=topk`` — exactly what vLLM/SGLang and the
Moonshot completions API expose.

Authorized use only: Moonshot official API with your key, or self-hosted K3 open
weights via vLLM/SGLang. No multi-account scraping.

The client is transport-injectable: pass ``transport=<httpx.MockTransport ...>`` to
run tests against a canned response without any live network call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Self

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class TopLogprob:
    """One ranked teacher candidate at a position."""

    token: str
    logprob: float


@dataclass
class StepLogprobs:
    """Per-token teacher distribution at ONE generated position."""

    text: str  # the sampled token's literal text
    logprob: float  # log-prob of the sampled token
    top_logprobs: list[TopLogprob] = field(default_factory=list)


@dataclass
class TeacherCompletion:
    """Raw teacher completion with sparse per-step top-k logprobs.

    Token ids are NOT resolved here — the completions API returns token strings.
    ``unme.teacher.generate`` turns these into ``unme.schemas.StepLogits`` token ids
    via the configured tokenizer.
    """

    text: str
    model: str
    steps: list[StepLogprobs]  # one per generated token
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class TeacherLogitsClient:
    """OpenAI-compatible completions client capturing top-k logprobs.

    Args:
        base_url: e.g. "http://localhost:8000/v1" (Moonshot or self-hosted vLLM).
        api_key: bearer token; empty when self-hosting without auth.
        model: served model name.
        topk: number of top logprobs requested per token (maps to OpenAI ``logprobs``).
        temperature: sampling temperature sent on every request.
        max_tokens: default generation budget.
        timeout: httpx request timeout in seconds.
        transport: optional ``httpx.BaseTransport`` for mocking (no live network).
        extra_headers: extra HTTP headers.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        topk: int = 20,
        temperature: float = 1.0,
        api_key: str | None = None,
        max_tokens: int = 1024,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.topk = topk
        self.temperature = temperature
        self.api_key = api_key or ""
        self.max_tokens = max_tokens
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return self._client.headers  # type: ignore[return-value]

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        r = self._client.post("/completions", json=body)
        r.raise_for_status()
        return r.json()

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int | None = None) -> TeacherCompletion:
        """Call the teacher for ``prompt`` with top-k logprobs.

        vLLM/Moonshot "completions" response shape (of interest here):
            choices[0].text
            choices[0].logprobs.content  -> list of {
                token, logprob, top_logprobs: list of {token, logprob}
            }
            usage, model, choices[0].finish_reason
        The older "logprobs.tokens" format is also tolerated.
        """
        if system:
            full_prompt = f"{system}\n\n{prompt}"
        else:
            full_prompt = prompt

        body: dict[str, Any] = {
            "model": self.model,
            "prompt": full_prompt,
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "logprobs": self.topk,
        }
        data = self._post(body)

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"teacher returned no choices: {data}")
        choice = choices[0]
        text = choice.get("text", "")
        # OpenAI / vLLM: choice.logprobs; llama.cpp may use completion_probabilities.
        steps = _parse_logprobs(choice.get("logprobs"), self.topk)
        if not steps:
            steps = _parse_logprobs(choice.get("completion_probabilities"), self.topk)
        return TeacherCompletion(
            text=text,
            model=data.get("model", self.model),
            steps=steps,
            usage=data.get("usage") or {},
            finish_reason=choice.get("finish_reason"),
            raw=choice,
        )


def _candidate_token_logprob(cand: dict[str, Any]) -> tuple[str, float]:
    """Normalize one top-k candidate dict → (token_str, logprob).

    Accepts aliases used by OpenAI (`token`/`logprob`) and llama.cpp
    (`tok_str`/`prob` as linear probability, converted to log).
    """
    token = cand.get("token")
    if token is None:
        token = cand.get("tok_str")
    if token is None:
        token = cand.get("content")
    token_str = str(token) if token is not None else ""

    if "logprob" in cand and cand["logprob"] is not None:
        return token_str, float(cand["logprob"])
    # Linear probability keys → log-space (floor for numerical safety).
    for key in ("prob", "probs"):
        if key in cand and cand[key] is not None and not isinstance(cand[key], (list, dict)):
            p = float(cand[key])
            return token_str, math.log(max(p, 1e-12))
    return token_str, 0.0


def _top_candidates_from_entry(entry: dict[str, Any], topk: int) -> list[TopLogprob]:
    """Extract top-k list from an entry; prefers OpenAI names, then llama aliases."""
    raw_list = entry.get("top_logprobs")
    if raw_list is None:
        raw_list = entry.get("top_probs")
    if raw_list is None:
        raw_list = entry.get("probs")
    if not raw_list:
        return []
    out: list[TopLogprob] = []
    for item in list(raw_list)[:topk]:
        if isinstance(item, dict):
            tok, lp = _candidate_token_logprob(item)
            out.append(TopLogprob(token=tok, logprob=lp))
        # skip non-dict entries
    return out


def _step_from_entry(entry: dict[str, Any], topk: int) -> StepLogprobs | None:
    """One position: sampled token + top-k candidates (OpenAI or llama naming)."""
    if not isinstance(entry, dict):
        return None
    # Sampled token may live under token / tok_str / content
    text = entry.get("token")
    if text is None:
        text = entry.get("tok_str")
    if text is None:
        text = entry.get("content")
    text_str = str(text) if text is not None else ""

    if "logprob" in entry and entry["logprob"] is not None:
        lp = float(entry["logprob"])
    elif "prob" in entry and entry["prob"] is not None and not isinstance(entry["prob"], (list, dict)):
        lp = math.log(max(float(entry["prob"]), 1e-12))
    else:
        # Fall back to first top candidate's logprob if present
        tops = _top_candidates_from_entry(entry, topk)
        lp = tops[0].logprob if tops else 0.0
        if not text_str and tops:
            text_str = tops[0].token
        return StepLogprobs(text=text_str, logprob=lp, top_logprobs=tops)

    tops = _top_candidates_from_entry(entry, topk)
    return StepLogprobs(text=text_str, logprob=lp, top_logprobs=tops)


def _parse_logprobs(
    logprobs: dict[str, Any] | list[Any] | None,
    topk: int,
) -> list[StepLogprobs]:
    """Parse OpenAI logprobs shapes first; also tolerate llama.cpp aliases/shapes.

    Supported:
      - Modern OpenAI: ``{content: [{token, logprob, top_logprobs: [...]}]}``
      - Aliases: ``top_probs``, ``prob`` (linear→log), ``tok_str``
      - Legacy OpenAI: tokens / token_logprobs / top_logprobs maps
      - llama.cpp list: ``[{token|content, probs:[{tok_str|token, prob}]}]``
        (also used for ``completion_probabilities``)
    """
    if not logprobs:
        return []

    # llama.cpp: logprobs / completion_probabilities as a list of per-token dicts
    if isinstance(logprobs, list):
        steps: list[StepLogprobs] = []
        for entry in logprobs:
            step = _step_from_entry(entry, topk) if isinstance(entry, dict) else None
            if step is not None:
                steps.append(step)
        return steps

    if not isinstance(logprobs, dict):
        return []

    # Modern OpenAI first: content: [ {token, logprob, top_logprobs:[{token, logprob}, ...]} ]
    content = logprobs.get("content")
    if content is not None:
        steps = []
        for entry in content:
            if entry is None:
                continue
            if isinstance(entry, dict):
                step = _step_from_entry(entry, topk)
                if step is not None:
                    steps.append(step)
        return steps

    # Legacy: tokens: [...], token_logprobs: [...], top_logprobs: [ {token: logprob} ]
    tokens = logprobs.get("tokens") or []
    if tokens:
        tlp = logprobs.get("token_logprobs") or []
        tops = logprobs.get("top_logprobs") or logprobs.get("top_probs") or []
        steps = []
        for i, tok in enumerate(tokens):
            top_raw = tops[i] if i < len(tops) and tops[i] else {}
            top: list[TopLogprob] = []
            if isinstance(top_raw, dict):
                # map token -> logprob OR token -> linear prob
                for k, v in list(top_raw.items())[:topk]:
                    # values are typically already logprobs in legacy OpenAI;
                    # if key looks like linear mass in (0,1] and all positive small, still treat as log
                    # unless the container was named top_probs — then convert.
                    use_linear = isinstance(tops, list) and logprobs.get("top_probs") is tops
                    if use_linear:
                        top.append(TopLogprob(token=str(k), logprob=math.log(max(float(v), 1e-12))))
                    else:
                        top.append(TopLogprob(token=str(k), logprob=float(v)))
            elif isinstance(top_raw, list):
                for item in top_raw[:topk]:
                    if isinstance(item, dict):
                        t, lp = _candidate_token_logprob(item)
                        top.append(TopLogprob(token=t, logprob=lp))
            lp = float(tlp[i]) if i < len(tlp) and tlp[i] is not None else 0.0
            steps.append(StepLogprobs(text=str(tok), logprob=lp, top_logprobs=top))
        return steps

    return []
