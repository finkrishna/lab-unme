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
        steps = _parse_logprobs(choice.get("logprobs"), self.topk)
        return TeacherCompletion(
            text=text,
            model=data.get("model", self.model),
            steps=steps,
            usage=data.get("usage") or {},
            finish_reason=choice.get("finish_reason"),
            raw=choice,
        )


def _parse_logprobs(logprobs: dict[str, Any] | None, topk: int) -> list[StepLogprobs]:
    """Tolerate both the modern "content" shape and the legacy "tokens" shape."""
    if not logprobs:
        return []
    # Modern: content: [ {token, logprob, top_logprobs:[{token, logprob}, ...]} ]
    content = logprobs.get("content")
    if content is not None:
        steps: list[StepLogprobs] = []
        for entry in content:
            if entry is None:
                continue
            top = [
                TopLogprob(token=str(t.get("token", "")), logprob=float(t.get("logprob", 0.0)))
                for t in (entry.get("top_logprobs") or [])[:topk]
            ]
            steps.append(
                StepLogprobs(
                    text=str(entry.get("token", "")),
                    logprob=float(entry.get("logprob", 0.0)),
                    top_logprobs=top,
                )
            )
        return steps
    # Legacy: tokens: [...], token_logprobs: [...], top_logprobs: [ {token: logprob} ]
    tokens = logprobs.get("tokens") or []
    tlp = logprobs.get("token_logprobs") or []
    tops = logprobs.get("top_logprobs") or []
    steps = []
    for i, tok in enumerate(tokens):
        top_map = tops[i] if i < len(tops) and tops[i] else {}
        top = [TopLogprob(token=k, logprob=float(v)) for k, v in list(top_map.items())[:topk]]
        lp = float(tlp[i]) if i < len(tlp) and tlp[i] is not None else 0.0
        steps.append(StepLogprobs(text=tok, logprob=lp, top_logprobs=top))
    return steps
