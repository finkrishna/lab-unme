"""Shared data contracts for the Lab UnMe pipeline.

ARCHITECT-OWNED. The fleet (Grok/GLM) MUST NOT edit this file — every module keys
off these schemas, so drift here breaks everything downstream. If a signature here
is wrong for your task, stop and flag it for review; do not "fix" it locally.

Design notes:
- Teacher logits are stored SPARSE (top-k per position). Dense logits over a large
  vocab are infeasible to persist at scale; top-k captures ~all the mass we distill on.
- Hidden states are large and float-heavy; they are NOT embedded in these models.
  A Trace references an external .npz by path (see `hidden_path`).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    role: Role
    content: str


class Prompt(BaseModel):
    """One input the teacher will be run on. Domain drives eval routing + filtering."""

    id: str
    domain: str  # e.g. "cs", "education", "foreign_affairs", "industry"
    messages: list[Message]
    meta: dict[str, str] = Field(default_factory=dict)


class StepLogits(BaseModel):
    """Teacher's top-k next-token distribution at ONE generated position.

    `token_ids` and `logprobs` are parallel arrays of length k (already log-softmax
    values as returned by the serving endpoint). The chosen/sampled token is recorded
    separately in Trace.output_ids at the same index.
    """

    token_ids: list[int]
    logprobs: list[float]

    @model_validator(mode="after")
    def _same_length(self) -> "StepLogits":
        if len(self.token_ids) != len(self.logprobs):
            raise ValueError("token_ids and logprobs must have equal length (top-k)")
        return self


class Trace(BaseModel):
    """A single teacher generation with per-step top-k logits. The atomic unit of
    the distillation training set. Serialized one-per-line as JSONL via orjson."""

    prompt_id: str
    domain: str
    teacher_model: str
    input_ids: list[int]            # tokenized prompt
    output_ids: list[int]          # tokenized teacher continuation (the hard labels)
    steps: list[StepLogits]        # len == len(output_ids); top-k soft targets
    hidden_path: Optional[str] = None  # path to .npz of teacher hidden states, if emitted
    topk: int
    temperature: float
    meta: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _steps_align(self) -> "Trace":
        if len(self.steps) != len(self.output_ids):
            raise ValueError("steps must align 1:1 with output_ids")
        return self


class FilterVerdict(BaseModel):
    """Output of synth/ filtering on a Trace. `keep=False` drops it from training."""

    prompt_id: str
    keep: bool
    reasons: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """One domain's score for a candidate student, measured against the teacher."""

    domain: str
    metric: str
    student_score: float
    teacher_score: float

    @property
    def ratio(self) -> float:
        if self.teacher_score == 0:
            return 0.0
        return self.student_score / self.teacher_score


class GateReport(BaseModel):
    """Regression gate outcome. A candidate promotes only if `passed` is True."""

    candidate: str
    results: list[EvalResult]
    regression_floor: float

    @property
    def passed(self) -> bool:
        # No domain may fall below floor * teacher.
        return all(r.ratio >= self.regression_floor for r in self.results)
