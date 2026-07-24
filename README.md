# Lab UnMe — Distillation Pipeline

Compress an open-weight **teacher** (Kimi K3, 2.7T MoE) into a smaller, cheaper
**student** that *matches* the teacher's quality within a target deployment envelope.

**Goal:** frontier-matching-cheaper (not surpass). Distillation-heavy. No RLVR yet —
the `verify/` module is wired as a data filter now and is the seam where RLVR plugs
in later if the goal changes to "surpass on verifiable axes."

**Constraints honored:** open-weight teachers only. True logit + hidden-state
distillation (needs teacher probabilities → needs open weights). No closed-API scraping.

## The ceiling you're working under

Pure distillation asymptotically *approaches* the teacher; it does not exceed it.
Everything here is tuned to reach the teacher at lower cost/size. Surpassing the
teacher requires the `verify/` + RLVR path (deferred by design).

## Pipeline stages

```
teacher/   Stage 1  Generate teacher outputs + retain top-k logits (+ hidden states)
synth/     Stage 1b Filter traces: correctness (verify/) + quality + dedup
data/               Load filtered traces into a distillation dataset
train/     Stage 2  Student objective = KL(top-k logits) + hidden-state match + hard-label CE
eval/      Stage 0  Per-domain eval suite + regression GATE (built first, guards everything)
registry            Promote a checkpoint only if it passes the gate
verify/             Verifiers (code/math/consistency). Used as a filter now; RLVR seam later.
```

Sequencing: **build `eval/` first**, then `teacher/` → `synth/` → `train/`, prove the
loop on one domain, then scale. The two make-or-break stages are `synth/` filtering
(bad synthetic data silently caps you below the teacher) and, later, `verify/` design.

## Quick start

```bash
pip install -r requirements.txt
pip install -e .
python scripts/smoke_test.py          # tiny end-to-end run on tiny HF models
```

Point `configs/distill.yaml` at real checkpoints to scale up.
