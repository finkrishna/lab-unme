# Lab UnMe: Distilling Kimi K3 Toward Frontier

## 1. Is “Chinese labs distill US models to frontier” even possible?

### Short answer

**Yes, distillation can transfer a large fraction of *behavioral* capability from a stronger teacher to a weaker student — especially for reasoning style, coding habits, and instruction following.**  
**No, distillation alone does not magically invent a full frontier lab.** You still need base model capacity, compute, data mixture skill, post-training, evals, and usually some original pretraining or a strong open base.

### What distillation actually is

Knowledge distillation (KD) trains a **student** to match a **teacher**’s outputs (and sometimes intermediate representations):

| Mode | Teacher signal | Typical use |
|------|----------------|-------------|
| **Response KD** | Final answers | Cheap imitation |
| **Rationale KD** | Chain-of-thought / process traces | Reasoning uplift |
| **Preference KD** | Ranked pairs (better vs worse) | Alignment / style |
| **On-policy KD** | Student samples scored by teacher/verifier | Reduce distribution shift |
| **Feature KD** | Hidden states / logits | Needs weight access |

**Adversarial distillation** (policy term): industrial-scale harvesting of closed API outputs via ToS-violating access (fake accounts, evasion). That is a *data acquisition* method, not a new learning algorithm.

### What the evidence supports

- Distillation is **standard industry practice** (US and China). DeepSeek openly distilled R1 into smaller Qwen/Llama students.
- US labs have accused Chinese labs of large-scale **API extraction** (tens of millions of exchanges). That volume is enough to dominate a post-training corpus for specific skills.
- Measured effect (public analyses): meaningful **benchmark gains on targeted skills** (STEM, code, CoT format). Attribution of *entire* frontier parity to distillation alone is **not cleanly proven** — architecture, data engineering, and efficiency matter a lot (K3’s own scaling claims are about training efficiency, not just stolen traces).
- Student ceiling: a small student **cannot fully absorb** a 2.8T teacher’s world knowledge. Distillation moves you toward the teacher on the **support of the prompt distribution you query**. Outside that support, quality collapses.
- Safety note: adversarial distillation often **erodes alignment** (student copies capability without the full safety stack).

### Realistic transfer function

```
Frontier_student ≈ f(
  base_capacity,          # params, arch, pretrain quality
  teacher_trace_quality,  # CoT, tools, multi-sample
  coverage × difficulty,  # prompt curriculum
  verification,           # filter hallucinations
  post_train_RL,          # verifiable rewards
  system_layer            # tools, search, memory
)
```

**Rule of thumb for Lab UnMe:**

| Goal | Distillation role |
|------|-------------------|
| Match K3 on coding agents | High — if traces + unit tests + long-horizon rollouts |
| Match K3 on general chat | Medium — style/CoT transfer works well |
| Beat K3 overall from a tiny base | Near-impossible without more capacity or better systems |
| Build “UnMe-Frontier” cheaper than pretraining from scratch | **This is the real win** |

### Lab UnMe policy (hard constraints)

We distill **only from authorized sources**:

1. **Open weights** of Kimi K3 (local / self-hosted inference), and/or  
2. **Official paid API** under Moonshot ToS, and/or  
3. **Publicly released** K3 traces / demos.

We do **not** implement fraudulent multi-account scraping of closed US APIs. That is out of scope and not part of this repo.

---

## 2. Lab UnMe goal

**Product:** `UnMe-K3-Distill` — a family of models that approach Kimi K3 on selected frontier axes (code, reasoning, long-horizon agents), trained primarily from K3 teacher traces + verification + RL.

**Not the goal:** claim “we beat GPT/Claude by API theft.”

**Success metrics (capability ledger):**

| Axis | Metric examples | Target vs K3 teacher |
|------|-----------------|----------------------|
| Code | SWE-bench-style local, HumanEval+, LiveCodeBench | ≥ 90% of teacher |
| Math/reason | MATH, AIME-style, process accuracy | ≥ 90% of teacher |
| Agents | Multi-step tool tasks, repo edits | ≥ 85% of teacher |
| Cost | $/query, latency, active params | **Much better** than K3 |
| Safety | Refusal / jailbreak suite | No major regression vs policy |
| General | MMLU-Pro / Arena-hard proxies | ≥ 85% of teacher (soft) |

---

## 3. System architecture

```
                    ┌──────────────────────────┐
                    │  Teacher: Kimi K3         │
                    │  (weights or official API)│
                    └────────────┬─────────────┘
                                 │ traces (prompt, CoT, answer, tools)
                                 ▼
┌──────────────┐    ┌────────────────────────┐    ┌─────────────────┐
│ Curriculum   │───▶│ Synth Lab              │───▶│ Verify Lab      │
│ generators   │    │ multi-sample, CoT,     │    │ code/math/judge │
└──────────────┘    │ tool rollouts          │    └────────┬────────┘
                    └────────────────────────┘             │
                                                           ▼
                                              ┌────────────────────────┐
                                              │ Filtered corpus        │
                                              │ SFT / prefs / RL data  │
                                              └────────────┬───────────┘
                                                           ▼
                                              ┌────────────────────────┐
                                              │ Train Lab              │
                                              │ SFT → DPO/KTO → RL     │
                                              │ student: UnMe base     │
                                              └────────────┬───────────┘
                                                           ▼
                                              ┌────────────────────────┐
                                              │ Eval & Promotion       │
                                              │ capability ledger      │
                                              └────────────────────────┘
```

### Student choices (pick one path)

| Track | Student init | When |
|-------|--------------|------|
| **A. Mid-size open base** | Qwen3 / Llama-class 70B–100B+ dense or MoE | Fastest path to strong UnMe |
| **B. From K3 slice** | Distill into smaller MoE / LoRA vertices on K3 | If you can host K3 |
| **C. Small edge** | 7B–32B | Cost/latency products; not full frontier |

Default pipeline config: **Track A** (train a strong open base on K3 traces).

---

## 4. Pipeline stages

### Stage 0 — Teacher harness
- Unified client: local vLLM/SGLang **or** official API
- Capture: tokens, logprobs (if available), tool calls, thinking blocks
- Rate limits, cost accounting, seed control

### Stage 1 — Curriculum
Domain packs:
- `code` — repo tasks, algorithms, debugging
- `math` — contest + process supervision
- `stem` — science reasoning
- `agent` — tool use, multi-step plans
- `general` — instruction following, writing
- `safety` — policy-preserving pairs

Each pack: seed prompts + generator prompts that ask K3 to **create harder variants**.

### Stage 2 — Synthesis
For each prompt:
1. Sample \(n\) teacher completions (temperature sweep)
2. Optional self-consistency / majority
3. Optional teacher critique pass
4. Emit `Trace` records (JSONL)

### Stage 3 — Verification (the real moat)
- **Code:** unit tests, typecheck, sandbox exec
- **Math:** exact match / sympy / numerical
- **Agent:** environment success
- **Open-ended:** multi-judge rubric (can use K3 as judge *with care* + cross-checks)
- Drop failures; keep preference pairs (pass > fail)

### Stage 4 — Training
1. **SFT** on verified traces (answer + optional full CoT)
2. **Preference** (DPO/KTO/IPO) on ranked pairs
3. **RL** on verifiable rewards (GRPO/PPO-style) where sandboxes exist
4. **Rehearsal mix** to avoid catastrophic forgetting of general skills

### Stage 5 — Eval & promote
- Automated ledger vs frozen teacher snapshot
- Promote checkpoint only if gates pass

---

## 5. Data scale guidance (order of magnitude)

| Phase | Tokens / examples | Notes |
|-------|-------------------|-------|
| Bootstrap SFT | 100M–2B tokens verified | Quality >> quantity |
| Preference | 50k–500k pairs | Domain-balanced |
| RL | 10k–200k rollouts | Verifiable tasks only |
| Targeted distillation | Dense on failure modes | From eval gaps |

For comparison: public commentary estimated industrial adversarial harvests at **hundreds of billions of tokens** across labs — that is “national lab scale,” not a weekend project. UnMe should optimize **verified density**, not raw scrape volume.

---

## 6. What we implement in this repo

A **runnable research pipeline skeleton**:

- Config-driven stages
- Teacher client abstraction (mock + OpenAI-compatible + local)
- Curriculum + synth workers
- Pluggable verifiers
- Training job specs (HuggingFace / torchtune style launchers)
- Eval harness stubs + capability ledger schema
- CLI: `unme run --stage ...`

This is designed so Lab UnMe can plug in real K3 endpoints and GPU clusters without rewriting the graph.

---

## 7. Risks and non-goals

| Risk | Mitigation |
|------|------------|
| Student copies teacher errors | Hard verifiers; don’t trust teacher on unverifiable claims |
| License / ToS violation | Official access only; document license |
| Safety stripping | Keep safety mix; eval gates |
| Overfit to synthetic style | Mix human/public data you legally hold |
| “We are frontier” marketing | Report teacher-relative scores, not hype |

---

## 8. 90-day plan

| Weeks | Milestone |
|-------|-----------|
| 1–2 | Teacher harness + mock end-to-end dry run |
| 3–4 | Code+math verifiers; first 10k verified traces |
| 5–6 | SFT student v0; eval ledger v0 |
| 7–8 | Preference training; close gap on code |
| 9–10 | Agent sandboxes + RL |
| 11–12 | UnMe-K3-Distill-beta + public report: % of teacher |

---

## 9. Bottom line

- **Is distillation-to-frontier possible?** *Approaching* a teacher on **targeted capabilities** — yes. *Fully replacing* pretraining + research culture with API theft — overstated.
- **Lab UnMe strategy:** treat Kimi K3 as a **legal teacher oracle**, build a **verification-first** synth factory, and train a student that wins on **cost × capability**, not on mythology.
