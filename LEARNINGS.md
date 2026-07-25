# Lab UnMe — Learnings

A retrospective on what building and running the distillation pipeline actually taught
us — especially where it *fails*, which turned out to be the more valuable knowledge.

---

## What we built (and it works)

Two complete, well-engineered pipelines came out of this work:

1. **Distill** (`lab_unme/`) — a full open-weight distillation pipeline: teacher client
   (top-k logprobs), synthetic trace generation, code-execution filtering, top-k KL +
   hidden-state losses, a real training loop, functional (code-exec) evaluation, a
   regression gate, registry, and a `unme run` CLI. Certified end-to-end (60+ tests
   green), runnable fully locally on a laptop for $0 via a small `llama.cpp` teacher.
2. **Indify** (prior effort) — an SFT / DPO / LoRA adaptation pipeline for localizing a
   large OSS model to Indian context.

**Neither pipeline is the problem.** The engineering is sound. The limitation we hit is
one of **resourcing** — data and compute spend — not code.

---

## The pivotal experiment

We ran a genuine generalization test on the Distill pipeline: train a small student on
one set of coding tasks, evaluate on a **disjoint held-out set** it never saw, and grade
**functionally** (does the generated code pass the asserts).

| Model | Held-out score (20 tasks) |
|---|---|
| `Qwen2.5-0.5B` (base) distilled on 47 traces | **0 / 20** — it parroted asserts, never wrote functions |
| `Qwen2.5-0.5B-Instruct` distilled on 47 traces | **1 / 20** |
| `Qwen2.5-0.5B-Instruct`, **untrained baseline** | **14 / 20** |

The result that mattered was the last row. **The untrained model already scored 14/20.
Our distillation dragged it down to 1/20 — a 14× degradation.** The training didn't
teach it to code; it *destroyed* a model that was already good.

Note we were *one step* from reporting "1/20 = generalization success!" — until we
measured the baseline. Without the baseline, degradation looks like progress.

---

## The lessons

1. **Always measure the untrained baseline.** It is the single most important number in
   any training run. "Loss went down" and "the model got better" are unrelated claims;
   our beautiful falling loss curve (3.5 → 1.3) was the model *overfitting harder*.

2. **Catastrophic forgetting is the default, not the exception** — for small models on
   narrow data with any weight-modifying method (SFT, DPO, LoRA, distillation).

3. **"Protect the eval score with LoRA" is eval-hacking.** The baseline is a *diagnostic*
   ("did I break it?"), never a *target* ("tune the recipe until the number holds").
   Optimizing to preserve a proxy is theater.

4. **The capacity-frontier model** (the core insight):
   - Released small models are **already the output of big→small distillation** by the
     labs (Llama 3.2 1B/3B, Gemma 2B, small Qwens are all distilled/pruned from bigger
     siblings, trained on trillions of tokens). They sit **on the efficient frontier** —
     the max useful capability for their size.
   - A point *on* the frontier has **no slack**: every weight is already doing useful
     work, so any gradient update **overwrites** existing capability. Dense packing =
     fragility. That is why the 0.5B collapsed 14 → 1.
   - Fine-tuning can't move a point *up* off the frontier (no free capability to grab).
     It can only slide it *sideways* (trade general skill for narrow) or *down*
     (degrade). On a maxed small model there is almost no "sideways" — so it's "down".
   - **You cannot out-distill the labs on their own turf.** Starting from an
     already-maxed small model and distilling *further* is squeezing a wrung sponge;
     rapid descent to "stupid" is the expected outcome, not bad luck.

---

## The Indify parallel

The same wall, independently. On Indify, **every** adaptation method — instruction
tuning, SFT, DPO, LoRA — degraded the base model, and there we weren't even compressing
(same-size adaptation). The diagnosis there was the mirror image of Distill's:

- **The SFT runs, the knowledge base, and the human-expert data were insufficient** —
  too little, too narrow. With inadequate high-quality data, fine-tuning a capable model
  can only trade away what it has.

Distill failed from *toy-scale spend against a target with no headroom*; Indify failed
from *insufficient data volume/quality*. Two faces of the same underlying constraint:
**you need enough of the right data, and enough room in the model, or weight-modification
is a net negative.**

---

## What would actually work

Both pipelines are ready; to get a *positive* result they need to be pointed at the right
regime:

- **Distill — start bigger, and spend.** Don't distill *from* a maxed 0.5B; distill
  *into* a larger student that still has headroom (e.g. a 7B–32B), *from* a genuinely
  larger/stronger teacher (a frontier model via API), on **lots** of data — i.e. real
  money and real data volume, not weekend toy runs. The pipeline is built for exactly
  this; only the resourcing changes.

- **Indify — feed it enough.** Far more SFT data, a richer KB, and substantially more
  human-expert annotations. Same pipeline, real data investment.

- **For injecting *knowledge* into a small local model — don't use the weights at all.**
  Use **RAG / retrieval**: freeze the good base, put the domain knowledge in a retrieval
  layer, inject relevant chunks at inference. Zero training → zero forgetting. The model
  stays as smart as it was *and* gains the knowledge. This is why the field converged on
  RAG for "make it know my documents," and reserves fine-tuning for *behavior/style*.

---

## Bottom line

We set out to *understand* distillation and ended up understanding its **limits** — the
rarer and more useful knowledge. The pipelines are done and correct. The path to a real
result is **more resource, bigger base, or a non-destructive (retrieval) architecture** —
not more clever training tricks on a small model.

**Threads closed:** Distill and Indify (pipelines complete; blocked on resourcing, not
engineering).
**Next thread:** Data Preparation — and note the honest implication above: the prepared
data should likely be *married by retrieval, not surgery*.
