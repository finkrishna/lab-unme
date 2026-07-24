# Lab UnMe — real-run operator runbook

This guide walks through serving **Kimi K3** as the teacher, generating sparse
top-k Trace JSONL, filtering, distilling a **shared-tokenizer** student, evaluating,
and promoting only if the regression gate passes.

Scaffold / CI smoke continues to use `configs/distill.yaml` (tiny HF models).
**Production-shaped runs use `configs/distill.real.yaml`.**
**Probe / step validation uses `configs/distill.probe.yaml` (3-prompt slice).**

---

## Step 1/2 validation (before a full real run)

Confirm the served teacher returns **per-token top-k logprobs** and that the short
pipeline path works offline of heavy training.

**Step 1 — teacher logprobs probe** (one short `/completions` call via
`TeacherLogitsClient`; exit 0 = PASS):

```bash
# Edit configs/distill.probe.yaml: teacher.base_url / model (+ api_key env if needed)
python scripts/probe_teacher.py --config configs/distill.probe.yaml
```

Optional flags: `--base-url`, `--model`, `--topk`, `--max-tokens 5`, `--api-key` (never printed).

**Step 2 — short `unme run` without train** (uses the 3-line prompt set at
`data/prompts/pilot_probe.jsonl`; full pilot remains `data/prompts/pilot.jsonl`):

```bash
unme run --config configs/distill.probe.yaml --skip-train --candidate probe
```

Notes:

- Probe config points `data.prompts` at **`pilot_probe.jsonl`** (3 lines). For a
  full curriculum, switch to `pilot.jsonl` or set a larger file in config (there is
  no generate `--limit` flag yet — use the probe subset file instead).
- Step 2 still requires a live `teacher.base_url` unless you pass `--skip-generate`
  and already have Trace JSONL under `data/traces`.
- Placeholder eval may refuse promote under `--skip-train`; that is expected until
  real suite scores are recorded.

---

## 1. Serve Kimi K3 with top-k logprobs

The teacher client (`unme.teacher.client.TeacherLogitsClient`) calls:

```http
POST {base_url}/completions
```

with a JSON body shaped like:

```json
{
  "model": "kimi-k3",
  "prompt": "<full prompt text>",
  "temperature": 1.0,
  "max_tokens": 2048,
  "logprobs": 20
}
```

`logprobs` is the **top-k** (integer N). vLLM and SGLang expose this on the
OpenAI-compatible completions API when the model is served with logprobs enabled.

### Response shape the client parses

Preferred (modern) shape — `choices[0].logprobs.content`:

```json
{
  "id": "cmpl-...",
  "model": "kimi-k3",
  "choices": [
    {
      "text": "full completion string",
      "finish_reason": "stop",
      "logprobs": {
        "content": [
          {
            "token": "def",
            "logprob": -0.12,
            "top_logprobs": [
              {"token": "def", "logprob": -0.12},
              {"token": "class", "logprob": -2.4}
            ]
          }
        ]
      }
    }
  ],
  "usage": {"prompt_tokens": 128, "completion_tokens": 64}
}
```

Legacy shape is also tolerated: `logprobs.tokens` + `token_logprobs` +
`top_logprobs` as list of `{token: logprob}` maps.

### Auth

- **Local vLLM/SGLang:** often no auth; leave `teacher.api_key` empty.
- **Moonshot / cloud:** set a bearer token in the process environment and inject it
  into the config (YAML does not expand `${ENV}` by itself). Example wrapper:

```bash
export MOONSHOT_API_KEY=sk-...
# optional: sed or a small launcher that writes api_key into a private local yaml
```

`TeacherLogitsClient` sends `Authorization: Bearer <api_key>` when `api_key` is non-empty.

### Example: vLLM

```bash
# Illustrative — use your K3 weights path / served name
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/kimi-k3 \
  --served-model-name kimi-k3 \
  --host 0.0.0.0 --port 8000
# Ensure completions + logprobs are available on /v1/completions
```

Point `teacher.base_url` at `http://127.0.0.1:8000/v1` in `configs/distill.real.yaml`.

---

## 2. Full pipeline walkthrough

```bash
pip install -e '.[train,dev]'

# Edit configs/distill.real.yaml:
#   teacher.base_url, teacher.model, teacher.tokenizer
#   student.model  (MUST share teacher tokenizer / vocab)
#   teacher.api_key if needed

unme run --config configs/distill.real.yaml --candidate k3-run-001
```

Stage order inside `unme run`:

| Step | Command path | Output |
|------|----------------|--------|
| generate | `teacher.generate` | `data/traces/<domain>.jsonl` (+ optional `.npz` hiddens) |
| filter | `synth.filter` + verifiers | `data/filtered/kept.jsonl`, `verdicts.jsonl` |
| train | `train` CLI / `unme.train.distill` | student checkpoint under `output_dir` |
| eval | registry GateReport (placeholder scores in CLI until suite is wired live) | `outputs/registry/candidates/<name>/gate_report.json` |
| promote | `registry.promote` | succeeds only if `GateReport.passed` |

### Offline / partial runs

- **No teacher endpoint yet:**  
  `unme run --skip-generate --skip-train` reuses existing traces (see CLI message).
- **Traces only, no GPU train:**  
  `unme run --config configs/distill.real.yaml --skip-train`  
  still requires `teacher.base_url` unless you also pass `--skip-generate`.
- **Scaffold CI:** keep using `configs/distill.yaml` + mocks in tests.

---

## 3. Shared-tokenizer invariant

`train/distill.py` asserts:

```text
teacher_top_ids.max() < student_vocab_size
output_ids.max()      < student_vocab_size
```

**Why:** distillation targets are **token ids** from the teacher’s sparse top-k
distribution. If the student uses a different BPE/vocab, those ids address the
wrong symbols — silent corruption, not a loud API error.

**Operator rule:**

1. Tokenize teacher strings with `teacher.tokenizer` (or `teacher.model` if it
   is the HF tokenizer id) during generate.
2. Load a **student** that uses that **same** tokenizer / vocab (or a student
   checkpoint trained on the identical tokenizer).
3. Confirm `student.model` in `distill.real.yaml` after the K3 tokenizer id is
   finalized (TODOs in the config).

`emit_hidden_states` stays `false` and `alpha_hidden: 0.0` until a weights-pass
teacher can emit real hidden tensors; completions logprobs alone cannot.

---

## 4. Regression gate before promote

Eval builds a `unme.schemas.GateReport`:

- Per domain: `EvalResult(domain, metric, student_score, teacher_score)`
- `ratio = student_score / teacher_score` (0 if teacher is 0)
- `GateReport.passed` is **True only if every domain** has  
  `ratio >= regression_floor`  
  (`eval.regression_floor`, default **0.98** in configs)

`registry.promote(candidate)`:

- loads the stored gate report;
- **refuses** with `PromotionError` if `passed` is False;
- writes `promoted.json` only on success.

The CLI `eval` subcommand currently records **placeholder** scores (student 0.0)
so a smoke `unme run` will **not** promote until you wire real student/teacher
suite scores into the registry (or replace the placeholder eval path). That refuse
is expected under incomplete eval wiring; under `--skip-train` the chain treats
promote refusal as non-fatal so generate/filter/dataset-load can still be validated.

Suite layout for the real harness (`eval/harness.py`):

```text
data/eval/<domain>.jsonl
  {"prompt": "...", "answer": "...", "metric": "exact_match", ...}
```

---

## 5. Checklist

- [ ] K3 served; `POST /v1/completions` returns `logprobs.content` with top-k
- [ ] `teacher.base_url` / `model` / `tokenizer` set in `distill.real.yaml`
- [ ] `student.model` shares teacher tokenizer (vocab assert will fire otherwise)
- [ ] Prompts present under `data/prompts/`
- [ ] `unme run --config configs/distill.real.yaml`
- [ ] Inspect `data/filtered/kept.jsonl` pass rate
- [ ] Train checkpoint written; eval ratios ≥ floor on all domains
- [ ] `unme promote <candidate>` succeeds only when the gate passes
