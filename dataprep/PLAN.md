---
artifact_type: plan
component: dataprep
owner: data-engineer
status: draft
as_of: 2026-07-25
sources_checked: 0
---

# Data Preparation Pipeline Plan
**Component:** `lab_unme/dataprep/`
**Purpose:** Transform heterogeneous enterprise file dumps (PDF, DOCX, images, chat exports, code, etc.) into clean, deduplicated, PII-scrubbed, quality-filtered SFT/RLHF-ready datasets for downstream distillation.

---

## 1. Input Contract

**Input:** Directory tree `raw_input/` containing arbitrary files:
- Documents: PDF, DOCX, DOC, PPTX, XLSX, ODT, RTF, TXT, MD
- Images: PNG, JPG, JPEG, TIFF, BMP, HEIC (OCR target)
- Code: `.py, .js, .ts, .java, .go, .rs, .cpp, .h, .sql, .sh, .yaml, .yml, .json, .toml, .ini, .cfg, .conf`
- Chat/Comms: Slack export (JSON), Teams export, Discord JSON, WhatsApp TXT, iMessage, Signal, Telegram JSON
- Email: `.eml`, `.msg`, `.mbox`, `.pst` (Outlook)
- Archives: ZIP, TAR, GZ, 7Z (recursive extraction)
- Other: CSV, TSV, XML, HTML, EPUB

**Assumptions:**
- No prior structure/schema; files dumped by IT export or manual drag-drop
- Total volume: 10K–10M files, 10GB–10TB
- PII/PHI/PCI present; compliance (GDPR, HIPAA, SOC2) required
- Multilingual (EN primary, +ES, FR, DE, JA, ZH, KO, AR, HI, PT)

---

## 2. Output Contract

**Primary output:** `data/processed/{corpus_id}/sft_pairs.jsonl` — one JSONL per corpus version
```jsonl
{"id": "uuid", "source": "raw_input/path/file.pdf#chunk_12", "instruction": "...", "response": "...", "meta": {"lang": "en", "domain": "legal", "pii_risk": "low", "quality_score": 0.92, "tokens": 512}}
```

**Secondary outputs:**
- `manifest.jsonl` — file-level lineage: `{file_hash, path, mime, bytes, pages, chunks, status, errors}`
- `pii_audit.jsonl` — per-chunk PII findings + redaction map
- `quality_report.html` — per-domain stats, score distributions, drop rates
- `dedup_index.faiss` / `dedup_clusters.jsonl` — near-dedup clusters for review

**Registry entry** (appended to `lab_unme/registry/datasets.jsonl`):
```json
{"corpus_id": "corp_legal_v3", "version": 3, "created": "2026-07-25", "source_root": "/mnt/raw/legal_dump", "sft_pairs": 184721, "tokens": 94000000, "domains": ["legal", "compliance", "contracts"], "quality_threshold": 0.85, "pii_policy": "redact_pseudonymize", "status": "ready_for_distill"}
```

---

## 3. Pipeline Stages (DAG)

```
raw_input/
   │
   ▼
[1] DISCOVER & INVENTORY          → manifest.jsonl
   │
   ▼
[2] EXTRACT & NORMALIZE TEXT      → raw_text/{file_hash}.txt + meta.json
   │       (markitdown + OCR + code-aware)
   ▼
[3] LANGUAGE ID & ROUTING         → lang/{en,zh,ja,...}/
   │
   ▼
[4] SEMANTIC CHUNKING             → chunks/{file_hash}_chunk_{n}.json
   │       (structure-aware: headings, tables, code blocks, threads)
   ▼
[5] PII/PHI/PCI DETECTION & REDACTION → pii/{file_hash}_pii.jsonl + redacted chunks
   │
   ▼
[6] QUALITY SCORING & FILTER      → quality/{file_hash}_scored.jsonl
   │       (perplexity, coherence, toxicity, domain relevance, instruction-pair viability)
   ▼
[7] INSTRUCTION-RESPONSE SYNTHESIS → sft_pairs/{file_hash}_pairs.jsonl
   │       (LLM-judge: verify/verify module generates instruction/response from chunks)
   ▼
[8] NEAR-DEDUP (MinHash + embedding) → dedup/{corpus_id}_clusters.jsonl
   │
   ▼
[9] FINAL ASSEMBLY & SPLIT        → sft_pairs.jsonl + train/val/test splits
   │
   ▼
[10] REGISTRY & ARTIFACT PUBLISH   → registry/datasets.jsonl + provenance
```

**Each stage:** idempotent, resumable, writes to `data/processed/{corpus_id}/stage_N/`, emits metrics to `logs/metrics/{stage_N}.jsonl`.

---

## 4. Stage Specifications

### Stage 1: Discover & Inventory
- **Input:** `raw_input/` (local path or s3:// / gs:// / az://)
- **Tools:** `fsspec`, `python-magic`, `xxhash` (xxh128), `scandir` parallel walk
- **Output:** `manifest.jsonl` — one line per file:
  ```json
  {"file_hash": "xxh128...", "rel_path": "legal/contracts/msa.pdf", "mime": "application/pdf", "size_bytes": 2048576, "modified": "2024-03-15T14:22:00Z", "status": "pending"}
  ```
- **Config:** `max_file_size_mb`, `allowed_mimes`, `exclude_patterns` (regex)

### Stage 2: Extract & Normalize Text
- **Strategy per mime:**
  - PDF → `marker` (preferred) → fallback `pymupdf` + `pdfplumber` for tables
  - DOCX/PPTX/XLSX → `markitdown` (Microsoft) → markdown
  - ODT/RTF → `pandoc` → markdown
  - Images → `tesseract` (fast) + `surya` (layout-aware) + `paddleocr` (CJK) → markdown
  - Code → `tree-sitter` parse → extract: docstrings, comments, function/class defs, config blocks
  - Chat exports → custom parsers per platform → normalize to `[{role, content, timestamp, thread_id}]`
  - Email → `eml-parser` / `msg-extractor` → thread reconstruction
  - Archives → recursive extract (depth limit 5, size limit 1GB)
- **Output:** `raw_text/{file_hash}.txt` + `raw_text/{file_hash}.meta.json` (page_map, structure, extracted_images_ocr)
- **Checkpoint:** Write manifest `status: extracted|failed` with error details

### Stage 3: Language ID & Routing
- **Tool:** `fasttext` (lid.176.bin) + `cld3` ensemble; threshold 0.95
- **Route:** Copy/symlink to `lang/{iso639_1}/` — downstream stages run per-lang
- **Unsupported langs:** Quarantine to `lang/und/` for human review

### Stage 4: Semantic Chunking
- **Document-aware chunkers:**
  - Markdown/HTML → heading-tree chunks (max 1024 tokens, overlap 128)
  - Code → AST chunks (function/class/module) via `tree-sitter`
  - Chat/Email → thread-aware: one chunk = one conversation turn or thread segment
  - Tables → serialize to markdown + caption chunk
  - Images (OCR) → caption + surrounding text
- **Output:** `chunks/{file_hash}_chunk_{n}.json`:
  ```json
  {"chunk_id": "uuid", "file_hash": "...", "text": "...", "token_count": 342, "meta": {"section": "Section 4.2", "type": "clause", "lang": "en", "page": 12}}
  ```

### Stage 5: PII/PHI/PCI Detection & Redaction
- **Detectors (ensemble):**
  - `presidio-analyzer` (regex + NER) — EN, ES, FR, DE, IT, PT, NL, PL, RO
  - `piiranha` (transformer) — EN, multilingual fine-tunes
  - Custom regex: Indian PAN/Aadhaar, CN ID, JP MyNumber, credit card, IBAN, SWIFT, medical codes (ICD-10, CPT)
  - LLM-judge (`verify/` module) for context-dependent PII (e.g., "my client John S. at Acme")
- **Policy per domain (config):**
  - `redact` → replace with `[PERSON_1]`, `[ORG_3]`, `[EMAIL_2]`
  - `pseudonymize` → consistent FPE (format-preserving encryption) per entity
  - `drop_chunk` → if PII density > threshold
  - `quarantine` → flag for human review
- **Output:** `pii/{file_hash}_pii.jsonl` (findings + actions) + redacted chunks

### Stage 6: Quality Scoring & Filter
- **Scorers (weighted, configurable):**
  - Perplexity (KenLM / small LM) — `weight 0.25`
  - Coherence (sentence embedding continuity) — `0.15`
  - Toxicity (`detoxify` / `unitary`) — `0.15`
  - Domain relevance (embedding similarity to domain centroids) — `0.20`
  - Instruction-pair viability (LLM-judge: "can this become a good Q/A?") — `0.25`
- **Thresholds:** Per-domain, configurable. Default `quality_score >= 0.75`
- **Output:** `quality/{file_hash}_scored.jsonl` with scores + `pass: bool`

### Stage 7: Instruction-Response Synthesis (SFT Pair Generation)
- **Input:** High-quality chunks (passed Stage 6)
- **Method:** LLM-judge (`verify/` module) — few-shot prompts per domain:
  - Legal → "Generate a question a junior associate would ask about this clause"
  - Code → "Write a docstring / unit test / refactor prompt for this function"
  - Chat → "Summarize the decision made in this thread"
  - General → "Create a realistic user query this text answers"
- **Verification:** Second LLM pass scores pair quality (1-5), keeps ≥4
- **Output:** `sft_pairs/{file_hash}_pairs.jsonl`:
  ```json
  {"id": "uuid", "source_chunk": "chunk_id", "instruction": "...", "response": "...", "meta": {"domain": "legal", "quality": 4.5, "tokens_in": 256, "tokens_out": 380}}
  ```

### Stage 8: Near-Deduplication
- **MinHash LSH** (datasketch) — 13-gram, 128 perms, threshold 0.7 Jaccard
- **Embedding dedup** (optional) — `bge-small-en-v1.5` + FAISS HNSW, cosine > 0.95
- **Cross-file / cross-domain** — cluster globally
- **Policy:** Keep highest-quality representative per cluster; log cluster members for audit
- **Output:** `dedup/{corpus_id}_clusters.jsonl` + `dedup/{corpus_id}_kept_ids.txt`

### Stage 9: Final Assembly & Splits
- **Aggregate** all kept pairs → shuffle (seed=42) → split 90/5/5 (train/val/test) by `source_file_hash` (no leakage)
- **Token count** verification (tiktoken cl100k_base)
- **Output:** `sft_pairs.jsonl` + `splits/{train,val,test}.jsonl`

### Stage 10: Registry & Publish
- Write registry entry (`lab_unme/registry/datasets.jsonl`)
- Copy/manifest to `data/processed/{corpus_id}/v{version}/`
- Emit `quality_report.html` (plots: score dists, domain mix, token counts, dedup rates, PII rates)
- Emit `LINEAGE.md` (reproducible: config hash, code version, input manifest hash)

---

## 5. Configuration Schema (`dataprep/config.yaml`)

```yaml
corpus_id: "corp_legal_v1"
source_root: "/mnt/raw/legal_dump"
output_root: "data/processed"

discover:
  max_file_size_mb: 500
  allowed_mimes: ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", ...]
  exclude_patterns: [".*[/\\\\]\\..*", ".*~$", ".*Thumbs\\.db"]

extract:
  pdf_engine: "marker"  # marker|pymupdf|pdfplumber
  ocr_engines: ["surya", "tesseract", "paddleocr"]
  ocr_langs: ["en", "zh", "ja", "ko", "ar", "hi"]
  code_parsers: ["python", "javascript", "typescript", "java", "go", "rust", "cpp", "sql"]
  max_archive_depth: 5
  max_archive_size_gb: 1

chunk:
  max_tokens: 1024
  overlap_tokens: 128
  min_tokens: 64
  respect_boundaries: true  # headings, functions, threads

pii:
  policy: "redact"  # redact|pseudonymize|drop|quarantine
  detectors: ["presidio", "piiranha", "regex_custom", "llm_judge"]
  presidio_langs: ["en", "es", "fr", "de", "it", "pt", "nl", "pl", "ro"]
  custom_regex_file: "config/pii_regex.yaml"
  llm_judge_model: "lab_unme/verify/pii_judge_v1"
  density_threshold_drop: 0.15

quality:
  scorers:
    perplexity: {weight: 0.25, model: "kenlm/en_3gram.arpa"}
    coherence: {weight: 0.15, embed_model: "sentence-transformers/all-MiniLM-L6-v2"}
    toxicity: {weight: 0.15, model: "unitary/toxic-bert"}
    domain_relevance: {weight: 0.20, centroids: "config/domain_centroids/"}
    instruction_viability: {weight: 0.25, judge_model: "lab_unme/verify/sft_judge_v1"}
  threshold: 0.75
  per_domain_overrides:
    legal: 0.80
    code: 0.70
    chat: 0.70

sft_synthesis:
  judge_model: "lab_unme/verify/sft_judge_v1"
  fewshot_dir: "config/fewshots/"
  min_quality: 4  # 1-5 scale
  max_pairs_per_chunk: 3

dedup:
  minhash: {ngram: 13, perms: 128, threshold: 0.7}
  embedding: {enabled: true, model: "BAAI/bge-small-en-v1.5", threshold: 0.95}
  keep_policy: "highest_quality"

splits:
  train: 0.90
  val: 0.05
  test: 0.05
  split_by: "source_file_hash"
  seed: 42

execution:
  workers: 32
  batch_size: 64
  checkpoint_every: 1000
  resume: true
  log_level: "INFO"
```

---

## 6. Infrastructure & Operations

### Compute Profile
| Stage | Compute | Memory | Parallelism | Est. Time (1M files) |
|---|---|---|---|---|
| 1 Discover | CPU | Low | 64 processes | 15 min |
| 2 Extract | CPU + GPU (OCR) | 16-32GB | 16 workers × 2 GPU | 8-24 hrs |
| 3 LangID | CPU | Low | 32 processes | 30 min |
| 4 Chunk | CPU | Low | 32 processes | 1 hr |
| 5 PII | CPU + GPU (transformer) | 16GB | 16 workers × 1 GPU | 4-8 hrs |
| 6 Quality | CPU + GPU (embed/LM) | 16GB | 16 workers × 1 GPU | 4-8 hrs |
| 7 SFT Synth | GPU (LLM judge) | 24-48GB | 8 workers × 2-4 GPU | 12-36 hrs |
| 8 Dedup | CPU + GPU (embed) | 32GB+ | 1 coordinator | 2-4 hrs |
| 9 Assemble | CPU | Low | 1 | 15 min |

**Recommended:** Kubernetes (Argo Workflows / Prefect / Temporal) or Ray on GPU cluster. Local dev: `modal` / `runhouse` / `dask` + single 8xH100 box.

### Observability
- **Metrics:** Prometheus + Grafana (stage throughput, error rates, quality distributions, PII rates, dedup ratios)
- **Logs:** Structured JSONL → Loki / Elastic
- **Lineage:** MLflow / Dagster / custom registry (every artifact has `config_hash`, `code_version`, `input_manifest_hash`)

### Incremental / Re-processing
- **Manifest hash** → skip unchanged files
- **Stage-level checkpoints** → resume from failure
- **Config versioning** → re-run only affected stages (e.g., PII policy change → re-run 5-10)
- **Corpus versioning** → `v1, v2, v3` immutable; registry tracks lineage

---

## 7. Testing & Quality Gates

### Unit Tests (per stage)
- Fixture files per mime type → golden outputs
- PII detector: precision/recall on labeled PII corpus (≥0.95/0.98)
- Chunker: structural invariants (no split mid-function, mid-table, mid-thread)
- Quality scorers: monotonicity on known good/bad samples

### Integration Tests
- End-to-end on `tests/fixtures/mini_corpus/` (50 files, mixed types) → assert output schema, token counts, split ratios
- Determinism: same input + same config → bitwise identical `sft_pairs.jsonl`

### Acceptance Criteria (per corpus version)
| Metric | Threshold |
|---|---|
| PII recall (audit sample) | ≥ 99.5% |
| PII precision (audit sample) | ≥ 95% |
| Quality score dist: % ≥ threshold | ≥ 80% |
| Dedup removal rate | 5-25% (domain dependent) |
| Instruction-pair pass rate (Stage 7) | ≥ 60% |
| Train/val/test token leakage | 0% |
| Schema validation pass | 100% |

---

## 8. Deliverables Checklist

- [ ] `dataprep/` package with CLI: `unme-dataprep run --config config.yaml`
- [ ] Stage modules: `discover/`, `extract/`, `langid/`, `chunk/`, `pii/`, `quality/`, `synthesize/`, `dedup/`, `assemble/`, `publish/`
- [ ] Config schema (Pydantic) + example `config.yaml`
- [ ] Dockerfile (CPU + GPU variants)
- [ ] Argo/Prefect/Dagster pipeline definition
- [ ] `tests/fixtures/mini_corpus/` + golden outputs
- [ ] `docs/ARCHITECTURE.md`, `docs/OPERATIONS.md`, `docs/PIL_CONFIG.md`
- [ ] Benchmark script: `scripts/bench_dataprep.py` (throughput, cost estimates)

---

## 9. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| OCR quality on scanned PDFs | High | Medium | Ensemble OCR (surya + paddleocr); flag low-confidence for human review |
| PII false negatives (PHI/PCI) | Medium | Critical | Defense-in-depth: regex + NER + LLM-judge; audit sample every run |
| LLM-judge cost (Stage 7) | High | High | Distill judge to smaller model; batch aggressively; cache prompts |
| Domain relevance centroids stale | Medium | Medium | Recompute centroids quarterly from fresh high-quality data |
| Code chunking misses context | Medium | Low | Tree-sitter + parent-scope inclusion; configurable context window |
| Chat thread fragmentation | High | Medium | Platform-specific thread reconstructors; fallback to time-window grouping |
| Data volume exceeds single-node | Medium | High | Design for distributed (Ray/Dask) from day 1; stateless stages |

---

## 10. Next Steps (Priority Order)

1. **Scaffold repo structure** + config schema + CLI entrypoint
2. **Stage 1-2 (Discover + Extract)** — highest reuse, biggest time saver for clients
3. **Stage 4 (Chunking)** — domain-aware chunking is the quality lever
4. **Stage 5 (PII)** — compliance gate; build audit harness first
5. **Stage 6 (Quality)** — pluggable scorers; start with perplexity + toxicity
6. **Stage 7 (SFT Synthesis)** — integrate with `lab_unme/verify/` LLM-judge
7. **Stage 8-9 (Dedup + Assemble)** — standard implementations
8. **Stage 10 (Registry)** — wire to `lab_unme/registry/`
9. **End-to-end test** on `tests/fixtures/mini_corpus/`
10. **Documentation + Docker + CI/CD**

---

## 11. Handoff Note

**Completed:** Plan document created at `dataprep/PLAN.md`
**Files changed:** `dataprep/PLAN.md` (new)
**Key findings:** Pipeline designed for heterogeneous enterprise dumps → SFT-ready JSONL with full lineage, PII safety, quality gates, and registry integration. Modular stages enable incremental adoption.
**Unresolved gaps:**
- LLM-judge models (`verify/`) not yet built — dependency
- Domain centroids for relevance scoring need seed data
- Distributed execution framework (Ray/Prefect/Argo) not selected
- Cost model for GPU stages needs real benchmark data
**Risks:** OCR/PII accuracy on non-English; Stage 7 GPU cost at scale
**Recommended next owner:** Data Engineer (scaffold + Stages 1-4) → ML Engineer (Stages 5-7, judge integration)