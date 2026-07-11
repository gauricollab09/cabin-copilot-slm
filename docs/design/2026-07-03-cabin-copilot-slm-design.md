# CabinCopilot-SLM — Design Specification

**Date:** 2026-07-03
**Author:** Gauri Tulsulkar (with Claude Code)
**Status:** Approved

## 1. Problem statement

Driver- and cabin-monitoring systems (DMS/CMS) emit structured perception signals every
frame: drowsiness level, gaze zone, phone usage, occupancy, speed context, trip duration.
Turning these signals into timely, safe, non-annoying natural-language coaching currently
requires a large cloud LLM — which is costly, latency-bound, and sends sensitive cabin
data off-device.

**Goal:** fine-tune a 3B-parameter small language model (SLM) that converts a structured
DMS signal snapshot into a driver-coaching response with a strict JSON contract, runs
**on-device** (quantized, CPU-capable), and demonstrably outperforms both its own base
model and a much larger prompted model on this task.

## 2. Success criteria

1. Fine-tuned 3B model achieves **≥ 98% JSON schema-pass rate** on a held-out eval set
   (base model expected well below this).
2. **Safety-rule compliance ≥ 95%** on hard rules (see §5.3).
3. **LLM-judge pairwise win-rate > 60%** vs the base 3B model with an engineered prompt.
4. Quantized GGUF (Q4_K_M) runs in Ollama on a 2019 Intel MacBook Pro (16 GB, CPU-only)
   — a genuine on-device demo.
5. Fully reproducible: seeded data generation, config-driven training, pinned deps, CI.

## 3. Output contract

Every model response is a single JSON object:

```json
{
  "severity": "none | info | caution | warning | critical",
  "message": "<coaching utterance, <= 20 words, spoken-style>",
  "action": "none | suggest_break | reduce_distraction | alert_drowsiness | phone_reminder | escalate_alarm",
  "rationale": "<one-sentence internal reason, not spoken>"
}
```

`message` must be empty when `severity` is `none`, and terse (≤ 8 words) when `critical`
(a distracted driver must not be given a paragraph to parse).

## 4. Architecture

```
scenario simulator ──▶ teacher generation ──▶ quality gate ──▶ dataset (JSONL)
      (seeded)          (pluggable LLM)        (validate/judge/balance)
                                                        │
                                                        ▼
   on-device demo ◀── GGUF Q4_K_M export ◀── QLoRA training (Kaggle T4)
     (Ollama, Mac)                                      │
                                                        ▼
   FastAPI serving ◀───────────────────────── evaluation harness
                                       (schema / safety / judge win-rate)
```

### 4.1 Scenario simulator (`src/cabin_copilot/simulator/`)
Deterministic, seeded generator of realistic DMS signal states. Covers a safety taxonomy:
drowsiness episodes (micro-sleep, yawning cadence), gaze-off-road (duration × speed),
phone-in-hand, passenger distraction, long-trip fatigue, and deliberately benign /
false-positive-prone cases (sunglasses, glance-to-mirror). Pure Python, no LLM, fully
unit-tested. Output: `Scenario` records (Pydantic).

### 4.2 Teacher generation (`src/cabin_copilot/teacher/`)
Pluggable backends behind one `TeacherBackend` interface:
- **Anthropic** (Claude) — best quality
- **OpenAI** (GPT) — strong alternative
- **Open/local** — Groq or OpenRouter hosted open models, or local Ollama; $0 path

Backend chosen by config/env. Engineered teacher system prompt produces the §3 contract
plus reasoning. Retries with exponential backoff, resumable JSONL checkpoints, cost
tracking.

### 4.3 Quality gate (`src/cabin_copilot/quality/`)
1. JSON schema validation (hard reject).
2. Deterministic safety rules (§5.3) (hard reject).
3. LLM-judge rubric scoring 1–5 on helpfulness/tone/appropriateness (reject < 4).
4. Near-duplicate removal; severity-class balancing.

Rejected samples are regenerated up to N attempts.

### 4.4 Training (`training/`)
QLoRA (4-bit NF4) fine-tune of **Qwen2.5-3B-Instruct** using TRL `SFTTrainer` + PEFT.
YAML-config-driven (model, LoRA rank/alpha, LR, epochs, seed). Runs on a free Kaggle
T4/P100, driven headlessly via Kaggle CLI. Loss curves and configs committed. Fallback:
Llama-3.2-1B locally on CPU if Kaggle is unavailable.

### 4.5 Evaluation (`src/cabin_copilot/eval/`)
Held-out scenario set (never seen by training). Metrics:
- JSON schema-pass rate
- Safety-rule compliance rate
- Severity accuracy (vs simulator ground-truth band)
- LLM-judge pairwise win-rate: fine-tuned 3B vs base 3B (both with same minimal prompt),
  and fine-tuned 3B vs larger prompted model
- Latency & memory footprint table (on-device)

### 4.6 Export (`export/`)
Merge LoRA into base weights → convert to GGUF → quantize Q4_K_M → Ollama `Modelfile`.
Verified end-to-end on the Intel MacBook.

### 4.7 Serving (`serving/`)
FastAPI microservice: `POST /coach` (signals in → contract out), `/healthz`, request
validation, model backend behind an interface (Ollama local). Dockerfile. Pytest suite.

### 4.8 Explainers (`docs/explainers/`)
One self-contained HTML page per phase documenting every decision, the reasoning, and
rejected alternatives — written live during the build.

## 5. Data design

### 5.1 Scenario schema (input)
```
speed_kmh, trip_minutes, time_of_day, drowsiness{eye_closure_pct, yawn_count_5min,
head_nod_events}, gaze{zone, off_road_ms}, phone{in_hand, screen_on}, cabin{n_passengers,
child_present, conversation}, history{alerts_last_30min}
```
Serialized as compact JSON in the user turn.

### 5.2 Dataset
~1,500–2,500 accepted training pairs + 150 held-out eval scenarios (stratified across
taxonomy cells). Chat format (system/user/assistant) in JSONL.

### 5.3 Hard safety rules (deterministic, checkable)
- `critical` severity ⇒ message ≤ 8 words and action ∈ {alert_drowsiness, escalate_alarm}
- phone in hand + moving vehicle ⇒ severity ≥ caution
- eye closure ≥ 60% sustained ⇒ severity ≥ warning
- benign scenario (taxonomy: none) ⇒ severity = none and empty message (no nagging)
- never recommend interacting with a screen/phone while moving
- alert fatigue: ≥ 3 alerts in last 30 min and severity < warning ⇒ suppress (none)

## 6. Testing strategy
- Unit tests: simulator determinism/coverage, schema validators, safety rules, quality
  gate, serving endpoint (mocked model).
- Integration test: tiny end-to-end run with a stub teacher backend.
- CI: GitHub Actions — ruff + pytest on every push.

## 7. Risks & fallbacks
| Risk | Fallback |
|---|---|
| No teacher API key provided | Local Ollama teacher (llama3.2:3b) — slower, lower quality |
| Kaggle GPU unavailable | Local CPU LoRA on Llama-3.2-1B overnight |
| GGUF conversion friction on Intel Mac | Convert on Kaggle, download artifact |
| Teacher rate limits | Backoff + resumable checkpoints |

## 8. Out of scope
Real camera input, audio/TTS, multi-turn dialogue memory, the RAG project (project #2,
next week).
