# CabinCopilot-SLM

**A fine-tuned, on-device small language model that turns driver-monitoring signals into
safe, context-aware spoken coaching.**

Driver/cabin-monitoring systems emit structured perception signals every frame —
drowsiness level, gaze zone, phone usage, occupancy, trip context. Turning those into
timely, non-annoying natural-language coaching normally means prompting a large cloud
LLM: costly, latency-bound, and it ships sensitive cabin data off-device.

This project distills that behaviour into a **3B model (Qwen2.5-3B-Instruct, QLoRA)**
that emits a strict JSON contract, runs **quantized on a 2019 Intel laptop CPU**, and is
built end-to-end on **free infrastructure** — no paid APIs, no owned GPUs.

> **The thesis: fine-tuning as prompt compression.** The teacher model needs a
> ~1,200-token engineered prompt to do this task well. The fine-tuned student does it
> with a ~60-token prompt — a 20× reduction that, on-device, *is* the latency budget.

```
scenario simulator ──▶ teacher generation ──▶ quality gate ──▶ dataset (JSONL)
      (seeded)          (pluggable LLM)        (validate/judge/balance)
                                                        │
                                                        ▼
   on-device demo ◀── GGUF Q4_K_M export ◀── QLoRA training (free Kaggle T4)
     (Ollama, Mac)                                      │
                                                        ▼
   FastAPI serving ◀───────────────────────── evaluation harness
                                       (schema / safety / judge win-rate)
```

## The output contract

```json
{
  "severity": "none | info | caution | warning | critical",
  "message":  "spoken sentence, ≤20 words — empty when severity is none",
  "action":   "none | suggest_break | reduce_distraction | alert_drowsiness | phone_reminder | escalate_alarm",
  "rationale":"one internal sentence, never spoken"
}
```

Behavioural guarantees are enforced twice: deterministic safety rules hard-gate the
training data, then the same rules become an eval metric (e.g. *critical ⇒ ≤ 8 words and
an alarm action*; *phone-in-hand while moving may never be under-called*; *≥3 recent
alerts ⇒ suppress low-severity nagging*).

## Results

<!-- RESULTS_TABLE: filled by eval run -->
*Training in progress — this table is populated by `cabin-copilot eval` /
`cabin-copilot compare` outputs in `eval/results/`.*

| Model | System prompt | Schema-pass | Safety compliance | Severity-in-band | Judge win-rate vs base |
|---|---|---|---|---|---|
| Qwen2.5-3B base | student (60 tok) | 20.0% | 10.7% | 11.3% | – |
| Qwen2.5-3B base | teacher (1.2k tok) | – | – | – | – |
| **CabinCopilot-3B (ours)** | student (60 tok) | – | – | – | – |

Baseline (student prompt, n=150, local Ollama Q4_K_M on a 2019 Intel MacBook,
p50 latency 5.6 s): the base model can barely hold the contract without the full
engineered prompt — that gap is what fine-tuning must close.

## Quickstart

```bash
pip install -e ".[dev]"
pytest                      # 35 unit tests, no network needed
```

### Reproduce the dataset (pick any backend)

```bash
cabin-copilot simulate --n 2400 --seed 7 --prefix train --out data/scenarios/train.jsonl

# choose ONE teacher backend:
export ANTHROPIC_API_KEY=...   # best quality
cabin-copilot generate --scenarios data/scenarios/train.jsonl --out data/raw/train.jsonl \
    --provider anthropic --hint-band

# or OpenAI / Groq / OpenRouter (same flag shape), or fully local & free:
cabin-copilot generate --scenarios data/scenarios/train.jsonl --out data/raw/train.jsonl \
    --provider ollama --model llama3.2 --hint-band

cabin-copilot judge  --raw data/raw/train.jsonl --out data/raw/judge.jsonl --provider ...
cabin-copilot filter --raw data/raw/train.jsonl --judge data/raw/judge.jsonl --out data/dataset
```

Scenario generation is seeded and deterministic — the same seed yields byte-identical
scenarios on any machine, which is how the Kaggle GPU kernel regenerates them instead of
shipping data around.

### Train (free Kaggle T4)

```bash
scripts/kaggle_push.sh kaggle/teacher_gen   # dataset via 14B teacher on 2×T4 (vLLM)
scripts/kaggle_push.sh kaggle/train         # QLoRA + merge + GGUF Q4_K_M export
# or anywhere with a CUDA GPU:
python training/train_sft.py --config training/configs/qlora_qwen25_3b.yaml \
    --dataset data/dataset/train.jsonl
```

### Run on-device

```bash
ollama create cabin-copilot -f export/Modelfile     # GGUF next to the Modelfile
ollama run cabin-copilot '{"speed_kmh":88.0,"trip_minutes":190,...}'
```

### Serve

```bash
uvicorn cabin_copilot.serving.app:app --port 8080
# or: docker build -t cabin-copilot . && docker run -p 8080:8080 \
#       -e CABIN_BASE_URL=http://host.docker.internal:11434/v1 cabin-copilot
curl -X POST localhost:8080/coach -H 'content-type: application/json' -d @examples/drowsy.json
```

## Repository map

| Path | What it is |
|---|---|
| `src/cabin_copilot/simulator/` | Seeded scenario generator over a 15-cell safety taxonomy |
| `src/cabin_copilot/teacher/` | Pluggable LLM backends (Anthropic/OpenAI/Groq/OpenRouter/Ollama), prompts, resumable generation |
| `src/cabin_copilot/quality/` | Deterministic safety rules, LLM-judge rubric, dedup + class balancing |
| `src/cabin_copilot/eval/` | Metrics harness + position-bias-controlled pairwise judge |
| `src/cabin_copilot/serving/` | FastAPI `/coach` service with contract validation |
| `training/` | Config-driven QLoRA (TRL + PEFT) |
| `kaggle/` | Free-GPU kernels: 14B-teacher data generation, training + GGUF export |
| `export/` | Ollama Modelfile (student prompt baked in) |
| `docs/explainers/` | **The build log**: every decision, its reasoning, and rejected alternatives |
| `docs/design/` | Design specification |

## Design highlights (the interview version)

- **Simulated scenarios, not LLM-imagined ones** — stratified coverage of rare critical
  cases, deterministic, reproducible.
- **Label-conditioned generation** — a 3B teacher mis-calibrated severity on 77% of
  samples; conditioning the teacher on the simulator's ground-truth band lifted quality-gate
  acceptance from ~10% to ~85% (measured, see explainer 02).
- **Safety as code, twice** — the same deterministic rules hard-gate training data and
  score the final model, independently of schema validity.
- **Teacher/student prompt asymmetry** — all engineered behaviour must survive
  distillation into the weights; the eval harness exists to prove it did.

## License

MIT — see [LICENSE](LICENSE). Synthetic data only; no proprietary systems reproduced.
