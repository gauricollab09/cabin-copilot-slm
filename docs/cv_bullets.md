# CV bullets — draft (numbers to be finalized after eval)

## For the CV (AI Engineer / ML Engineer framing)

**Personal project — CabinCopilot-SLM** (github.com/gauricollab09/cabin-copilot-slm)

- Built an end-to-end fine-tuning pipeline that distills a 1,200-token engineered
  teacher prompt into a 3B on-device model (QLoRA on Qwen2.5-3B): seeded scenario
  simulation over a 15-cell driver-safety taxonomy, label-conditioned synthetic data
  generation with pluggable teachers (Anthropic/OpenAI/open models), and a
  deterministic safety gate that lifted usable-data yield from 10% to 85%.
- Designed a three-layer evaluation harness — JSON schema-pass rate, deterministic
  safety-rule compliance, and position-bias-controlled pairwise LLM-judge — showing the
  fine-tuned 3B with a 60-token prompt [beats the prompted base model on X% of
  comparisons / achieves Y% schema-pass vs Z% base].
- Shipped the result as a production artifact on $0 infrastructure: headless QLoRA +
  GGUF Q4_K_M export on free Kaggle T4s, running on-device in Ollama on a 2019 Intel
  laptop, behind a contract-validating FastAPI service with CI and 35 unit tests.

## Notes for tailoring

- **AI Engineer roles**: lead with bullet 3 (shipping + serving), then 1.
- **ML Engineer roles**: lead with bullet 1 (data/training rigor), add loss-curve /
  ablation detail from explainer 03.
- **Prompt Engineer flavor**: the teacher-prompt design + "fine-tuning as prompt
  compression" thesis (explainer 01) is the story; the pairwise judge design is the
  methodology proof.
- One-liner for the summary section: *"Recent: fine-tuned and shipped an on-device 3B
  driver-coaching SLM end-to-end (QLoRA, synthetic data w/ safety gating, LLM-judge
  evals, GGUF/Ollama deployment) — github.com/gauricollab09/cabin-copilot-slm."*
