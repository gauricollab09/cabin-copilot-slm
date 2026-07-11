"""Kaggle GPU kernel: QLoRA training + GGUF export.

Expects a Kaggle dataset (see kernel-metadata.json dataset_sources) containing
train.jsonl produced by the teacher-gen kernel. Trains the adapter, merges it into the
base model, converts to GGUF and quantizes Q4_K_M — so the only thing to download is
the final on-device artifact.

Outputs in /kaggle/working: adapter/, loss_log.csv, final_metrics.json,
cabin-copilot-q4_k_m.gguf
"""

import os
import subprocess
from pathlib import Path

REPO = "https://github.com/gauricollab09/cabin-copilot-slm"
BRANCH = os.environ.get("CABIN_BRANCH", "main")
WORK = Path("/kaggle/working")
DATASET_DIR = Path("/kaggle/input/cabin-copilot-dataset")


def sh(cmd: str) -> None:
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    sh(f"git clone --depth 1 -b {BRANCH} {REPO} {WORK}/repo")
    sh("pip install -q -U trl peft bitsandbytes datasets transformers accelerate")

    train_jsonl = next(DATASET_DIR.rglob("train.jsonl"))
    sh(
        f"python {WORK}/repo/training/train_sft.py "
        f"--config {WORK}/repo/training/configs/qlora_qwen25_3b.yaml "
        f"--dataset {train_jsonl} --output-dir {WORK}/run"
    )

    # Merge LoRA into the base model at fp16 for GGUF conversion.
    merge_code = f"""
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct", torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(base, "{WORK}/run/adapter")
model = model.merge_and_unload()
model.save_pretrained("{WORK}/merged")
AutoTokenizer.from_pretrained("{WORK}/run/adapter").save_pretrained("{WORK}/merged")
"""
    (WORK / "merge.py").write_text(merge_code)
    sh(f"python {WORK}/merge.py")

    sh(f"git clone --depth 1 https://github.com/ggml-org/llama.cpp {WORK}/llama.cpp")
    sh(f"pip install -q -r {WORK}/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")
    sh(
        f"python {WORK}/llama.cpp/convert_hf_to_gguf.py {WORK}/merged "
        f"--outfile {WORK}/cabin-copilot-f16.gguf --outtype f16"
    )
    sh(
        f"cmake -S {WORK}/llama.cpp -B {WORK}/llama.cpp/build -DGGML_CUDA=OFF "
        f"&& cmake --build {WORK}/llama.cpp/build --target llama-quantize -j4"
    )
    sh(
        f"{WORK}/llama.cpp/build/bin/llama-quantize {WORK}/cabin-copilot-f16.gguf "
        f"{WORK}/cabin-copilot-q4_k_m.gguf Q4_K_M"
    )

    # keep only shippable artifacts
    sh(f"cp {WORK}/run/loss_log.csv {WORK}/run/final_metrics.json {WORK}/ || true")
    sh(f"cp -r {WORK}/run/adapter {WORK}/adapter")
    sh(f"rm -rf {WORK}/repo {WORK}/llama.cpp {WORK}/merged {WORK}/run {WORK}/cabin-copilot-f16.gguf")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
