"""Kaggle GPU kernel: full data pipeline with a strong open teacher (no API keys).

Runs on 2x T4: serves Qwen2.5-14B-Instruct-AWQ with vLLM (OpenAI-compatible), then runs
the repo's own CLI stages against it: simulate -> generate -> judge -> filter.
Scenario generation is seeded, so the scenarios here are byte-identical to local ones.

Outputs in /kaggle/working: raw_train.jsonl, judge.jsonl, dataset/{train.jsonl,stats.json}
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = "https://github.com/gauricollab09/cabin-copilot-slm"
BRANCH = os.environ.get("CABIN_BRANCH", "main")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
N_TRAIN = int(os.environ.get("N_TRAIN", "2400"))
WORK = Path("/kaggle/working")


def sh(cmd: str, **kw) -> None:
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def main() -> None:
    sh(f"git clone --depth 1 -b {BRANCH} {REPO} /kaggle/working/repo")
    sh("pip install -q -e /kaggle/working/repo")
    sh("pip install -q vllm")

    ngpu = subprocess.run(
        "nvidia-smi -L | wc -l", shell=True, capture_output=True, text=True, check=True
    )
    tp = max(1, int(ngpu.stdout.strip()))
    server = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", TEACHER_MODEL,
            "--dtype", "float16",
            "--tensor-parallel-size", str(tp),
            "--gpu-memory-utilization", "0.92",
            "--max-model-len", "4096",
            "--enable-prefix-caching",
            "--port", "8000",
        ]
    )
    deadline = time.time() + 1800
    while time.time() < deadline:
        try:
            if httpx.get("http://localhost:8000/health", timeout=5).status_code == 200:
                break
        except httpx.TransportError:
            pass
        if server.poll() is not None:
            raise RuntimeError("vLLM server exited during startup")
        time.sleep(10)
    else:
        raise TimeoutError("vLLM server did not become healthy in 30 min")
    print("vLLM ready", flush=True)

    env = {**os.environ, "OPENAI_API_KEY": "local"}
    base = f"--provider openai --model {TEACHER_MODEL} --base-url http://localhost:8000/v1"
    sh(
        f"cabin-copilot simulate --n {N_TRAIN} --seed 7 --prefix train "
        f"--out {WORK}/scenarios_train.jsonl",
        env=env,
    )
    sh(
        f"cabin-copilot generate --scenarios {WORK}/scenarios_train.jsonl "
        f"--out {WORK}/raw_train.jsonl {base}",
        env=env,
    )
    sh(
        f"cabin-copilot judge --raw {WORK}/raw_train.jsonl --out {WORK}/judge.jsonl {base}",
        env=env,
    )
    sh(
        f"cabin-copilot filter --raw {WORK}/raw_train.jsonl --judge {WORK}/judge.jsonl "
        f"--out {WORK}/dataset",
        env=env,
    )
    server.terminate()
    sh("rm -rf /kaggle/working/repo")  # keep kernel output small
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
