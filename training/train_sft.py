"""QLoRA supervised fine-tuning, config-driven and GPU-host agnostic.

Runs anywhere with a CUDA GPU (designed for a free Kaggle T4):

    python training/train_sft.py --config training/configs/qlora_qwen25_3b.yaml \
        --dataset data/dataset/train.jsonl

Outputs (under training.output_dir): adapter/ (PEFT LoRA), loss_log.csv, config
snapshot. The heavy imports live inside main() so the repo's core package never
depends on GPU libraries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default=None, help="override config output_dir")
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    cfg = yaml.safe_load(Path(args.config).read_text())
    quant, lora, tr = cfg["quantization"], cfg["lora"], cfg["training"]
    output_dir = Path(args.output_dir or tr["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_snapshot.yaml").write_text(Path(args.config).read_text())

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=quant["load_in_4bit"],
            bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, quant["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
        ),
        device_map="auto",
    )

    dataset = load_dataset("json", data_files=args.dataset, split="train")
    dataset = dataset.train_test_split(
        test_size=tr["eval_holdout_fraction"], seed=cfg["seed"]
    )
    print(f"train={len(dataset['train'])} eval={len(dataset['test'])}")

    loss_rows: list[dict] = []

    class LossLogger(TrainerCallback):
        def on_log(self, _args, state, control, logs=None, **kwargs):
            if logs and ("loss" in logs or "eval_loss" in logs):
                loss_rows.append({"step": state.global_step, **logs})

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        ),
        args=SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=tr["epochs"],
            per_device_train_batch_size=tr["per_device_batch_size"],
            gradient_accumulation_steps=tr["gradient_accumulation_steps"],
            learning_rate=float(tr["learning_rate"]),
            lr_scheduler_type=tr["lr_scheduler_type"],
            warmup_ratio=tr["warmup_ratio"],
            max_length=tr["max_seq_length"],
            logging_steps=tr["logging_steps"],
            eval_strategy="epoch",
            save_strategy="no",
            gradient_checkpointing=tr["gradient_checkpointing"],
            optim=tr["optim"],
            seed=cfg["seed"],
            fp16=True,
            report_to=[],
        ),
        callbacks=[LossLogger()],
    )
    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    with (output_dir / "loss_log.csv").open("w", newline="") as f:
        if loss_rows:
            fieldnames = sorted({k for row in loss_rows for k in row})
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(loss_rows)

    metrics = trainer.evaluate()
    (output_dir / "final_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
