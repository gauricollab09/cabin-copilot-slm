"""Quality gate: raw teacher records -> curated chat-format training dataset.

Stages (in order, each recorded in the stats report):
1. drop records with no parsed response
2. hard-reject on any deterministic safety-rule violation
3. optional LLM-judge rubric filter (score >= threshold)
4. near-duplicate removal on normalised message text (per severity class)
5. severity-class rebalancing by capping over-represented classes

Output is chat-format JSONL ready for TRL's SFTTrainer, using the SHORT student system
prompt — the whole point of the fine-tune is that the long teacher prompt is distilled
away.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from cabin_copilot.quality.safety_rules import check_safety
from cabin_copilot.schemas import CoachResponse, Scenario
from cabin_copilot.teacher.prompts import STUDENT_SYSTEM_PROMPT

_norm_re = re.compile(r"[^a-z0-9 ]+")


def normalise(message: str) -> str:
    return _norm_re.sub("", message.lower()).strip()


def to_chat_example(scenario: Scenario, response: CoachResponse) -> dict:
    return {
        "messages": [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": scenario.signals_json()},
            {"role": "assistant", "content": json.dumps(response.model_dump())},
        ]
    }


def run_filter(
    raw_path: Path,
    out_dir: Path,
    *,
    judge_scores: dict[str, int] | None = None,
    judge_threshold: int = 4,
    balance_cap_ratio: float = 2.5,
) -> dict:
    """Filter raw teacher JSONL into out_dir/{train.jsonl,stats.json}.

    judge_scores: optional map scenario_id -> rubric score (computed separately so the
    expensive judge pass can be run/resumed independently of this cheap pass).
    """
    stats: dict = {"input": 0, "no_response": 0, "safety_reject": 0, "judge_reject": 0,
                   "duplicate": 0, "balance_drop": 0, "accepted": 0,
                   "violations": Counter(), "by_severity": Counter(), "by_taxonomy": Counter()}
    kept: list[tuple[Scenario, CoachResponse]] = []
    seen: dict[str, set[str]] = defaultdict(set)

    with raw_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            stats["input"] += 1
            record = json.loads(line)
            if not record.get("response"):
                stats["no_response"] += 1
                continue
            scenario = Scenario.model_validate(record["scenario"])
            response = CoachResponse.model_validate(record["response"])

            violations = check_safety(scenario, response)
            if violations:
                stats["safety_reject"] += 1
                stats["violations"].update(violations)
                continue

            if judge_scores is not None:
                score = judge_scores.get(scenario.scenario_id)
                if score is not None and score < judge_threshold:
                    stats["judge_reject"] += 1
                    continue

            key = normalise(response.message)
            if key and key in seen[response.severity.value]:
                stats["duplicate"] += 1
                continue
            seen[response.severity.value].add(key)
            kept.append((scenario, response))

    # Rebalance: no severity class may exceed balance_cap_ratio x the smallest
    # non-empty class. Order within a class is preserved (deterministic).
    counts = Counter(r.severity.value for _, r in kept)
    if counts:
        cap = int(min(counts.values()) * balance_cap_ratio) or 1
        taken: Counter = Counter()
        balanced = []
        for scenario, response in kept:
            sev = response.severity.value
            if taken[sev] >= cap:
                stats["balance_drop"] += 1
                continue
            taken[sev] += 1
            balanced.append((scenario, response))
        kept = balanced

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    with train_path.open("w") as out:
        for scenario, response in kept:
            stats["accepted"] += 1
            stats["by_severity"][response.severity.value] += 1
            stats["by_taxonomy"][scenario.taxonomy] += 1
            out.write(json.dumps(to_chat_example(scenario, response)) + "\n")

    stats["violations"] = dict(stats["violations"])
    stats["by_severity"] = dict(stats["by_severity"])
    stats["by_taxonomy"] = dict(stats["by_taxonomy"])
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
