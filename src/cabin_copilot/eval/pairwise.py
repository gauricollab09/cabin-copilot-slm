"""Pairwise LLM-judge comparison between two evaluation runs.

Position bias is controlled by deterministically alternating which run is shown as
"Response 1" (seeded shuffle per scenario id), and the judge never sees model names.
Scenarios where either run produced no parsable response are decided mechanically:
a parsable response beats a broken one; two broken responses tie.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from cabin_copilot.schemas import Scenario
from cabin_copilot.teacher.base import LLMBackend
from cabin_copilot.teacher.generate import extract_json

PAIRWISE_SYSTEM_PROMPT = """\
You judge in-car driver-coaching responses. You get the driver-monitoring signals and
two candidate JSON responses. Decide which better serves a real driver: correct urgency
for the signals, natural calm spoken wording, appropriate action, no nagging on benign
situations, terse commands in emergencies.

Respond ONLY with JSON: {"winner": "1" | "2" | "tie", "reason": "<one sentence>"}
"""


def _index(run_path: Path) -> dict[str, dict]:
    data = json.loads(run_path.read_text())
    return {r["scenario_id"]: r for r in data["records"]}


def compare_runs(
    run_a: Path,
    run_b: Path,
    scenarios: list[Scenario],
    backend: LLMBackend,
    out_path: Path,
    *,
    seed: int = 13,
) -> dict:
    a_records, b_records = _index(run_a), _index(run_b)
    results = []
    wins = {"a": 0, "b": 0, "tie": 0}

    for scenario in scenarios:
        ra = a_records.get(scenario.scenario_id, {})
        rb = b_records.get(scenario.scenario_id, {})
        resp_a, resp_b = ra.get("response"), rb.get("response")

        if resp_a is None and resp_b is None:
            verdict, reason = "tie", "both responses unparsable"
        elif resp_a is None:
            verdict, reason = "b", "only B produced a valid response"
        elif resp_b is None:
            verdict, reason = "a", "only A produced a valid response"
        else:
            a_first = random.Random(f"{seed}:{scenario.scenario_id}").random() < 0.5
            first, second = (resp_a, resp_b) if a_first else (resp_b, resp_a)
            user = (
                f"Signals: {scenario.signals_json()}\n"
                f"Response 1: {json.dumps(first)}\n"
                f"Response 2: {json.dumps(second)}"
            )
            data = extract_json(
                backend.complete(PAIRWISE_SYSTEM_PROMPT, user, temperature=0.0, max_tokens=200)
            )
            winner = str(data.get("winner", "tie"))
            reason = str(data.get("reason", ""))
            if winner == "tie":
                verdict = "tie"
            elif winner == "1":
                verdict = "a" if a_first else "b"
            elif winner == "2":
                verdict = "b" if a_first else "a"
            else:
                verdict = "tie"
        wins[verdict] += 1
        results.append(
            {"scenario_id": scenario.scenario_id, "winner": verdict, "reason": reason}
        )

    decided = wins["a"] + wins["b"]
    summary = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "judge_model": backend.model,
        "n": len(scenarios),
        "wins_a": wins["a"],
        "wins_b": wins["b"],
        "ties": wins["tie"],
        "win_rate_a_over_decided": round(wins["a"] / decided, 4) if decided else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    return summary
