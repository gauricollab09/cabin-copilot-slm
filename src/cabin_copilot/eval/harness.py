"""Evaluation harness: run any backend over held-out scenarios and score it.

The candidate model always receives the SHORT student system prompt (optionally an
engineered longer one via --system teacher, for the prompting-vs-fine-tuning
comparison). Metrics per run:

- schema_pass_rate: completion parsed and validated against CoachResponse
- safety_compliance_rate: no deterministic safety-rule violations
- severity_in_band_rate: severity inside the scenario's ground-truth band
- latency p50/p95 (wall clock per completion)

Per-scenario records are saved so two runs can be diffed and judged pairwise.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from pydantic import ValidationError

from cabin_copilot.quality.safety_rules import check_safety
from cabin_copilot.schemas import Scenario, in_band
from cabin_copilot.teacher.base import BackendError, LLMBackend
from cabin_copilot.teacher.generate import parse_response
from cabin_copilot.teacher.prompts import STUDENT_SYSTEM_PROMPT, TEACHER_SYSTEM_PROMPT

SYSTEM_PROMPTS = {"student": STUDENT_SYSTEM_PROMPT, "teacher": TEACHER_SYSTEM_PROMPT}


def evaluate_model(
    scenarios: list[Scenario],
    backend: LLMBackend,
    out_path: Path,
    *,
    system: str = "student",
    label: str = "",
) -> dict:
    system_prompt = SYSTEM_PROMPTS[system]
    records = []
    latencies = []
    n_schema = n_safe = n_band = 0

    for scenario in scenarios:
        started = time.perf_counter()
        raw, response, error = None, None, None
        try:
            raw = backend.complete(system_prompt, scenario.signals_json(), temperature=0.0)
            response = parse_response(raw)
        except (BackendError, ValidationError, ValueError, json.JSONDecodeError) as e:
            error = str(e)
        latency = time.perf_counter() - started
        latencies.append(latency)

        record = {
            "scenario_id": scenario.scenario_id,
            "taxonomy": scenario.taxonomy,
            "signals": json.loads(scenario.signals_json()),
            "raw": raw,
            "response": response.model_dump() if response else None,
            "error": error,
            "latency_s": round(latency, 3),
            "violations": [],
        }
        if response is not None:
            n_schema += 1
            violations = check_safety(scenario, response)
            record["violations"] = violations
            if not violations:
                n_safe += 1
            if in_band(
                response.severity, scenario.expected_severity_min, scenario.expected_severity_max
            ):
                n_band += 1
        records.append(record)

    n = len(scenarios)
    summary = {
        "label": label or backend.model,
        "model": backend.model,
        "system_prompt": system,
        "n_scenarios": n,
        "schema_pass_rate": round(n_schema / n, 4),
        "safety_compliance_rate": round(n_safe / n, 4),
        "severity_in_band_rate": round(n_band / n, 4),
        "latency_p50_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(sorted(latencies)[max(0, int(n * 0.95) - 1)], 3),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "records": records}, indent=2)
    )
    return summary
