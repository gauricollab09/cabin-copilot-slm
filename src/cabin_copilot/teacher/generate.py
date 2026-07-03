"""Teacher data generation: scenarios in, validated (scenario, response) pairs out.

Resumable by design: every accepted or failed record is appended to the output JSONL
immediately, and already-processed scenario_ids are skipped on restart — an interrupted
overnight run loses nothing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pydantic import ValidationError

from cabin_copilot.schemas import CoachResponse, Scenario
from cabin_copilot.teacher.base import BackendError, LLMBackend
from cabin_copilot.teacher.prompts import TEACHER_SYSTEM_PROMPT

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a possibly-noisy completion."""
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError("no JSON object found in completion")
    return json.loads(match.group(0))


def parse_response(text: str) -> CoachResponse:
    return CoachResponse.model_validate(extract_json(text))


def teacher_user_turn(scenario: Scenario, hint_band: bool) -> str:
    """The teacher's user message: signals, optionally with a calibration hint.

    Label-conditioned generation: the simulator's ground-truth severity band is shown
    to the TEACHER so it only has to choose phrasing/action/judgment inside the band.
    Training pairs still contain only the raw signals — the student never sees hints —
    and the deterministic safety gate independently re-checks the band on every output.
    """
    signals = scenario.signals_json()
    if not hint_band:
        return signals
    low, high = scenario.expected_severity_min.value, scenario.expected_severity_max.value
    band = f'"{low}"' if low == high else f'"{low}" and "{high}" (inclusive)'
    return (
        f"{signals}\n\nCalibration note (internal — never mention it): the appropriate "
        f"severity for this snapshot is {'exactly ' if low == high else 'between '}{band}."
    )


def done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    ids: set[str] = set()
    with out_path.open() as f:
        for line in f:
            try:
                ids.add(json.loads(line)["scenario_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def generate_pairs(
    scenarios: list[Scenario],
    backend: LLMBackend,
    out_path: Path,
    *,
    temperature: float = 0.7,
    max_attempts: int = 3,
    hint_band: bool = False,
    log=sys.stderr,
) -> dict:
    """Run the teacher over scenarios, appending JSONL records to out_path.

    Record shape: {scenario_id, taxonomy, scenario, response|null, error|null, raw}.
    Records with response=null are counted as failures and can be retried by rerunning.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    skip = done_ids(out_path)
    stats = {"total": len(scenarios), "skipped": len(skip), "ok": 0, "failed": 0}

    with out_path.open("a") as out:
        for i, scenario in enumerate(scenarios):
            if scenario.scenario_id in skip:
                continue
            record = {
                "scenario_id": scenario.scenario_id,
                "taxonomy": scenario.taxonomy,
                "scenario": scenario.model_dump(),
                "response": None,
                "error": None,
                "raw": None,
                "teacher_model": backend.model,
            }
            last_err = None
            for _ in range(max_attempts):
                try:
                    raw = backend.complete(
                        TEACHER_SYSTEM_PROMPT,
                        teacher_user_turn(scenario, hint_band),
                        temperature=temperature,
                    )
                    record["raw"] = raw
                    record["response"] = parse_response(raw).model_dump()
                    break
                except (BackendError, ValidationError, ValueError, json.JSONDecodeError) as e:
                    last_err = str(e)
            if record["response"] is None:
                record["error"] = last_err
                stats["failed"] += 1
            else:
                stats["ok"] += 1
            out.write(json.dumps(record) + "\n")
            out.flush()
            if (i + 1) % 25 == 0:
                print(
                    f"[teacher] {i + 1}/{len(scenarios)} ok={stats['ok']} "
                    f"failed={stats['failed']} tokens_out={backend.usage.output_tokens}",
                    file=log,
                )
    return stats
