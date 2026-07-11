"""LLM-judge scoring for data quality (single-sample rubric).

Pairwise judging for model comparison lives in ``cabin_copilot.eval.judge_pairwise`` —
the two are separate because they answer different questions ("is this training pair
good enough to learn from?" vs "which model's answer is better?").
"""

from __future__ import annotations

import json

from cabin_copilot.schemas import CoachResponse, Scenario
from cabin_copilot.teacher.base import LLMBackend
from cabin_copilot.teacher.generate import extract_json

JUDGE_SYSTEM_PROMPT = """\
You are a strict quality judge for an in-car driver-coaching dataset. You receive the
driver-monitoring signals and a candidate coaching response. Score the response 1-5:

5 = severity is right for the signals; message is natural spoken language, calm,
    specific to the situation; action matches.
4 = right severity and action, message is fine but generic.
3 = severity plausible but message is awkward, robotic, judgmental, or vague.
2 = severity off by one level, or message inappropriate for a driver mid-drive.
1 = severity clearly wrong, unsafe advice, or message that would erode driver trust.

Respond with ONLY JSON: {"score": <1-5>, "reason": "<one sentence>"}
"""


def judge_pair(
    backend: LLMBackend, scenario: Scenario, response: CoachResponse
) -> tuple[int, str]:
    user = (
        f"Signals: {scenario.signals_json()}\n"
        f"Candidate response: {json.dumps(response.model_dump())}"
    )
    raw = backend.complete(JUDGE_SYSTEM_PROMPT, user, temperature=0.0, max_tokens=200)
    data = extract_json(raw)
    return int(data["score"]), str(data.get("reason", ""))
