import json

from cabin_copilot.quality.filter import normalise, run_filter, to_chat_example
from cabin_copilot.schemas import Action, Severity
from cabin_copilot.teacher.prompts import STUDENT_SYSTEM_PROMPT
from tests.conftest import make_response, make_scenario


def _record(scenario, response):
    return {
        "scenario_id": scenario.scenario_id,
        "taxonomy": scenario.taxonomy,
        "scenario": scenario.model_dump(),
        "response": response.model_dump() if response else None,
        "error": None if response else "boom",
        "raw": "",
        "teacher_model": "stub",
    }


def _write_raw(tmp_path, records):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r, default=str) for r in records) + "\n")
    return raw


def test_chat_example_uses_student_prompt():
    ex = to_chat_example(make_scenario(), make_response())
    assert ex["messages"][0]["content"] == STUDENT_SYSTEM_PROMPT
    assert json.loads(ex["messages"][2]["content"])["severity"] == "none"


def test_filter_rejects_safety_violations(tmp_path):
    good = (make_scenario(scenario_id="a"), make_response())
    bad = (
        make_scenario(scenario_id="b"),
        make_response(severity=Severity.INFO, message="Nice!", action=Action.NONE),
    )  # out of band for a benign_cruise scenario
    raw = _write_raw(tmp_path, [_record(*good), _record(*bad)])
    stats = run_filter(raw, tmp_path / "out")
    assert stats["accepted"] == 1
    assert stats["safety_reject"] == 1
    assert "severity_out_of_band" in stats["violations"]


def test_filter_drops_failed_and_dedups(tmp_path):
    s1 = make_scenario(
        scenario_id="a",
        taxonomy="mild_fatigue",
        expected_severity_min=Severity.INFO,
        expected_severity_max=Severity.CAUTION,
    )
    r1 = make_response(
        severity=Severity.INFO, message="Time for a coffee break?", action=Action.SUGGEST_BREAK
    )
    s2 = s1.model_copy(update={"scenario_id": "b"})
    r2 = make_response(
        severity=Severity.INFO, message="Time for a coffee break!?", action=Action.SUGGEST_BREAK
    )  # same after normalisation
    failed = make_scenario(scenario_id="c")
    raw = _write_raw(tmp_path, [_record(s1, r1), _record(s2, r2), _record(failed, None)])
    stats = run_filter(raw, tmp_path / "out")
    assert stats["no_response"] == 1
    assert stats["duplicate"] == 1
    assert stats["accepted"] == 1


def test_filter_applies_judge_scores(tmp_path):
    s = make_scenario(
        scenario_id="a",
        taxonomy="mild_fatigue",
        expected_severity_min=Severity.INFO,
        expected_severity_max=Severity.CAUTION,
    )
    r = make_response(
        severity=Severity.INFO, message="Feeling okay to keep going?", action=Action.SUGGEST_BREAK
    )
    raw = _write_raw(tmp_path, [_record(s, r)])
    stats = run_filter(raw, tmp_path / "out", judge_scores={"a": 2})
    assert stats["judge_reject"] == 1 and stats["accepted"] == 0


def test_normalise():
    assert normalise("Time for a break!?") == normalise("time for a break")
