import json

import pytest
from pydantic import ValidationError

from cabin_copilot.schemas import CoachResponse
from cabin_copilot.simulator import generate_scenarios


def test_signals_json_hides_ground_truth():
    scenario = next(iter(generate_scenarios(1, seed=1)))
    payload = json.loads(scenario.signals_json())
    assert "expected_severity_min" not in payload
    assert "expected_severity_max" not in payload
    assert "taxonomy" not in payload
    assert "scenario_id" not in payload
    assert "speed_kmh" in payload and "drowsiness" in payload


def test_coach_response_rejects_bad_enum():
    with pytest.raises(ValidationError):
        CoachResponse(severity="urgent", message="", action="none", rationale="x")


def test_coach_response_rejects_missing_field():
    with pytest.raises(ValidationError):
        CoachResponse.model_validate({"severity": "none", "message": ""})
