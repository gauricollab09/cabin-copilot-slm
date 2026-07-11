import pytest

from cabin_copilot.schemas import (
    Action,
    CabinState,
    CoachResponse,
    Drowsiness,
    Gaze,
    History,
    Phone,
    Scenario,
    Severity,
)


def make_scenario(**overrides) -> Scenario:
    defaults = dict(
        scenario_id="test-0001",
        taxonomy="benign_cruise",
        speed_kmh=80.0,
        trip_minutes=30,
        time_of_day="afternoon",
        drowsiness=Drowsiness(eye_closure_pct=8.0, yawn_count_5min=0, head_nod_events=0),
        gaze=Gaze(zone="windshield", off_road_ms=100),
        phone=Phone(in_hand=False, screen_on=False),
        cabin=CabinState(n_passengers=0, child_present=False, conversation=False),
        history=History(alerts_last_30min=0),
        expected_severity_min=Severity.NONE,
        expected_severity_max=Severity.NONE,
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def make_response(**overrides) -> CoachResponse:
    defaults = dict(
        severity=Severity.NONE, message="", action=Action.NONE, rationale="all clear"
    )
    defaults.update(overrides)
    return CoachResponse(**defaults)


@pytest.fixture
def scenario_factory():
    return make_scenario


@pytest.fixture
def response_factory():
    return make_response
