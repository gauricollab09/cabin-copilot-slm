from cabin_copilot.quality.safety_rules import check_safety
from cabin_copilot.schemas import Action, Drowsiness, History, Phone, Severity


def test_clean_pair_passes(scenario_factory, response_factory):
    assert check_safety(scenario_factory(), response_factory()) == []


def test_critical_message_too_long(scenario_factory, response_factory):
    s = scenario_factory(
        expected_severity_min=Severity.CRITICAL, expected_severity_max=Severity.CRITICAL
    )
    r = response_factory(
        severity=Severity.CRITICAL,
        message="You really seem extremely tired so please find somewhere to stop",
        action=Action.ESCALATE_ALARM,
    )
    assert "critical_message_too_long" in check_safety(s, r)


def test_critical_wrong_action(scenario_factory, response_factory):
    s = scenario_factory(
        expected_severity_min=Severity.CRITICAL, expected_severity_max=Severity.CRITICAL
    )
    r = response_factory(
        severity=Severity.CRITICAL, message="Pull over now.", action=Action.SUGGEST_BREAK
    )
    assert "critical_wrong_action" in check_safety(s, r)


def test_none_severity_must_be_silent(scenario_factory, response_factory):
    r = response_factory(severity=Severity.NONE, message="Nice driving!", action=Action.NONE)
    assert "none_severity_nonempty_message" in check_safety(scenario_factory(), r)


def test_phone_while_moving_undercalled(scenario_factory, response_factory):
    s = scenario_factory(
        phone=Phone(in_hand=True, screen_on=True),
        speed_kmh=60.0,
        expected_severity_min=Severity.CAUTION,
        expected_severity_max=Severity.WARNING,
    )
    r = response_factory(severity=Severity.NONE)
    assert "phone_while_moving_undercalled" in check_safety(s, r)


def test_phone_while_stopped_is_fine(scenario_factory, response_factory):
    s = scenario_factory(phone=Phone(in_hand=True, screen_on=True), speed_kmh=0.0)
    assert "phone_while_moving_undercalled" not in check_safety(s, response_factory())


def test_severe_drowsiness_undercalled(scenario_factory, response_factory):
    s = scenario_factory(
        drowsiness=Drowsiness(eye_closure_pct=75.0, yawn_count_5min=4, head_nod_events=3),
        expected_severity_min=Severity.CRITICAL,
        expected_severity_max=Severity.CRITICAL,
    )
    r = response_factory(
        severity=Severity.INFO, message="Feeling sleepy?", action=Action.SUGGEST_BREAK
    )
    assert "severe_drowsiness_undercalled" in check_safety(s, r)


def test_screen_suggestion_while_moving(scenario_factory, response_factory):
    s = scenario_factory(
        speed_kmh=70.0,
        expected_severity_min=Severity.INFO,
        expected_severity_max=Severity.CAUTION,
    )
    r = response_factory(
        severity=Severity.INFO,
        message="Please check your phone for the fastest route.",
        action=Action.REDUCE_DISTRACTION,
    )
    assert "suggests_screen_interaction_while_moving" in check_safety(s, r)


def test_alert_fatigue_suppression(scenario_factory, response_factory):
    s = scenario_factory(
        history=History(alerts_last_30min=4),
        expected_severity_min=Severity.NONE,
        expected_severity_max=Severity.NONE,
    )
    r = response_factory(
        severity=Severity.INFO, message="You seem a bit tired.", action=Action.SUGGEST_BREAK
    )
    violations = check_safety(s, r)
    assert "alert_fatigue_not_suppressed" in violations


def test_severity_out_of_band(scenario_factory, response_factory):
    s = scenario_factory(
        expected_severity_min=Severity.WARNING, expected_severity_max=Severity.CRITICAL
    )
    r = response_factory(severity=Severity.NONE)
    assert "severity_out_of_band" in check_safety(s, r)
