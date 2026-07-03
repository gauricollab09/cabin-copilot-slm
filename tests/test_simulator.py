from collections import Counter

from cabin_copilot.schemas import SEVERITY_ORDER
from cabin_copilot.simulator import TAXONOMY, generate_scenarios


def test_deterministic_for_same_seed():
    a = [s.model_dump() for s in generate_scenarios(120, seed=42)]
    b = [s.model_dump() for s in generate_scenarios(120, seed=42)]
    assert a == b


def test_different_seeds_differ():
    a = [s.model_dump() for s in generate_scenarios(120, seed=1)]
    b = [s.model_dump() for s in generate_scenarios(120, seed=2)]
    assert a != b


def test_exact_count_and_full_coverage():
    scenarios = list(generate_scenarios(150, seed=7))
    assert len(scenarios) == 150
    covered = {s.taxonomy for s in scenarios}
    assert covered == {c.name for c in TAXONOMY}


def test_bands_are_valid_and_attached():
    for s in generate_scenarios(100, seed=3):
        assert SEVERITY_ORDER[s.expected_severity_min] <= SEVERITY_ORDER[s.expected_severity_max]


def test_cell_invariants():
    for s in generate_scenarios(300, seed=11):
        if s.taxonomy == "microsleep_critical":
            assert s.drowsiness.eye_closure_pct >= 60
            assert s.speed_kmh >= 60
        if s.taxonomy == "phone_moving":
            assert s.phone.in_hand and s.speed_kmh > 10
        if s.taxonomy == "stopped_phone":
            assert s.speed_kmh == 0
        if s.taxonomy == "alert_fatigue":
            assert s.history.alerts_last_30min >= 3
        if s.taxonomy == "benign_cruise":
            assert not s.phone.in_hand
            assert s.drowsiness.eye_closure_pct < 15


def test_weighting_shapes_distribution():
    counts = Counter(s.taxonomy for s in generate_scenarios(1000, seed=5))
    # benign_cruise has weight 1.5 vs stopped_phone 0.8 — must be strictly larger
    assert counts["benign_cruise"] > counts["stopped_phone"]
