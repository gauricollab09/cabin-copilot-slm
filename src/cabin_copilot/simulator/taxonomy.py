"""Safety taxonomy: the cells of driver/cabin state the dataset must cover.

Each cell defines a sampler over signal ranges plus the ground-truth severity band a
correct coaching response must fall in. Bands are deliberately widths of 0–2 levels:
unambiguous cells (microsleep at highway speed) pin the response, ambiguous cells
(possible sensor false-positive) leave room for the teacher model to reason.

The taxonomy is where domain knowledge lives — thresholds follow common DMS practice
(PERCLOS-style eye-closure bands, gaze-off-road duration vs speed, alert-fatigue
suppression) without reproducing any proprietary system.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

from cabin_copilot.schemas import (
    CabinState,
    Drowsiness,
    Gaze,
    History,
    Phone,
    Severity,
    TimeOfDay,
)

DAY_TIMES: list[TimeOfDay] = ["morning", "afternoon", "evening"]
ALL_TIMES: list[TimeOfDay] = ["morning", "afternoon", "evening", "night", "late_night"]


def _base(rng: random.Random, **overrides) -> dict:
    """Benign defaults; cells override the fields that define them."""
    state = dict(
        speed_kmh=round(rng.uniform(40, 100), 1),
        trip_minutes=rng.randint(5, 60),
        time_of_day=rng.choice(ALL_TIMES),
        drowsiness=Drowsiness(
            eye_closure_pct=round(rng.uniform(3, 14), 1),
            yawn_count_5min=rng.randint(0, 1),
            head_nod_events=0,
        ),
        gaze=Gaze(zone="windshield", off_road_ms=rng.randint(0, 250)),
        phone=Phone(in_hand=False, screen_on=False),
        cabin=CabinState(n_passengers=rng.randint(0, 2), child_present=False, conversation=False),
        history=History(alerts_last_30min=0),
    )
    state.update(overrides)
    return state


def _benign_cruise(rng: random.Random) -> dict:
    return _base(rng)


def _mirror_check(rng: random.Random) -> dict:
    return _base(
        rng,
        gaze=Gaze(
            zone=rng.choice(["rearview_mirror", "left_mirror", "right_mirror"]),
            off_road_ms=rng.randint(300, 1100),
        ),
    )


def _stopped_phone(rng: random.Random) -> dict:
    return _base(
        rng,
        speed_kmh=0.0,
        gaze=Gaze(zone="phone", off_road_ms=rng.randint(2000, 15000)),
        phone=Phone(in_hand=True, screen_on=True),
    )


def _mild_fatigue(rng: random.Random) -> dict:
    return _base(
        rng,
        trip_minutes=rng.randint(90, 180),
        drowsiness=Drowsiness(
            eye_closure_pct=round(rng.uniform(16, 30), 1),
            yawn_count_5min=rng.randint(2, 4),
            head_nod_events=0,
        ),
    )


def _drowsy_moderate(rng: random.Random) -> dict:
    return _base(
        rng,
        trip_minutes=rng.randint(60, 240),
        time_of_day=rng.choice(["evening", "night", "late_night"]),
        drowsiness=Drowsiness(
            eye_closure_pct=round(rng.uniform(35, 55), 1),
            yawn_count_5min=rng.randint(3, 6),
            head_nod_events=rng.randint(1, 2),
        ),
    )


def _microsleep_critical(rng: random.Random) -> dict:
    return _base(
        rng,
        speed_kmh=round(rng.uniform(60, 120), 1),
        trip_minutes=rng.randint(60, 300),
        time_of_day=rng.choice(["night", "late_night"]),
        drowsiness=Drowsiness(
            eye_closure_pct=round(rng.uniform(60, 95), 1),
            yawn_count_5min=rng.randint(2, 8),
            head_nod_events=rng.randint(2, 5),
        ),
        gaze=Gaze(zone="eyes_closed", off_road_ms=rng.randint(1500, 4000)),
    )


def _gaze_offroad_short(rng: random.Random) -> dict:
    return _base(
        rng,
        speed_kmh=round(rng.uniform(30, 70), 1),
        gaze=Gaze(
            zone=rng.choice(["center_console", "instrument_cluster", "passenger"]),
            off_road_ms=rng.randint(800, 1600),
        ),
    )


def _gaze_offroad_long(rng: random.Random) -> dict:
    return _base(
        rng,
        speed_kmh=round(rng.uniform(50, 110), 1),
        gaze=Gaze(
            zone=rng.choice(["center_console", "passenger", "lap"]),
            off_road_ms=rng.randint(2000, 5000),
        ),
    )


def _phone_moving(rng: random.Random) -> dict:
    return _base(
        rng,
        speed_kmh=round(rng.uniform(30, 110), 1),
        gaze=Gaze(
            zone=rng.choice(["phone", "lap", "windshield"]),
            off_road_ms=rng.randint(500, 3000),
        ),
        phone=Phone(in_hand=True, screen_on=rng.random() < 0.7),
    )


def _child_distraction(rng: random.Random) -> dict:
    return _base(
        rng,
        speed_kmh=round(rng.uniform(30, 90), 1),
        gaze=Gaze(
            zone=rng.choice(["passenger", "rearview_mirror"]),
            off_road_ms=rng.randint(1000, 3000),
        ),
        cabin=CabinState(n_passengers=rng.randint(1, 3), child_present=True, conversation=True),
    )


def _passenger_chat(rng: random.Random) -> dict:
    return _base(
        rng,
        cabin=CabinState(n_passengers=rng.randint(1, 3), child_present=False, conversation=True),
        gaze=Gaze(zone="windshield", off_road_ms=rng.randint(0, 400)),
    )


def _long_trip(rng: random.Random) -> dict:
    return _base(
        rng,
        trip_minutes=rng.randint(180, 360),
        drowsiness=Drowsiness(
            eye_closure_pct=round(rng.uniform(12, 24), 1),
            yawn_count_5min=rng.randint(1, 3),
            head_nod_events=0,
        ),
    )


def _alert_fatigue(rng: random.Random) -> dict:
    return _base(
        rng,
        trip_minutes=rng.randint(60, 200),
        drowsiness=Drowsiness(
            eye_closure_pct=round(rng.uniform(14, 26), 1),
            yawn_count_5min=rng.randint(1, 3),
            head_nod_events=0,
        ),
        history=History(alerts_last_30min=rng.randint(3, 6)),
    )


def _night_fatigue(rng: random.Random) -> dict:
    return _base(
        rng,
        time_of_day="late_night",
        trip_minutes=rng.randint(45, 240),
        drowsiness=Drowsiness(
            eye_closure_pct=round(rng.uniform(25, 45), 1),
            yawn_count_5min=rng.randint(2, 5),
            head_nod_events=rng.randint(0, 1),
        ),
    )


def _sensor_ambiguity(rng: random.Random) -> dict:
    # High eye-closure reading with zero corroborating signals (no yawns, no nods,
    # short daytime trip) — the false-positive-prone case a naive system nags on.
    return _base(
        rng,
        time_of_day=rng.choice(DAY_TIMES),
        trip_minutes=rng.randint(5, 30),
        drowsiness=Drowsiness(
            eye_closure_pct=round(rng.uniform(40, 60), 1),
            yawn_count_5min=0,
            head_nod_events=0,
        ),
    )


@dataclass(frozen=True)
class TaxonomyCell:
    name: str
    description: str
    weight: float
    band: tuple[Severity, Severity]
    sampler: Callable[[random.Random], dict]


TAXONOMY: list[TaxonomyCell] = [
    TaxonomyCell(
        "benign_cruise", "attentive driving, nothing to say", 1.5,
        (Severity.NONE, Severity.NONE), _benign_cruise,
    ),
    TaxonomyCell(
        "mirror_check", "normal mirror scanning — must not be flagged", 1.0,
        (Severity.NONE, Severity.NONE), _mirror_check,
    ),
    TaxonomyCell(
        "stopped_phone", "phone in hand while stationary (red light)", 0.8,
        (Severity.NONE, Severity.INFO), _stopped_phone,
    ),
    TaxonomyCell(
        "passenger_chat", "conversation with eyes on road", 0.8,
        (Severity.NONE, Severity.INFO), _passenger_chat,
    ),
    TaxonomyCell(
        "sensor_ambiguity", "high eye-closure reading with no corroboration", 0.8,
        (Severity.NONE, Severity.CAUTION), _sensor_ambiguity,
    ),
    TaxonomyCell(
        "alert_fatigue", "mild signals but >=3 recent alerts — suppress nagging", 0.8,
        (Severity.NONE, Severity.NONE), _alert_fatigue,
    ),
    TaxonomyCell(
        "mild_fatigue", "early fatigue signs on a long-ish drive", 1.0,
        (Severity.INFO, Severity.CAUTION), _mild_fatigue,
    ),
    TaxonomyCell(
        "long_trip", "3-6h trip, subtle fatigue", 0.8,
        (Severity.INFO, Severity.CAUTION), _long_trip,
    ),
    TaxonomyCell(
        "gaze_offroad_short", "brief glance away at moderate speed", 1.0,
        (Severity.INFO, Severity.CAUTION), _gaze_offroad_short,
    ),
    TaxonomyCell(
        "night_fatigue", "late-night drive with moderate fatigue", 1.0,
        (Severity.CAUTION, Severity.WARNING), _night_fatigue,
    ),
    TaxonomyCell(
        "child_distraction", "attending to a child while moving", 0.8,
        (Severity.CAUTION, Severity.WARNING), _child_distraction,
    ),
    TaxonomyCell(
        "phone_moving", "phone in hand while driving", 1.2,
        (Severity.CAUTION, Severity.WARNING), _phone_moving,
    ),
    TaxonomyCell(
        "gaze_offroad_long", "sustained gaze off road at speed", 1.0,
        (Severity.WARNING, Severity.CRITICAL), _gaze_offroad_long,
    ),
    TaxonomyCell(
        "drowsy_moderate", "clear drowsiness pattern", 1.0,
        (Severity.WARNING, Severity.WARNING), _drowsy_moderate,
    ),
    TaxonomyCell(
        "microsleep_critical", "micro-sleep at speed — immediate intervention", 1.0,
        (Severity.CRITICAL, Severity.CRITICAL), _microsleep_critical,
    ),
]
