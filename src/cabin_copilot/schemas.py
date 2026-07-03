"""Core data contracts shared by every pipeline stage.

Two contracts matter:
- ``Scenario``: a structured snapshot of driver-monitoring signals (model input).
- ``CoachResponse``: the strict JSON the model must emit (model output).

``CoachResponse`` validates *structure* (types, enums, length bounds). Behavioural
constraints (e.g. "critical messages must be terse") are deterministic safety rules in
``cabin_copilot.quality.safety_rules`` so that schema-pass rate and safety compliance
can be measured independently during evaluation.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Severity(str, Enum):
    NONE = "none"
    INFO = "info"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"


SEVERITY_ORDER: dict[Severity, int] = {s: i for i, s in enumerate(Severity)}


class Action(str, Enum):
    NONE = "none"
    SUGGEST_BREAK = "suggest_break"
    REDUCE_DISTRACTION = "reduce_distraction"
    ALERT_DROWSINESS = "alert_drowsiness"
    PHONE_REMINDER = "phone_reminder"
    ESCALATE_ALARM = "escalate_alarm"


GazeZone = Literal[
    "windshield",
    "rearview_mirror",
    "left_mirror",
    "right_mirror",
    "instrument_cluster",
    "center_console",
    "phone",
    "passenger",
    "lap",
    "eyes_closed",
]

TimeOfDay = Literal["morning", "afternoon", "evening", "night", "late_night"]


class Drowsiness(BaseModel):
    eye_closure_pct: float = Field(ge=0, le=100, description="PERCLOS-style eye closure %")
    yawn_count_5min: int = Field(ge=0)
    head_nod_events: int = Field(ge=0, description="head-nod (micro-sleep) events in last 5 min")


class Gaze(BaseModel):
    zone: GazeZone
    off_road_ms: int = Field(ge=0, description="continuous gaze-off-road duration")


class Phone(BaseModel):
    in_hand: bool
    screen_on: bool


class CabinState(BaseModel):
    n_passengers: int = Field(ge=0, le=6)
    child_present: bool
    conversation: bool


class History(BaseModel):
    alerts_last_30min: int = Field(ge=0)


class Scenario(BaseModel):
    scenario_id: str
    taxonomy: str
    speed_kmh: float = Field(ge=0, le=200)
    trip_minutes: int = Field(ge=0)
    time_of_day: TimeOfDay
    drowsiness: Drowsiness
    gaze: Gaze
    phone: Phone
    cabin: CabinState
    history: History
    # Ground-truth severity band from the taxonomy cell. Never shown to the model;
    # used by the quality gate (hard reject) and the eval harness (severity accuracy).
    expected_severity_min: Severity
    expected_severity_max: Severity

    def signals_json(self) -> str:
        """Compact JSON of the signal fields only — exactly what the model sees."""
        payload = self.model_dump(
            exclude={"scenario_id", "taxonomy", "expected_severity_min", "expected_severity_max"}
        )
        return json.dumps(payload, separators=(",", ":"))


class CoachResponse(BaseModel):
    severity: Severity
    message: str = Field(max_length=200)
    action: Action
    rationale: str = Field(max_length=300)


def severity_at_least(a: Severity, b: Severity) -> bool:
    return SEVERITY_ORDER[a] >= SEVERITY_ORDER[b]


def in_band(sev: Severity, low: Severity, high: Severity) -> bool:
    return SEVERITY_ORDER[low] <= SEVERITY_ORDER[sev] <= SEVERITY_ORDER[high]
