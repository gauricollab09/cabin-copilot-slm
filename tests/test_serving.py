import json

from fastapi.testclient import TestClient

from cabin_copilot.serving.app import create_app
from tests.test_teacher import StubBackend

SIGNALS = {
    "speed_kmh": 80.0,
    "trip_minutes": 30,
    "time_of_day": "afternoon",
    "drowsiness": {"eye_closure_pct": 8.0, "yawn_count_5min": 0, "head_nod_events": 0},
    "gaze": {"zone": "windshield", "off_road_ms": 100},
    "phone": {"in_hand": False, "screen_on": False},
    "cabin": {"n_passengers": 0, "child_present": False, "conversation": False},
    "history": {"alerts_last_30min": 0},
}


def client(reply: str | None = None) -> TestClient:
    backend = StubBackend() if reply is None else StubBackend(reply=reply)
    return TestClient(create_app(backend=backend))


def test_healthz():
    resp = client().get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["model"] == "stub"


def test_coach_returns_contract():
    resp = client().post("/coach", json=SIGNALS)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"severity", "message", "action", "rationale"}
    assert body["severity"] == "none"


def test_coach_rejects_bad_signals():
    bad = dict(SIGNALS, speed_kmh=999)
    assert client().post("/coach", json=bad).status_code == 422


def test_coach_502_on_unparsable_model_output():
    resp = client(reply="I refuse to answer in JSON").post("/coach", json=SIGNALS)
    assert resp.status_code == 502


def test_coach_502_on_invalid_contract():
    reply = json.dumps(
        {"severity": "mega-urgent", "message": "x", "action": "none", "rationale": "y"}
    )
    assert client(reply=reply).post("/coach", json=SIGNALS).status_code == 502
