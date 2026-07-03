import json
from dataclasses import dataclass, field

import pytest

from cabin_copilot.simulator import generate_scenarios
from cabin_copilot.teacher.base import LLMBackend, Usage
from cabin_copilot.teacher.generate import done_ids, extract_json, generate_pairs


@dataclass
class StubBackend(LLMBackend):
    """Deterministic offline backend for tests and dry runs."""

    model: str = "stub"
    reply: str = json.dumps(
        {"severity": "none", "message": "", "action": "none", "rationale": "stub"}
    )
    fail_ids: set = field(default_factory=set)
    calls: int = 0

    def _request(self, system, user, temperature, max_tokens):
        self.calls += 1
        if any(fid in user for fid in self.fail_ids):
            return "I cannot help with that."
        return self.reply

    def __post_init__(self):
        self.usage = Usage()


def test_extract_json_from_noisy_text():
    raw = (
        "Sure! Here is the JSON:\n```json\n"
        '{"severity": "info", "message": "hi",\n "action": "none", "rationale": "x"}\n```'
    )
    assert extract_json(raw)["severity"] == "info"


def test_extract_json_raises_when_absent():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_generate_pairs_writes_records(tmp_path):
    scenarios = list(generate_scenarios(10, seed=1))
    out = tmp_path / "raw.jsonl"
    stats = generate_pairs(scenarios, StubBackend(), out)
    assert stats["ok"] == 10 and stats["failed"] == 0
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(lines) == 10
    assert all(rec["response"]["severity"] == "none" for rec in lines)


def test_generate_pairs_resumes(tmp_path):
    scenarios = list(generate_scenarios(10, seed=1))
    out = tmp_path / "raw.jsonl"
    generate_pairs(scenarios[:4], StubBackend(), out)
    backend = StubBackend()
    stats = generate_pairs(scenarios, backend, out)
    assert stats["skipped"] == 4
    assert backend.calls == 6
    assert len(done_ids(out)) == 10


def test_generate_pairs_records_failures(tmp_path):
    scenarios = list(generate_scenarios(5, seed=1))
    bad_id_fragment = json.loads(scenarios[0].signals_json())  # noqa: F841 — sanity parse
    backend = StubBackend(reply="not json at all")
    out = tmp_path / "raw.jsonl"
    stats = generate_pairs(scenarios, backend, out, max_attempts=2)
    assert stats["failed"] == 5
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert all(rec["response"] is None and rec["error"] for rec in lines)
