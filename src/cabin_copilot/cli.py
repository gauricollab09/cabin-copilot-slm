"""Command-line pipeline: simulate -> generate -> judge -> filter.

Each stage reads/writes JSONL so any stage can be inspected, rerun, or resumed
independently — no hidden state between stages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cabin_copilot.quality.filter import run_filter
from cabin_copilot.schemas import CoachResponse, Scenario
from cabin_copilot.simulator import generate_scenarios
from cabin_copilot.teacher.base import PROVIDERS, make_backend
from cabin_copilot.teacher.generate import done_ids, generate_pairs


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=sorted(PROVIDERS), default="ollama")
    p.add_argument("--model", default=None, help="override the provider's default model")
    p.add_argument("--base-url", default=None, help="override the provider's base URL")


def cmd_simulate(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for scenario in generate_scenarios(args.n, args.seed, id_prefix=args.prefix):
            f.write(scenario.model_dump_json() + "\n")
    print(f"wrote {args.n} scenarios to {out}")


def _load_scenarios(path: Path) -> list[Scenario]:
    with path.open() as f:
        return [Scenario.model_validate_json(line) for line in f if line.strip()]


def cmd_generate(args: argparse.Namespace) -> None:
    backend = make_backend(args.provider, args.model, args.base_url)
    scenarios = _load_scenarios(Path(args.scenarios))
    stats = generate_pairs(
        scenarios, backend, Path(args.out), temperature=args.temperature
    )
    print(json.dumps({**stats, "usage": backend.usage.__dict__}, indent=2))


def cmd_judge(args: argparse.Namespace) -> None:
    from cabin_copilot.quality.judge import judge_pair

    backend = make_backend(args.provider, args.model, args.base_url)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    skip = done_ids(out)
    n_done = 0
    with Path(args.raw).open() as f, out.open("a") as sink:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["scenario_id"] in skip or not record.get("response"):
                continue
            scenario = Scenario.model_validate(record["scenario"])
            response = CoachResponse.model_validate(record["response"])
            try:
                score, reason = judge_pair(backend, scenario, response)
            except Exception as e:  # noqa: BLE001 — a bad judge call must not kill the run
                print(f"[judge] {record['scenario_id']}: {e}", file=sys.stderr)
                continue
            sink.write(
                json.dumps(
                    {"scenario_id": record["scenario_id"], "score": score, "reason": reason}
                )
                + "\n"
            )
            sink.flush()
            n_done += 1
            if n_done % 25 == 0:
                print(f"[judge] scored {n_done}", file=sys.stderr)
    print(f"judged {n_done} records -> {out}")


def cmd_filter(args: argparse.Namespace) -> None:
    judge_scores = None
    if args.judge:
        judge_scores = {}
        with Path(args.judge).open() as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    judge_scores[rec["scenario_id"]] = rec["score"]
    stats = run_filter(
        Path(args.raw),
        Path(args.out),
        judge_scores=judge_scores,
        judge_threshold=args.judge_threshold,
    )
    print(json.dumps(stats, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cabin-copilot")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("simulate", help="generate seeded DMS scenarios")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--prefix", default="scn")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("generate", help="run the teacher model over scenarios")
    p.add_argument("--scenarios", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    _add_backend_args(p)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("judge", help="rubric-score raw teacher outputs")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    _add_backend_args(p)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("filter", help="quality-gate raw pairs into a training dataset")
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--judge", default=None, help="judge scores JSONL from the judge command")
    p.add_argument("--judge-threshold", type=int, default=4)
    p.set_defaults(func=cmd_filter)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
