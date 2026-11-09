#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

PROTOCOLS = ("pixelreasoner", "thyme", "deepeyes", "deepeyesv2")
NAMES = {"pixelreasoner": "Pixel-Reasoner", "thyme": "Thyme", "deepeyes": "DeepEyes", "deepeyesv2": "DeepEyesV2"}
BRANCHES = ("direct", "natural", "gt", "random", "blank")
PAIRS = (("gt", "natural"), ("natural", "random"), ("natural", "blank"), ("gt", "blank"), ("natural", "direct"))


def observation_dependence(gt: float | None, random: float | None, blank: float | None) -> float | None:
    if None in (gt, random, blank):
        return None
    baseline = (random + blank) / 2
    return None if baseline >= 1 else (gt - baseline) / (1 - baseline)


def qid_means(records_path: Path) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latest = {}
    for line in records_path.read_text().splitlines():
        record = json.loads(line)
        latest[(record["qid"], record["branch"])] = record
    for record in latest.values():
        if record["status"] == "ok":
            values[record["branch"]][record["qid"]].append(record["correctness"])
    return {branch: {qid: sum(v) / len(v) for qid, v in per.items()} for branch, per in values.items()}


def bootstrap(diffs: list[float], draws: int = 10000, seed: int = 20261017) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    stats = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return stats[int(0.025 * draws)], stats[int(0.975 * draws) - 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    results = args.root / "dependence" if (args.root / "dependence").is_dir() else args.root
    out = []
    summaries, means = {}, {}
    for protocol in PROTOCOLS:
        summaries[protocol] = json.loads((results / protocol / "summary.json").read_text())
        means[protocol] = qid_means(results / protocol / "records.jsonl")
    out.append("| Model | direct | natural | gt | random | blank | Dependence | Natural call rate |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for protocol in PROTOCOLS:
        groups = summaries[protocol]["groups"]
        cells = [f"{groups[b]['accuracy'] * 100:.2f}" if groups[b]["accuracy"] is not None else "—" for b in BRANCHES]
        dependence = observation_dependence(*(groups[b]["accuracy"] for b in ("gt", "random", "blank")))
        cells.append(f"{dependence * 100:.2f}" if dependence is not None else "—")
        out.append(f"| {NAMES[protocol]} | " + " | ".join(cells) + f" | {groups['natural']['natural_call_rate'] * 100:.1f}% |")
    out.append("")
    out.append("| Model | " + " | ".join(f"{a}−{b}" for a, b in PAIRS) + " |")
    out.append("|---|" + "---:|" * len(PAIRS))
    for protocol in PROTOCOLS:
        cells = []
        for left, right in PAIRS:
            shared = sorted(means[protocol][left].keys() & means[protocol][right].keys())
            diffs = [means[protocol][left][q] - means[protocol][right][q] for q in shared]
            low, high = bootstrap(diffs)
            cells.append(f"{sum(diffs) / len(diffs) * 100:+.2f} [{low * 100:+.1f}, {high * 100:+.1f}]")
        out.append(f"| {NAMES[protocol]} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("| Model | Branch | Coverage | infra | blocked | unavailable | invalid | unsupported | re-call |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for protocol in PROTOCOLS:
        for branch in BRANCHES:
            g = summaries[protocol]["groups"][branch]
            out.append(f"| {NAMES[protocol]} | {branch} | {g['scored']}/{g['expected']} | {g['infra_errors']} | {g['blocked']} | "
                       f"{g['data_unavailable']} | {g['invalid_rate'] * 100:.1f}% | {g['unsupported_rate'] * 100:.1f}% | {g['re_call_rate'] * 100:.1f}% |")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
