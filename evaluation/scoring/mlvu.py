#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd

from evaluation.scoring.perceptionbench import load_judge_verdicts, load_predictions

TASKS = ("plotQA", "findNeedle", "ego", "count", "order", "anomaly_reco", "topic_reasoning")
OPEN_ENDED = ("summary", "subPlot")


def compute_metrics(rows: list[dict], predictions: dict[str, dict], verdicts: dict) -> dict:
    by_task: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        task = str(row["category"])
        if task in OPEN_ENDED:
            continue
        if task not in TASKS:
            raise ValueError(f"Unknown MLVU task: {task}")
        record = predictions[str(row["qid"])]
        key = (str(record["questions"]), str(record["gts"]), str(record["final_answer"]))
        if key not in verdicts:
            raise KeyError(f"No judge verdict for qid={row['qid']}; run evaluation.scoring.judge first")
        by_task[task].append(int(verdicts[key]))
    missing = set(TASKS) - by_task.keys()
    if missing:
        raise ValueError(f"Missing MLVU tasks: {sorted(missing)}")
    tasks = {task: {"correct": sum(by_task[task]), "total": len(by_task[task]),
                    "accuracy": 100 * sum(by_task[task]) / len(by_task[task])} for task in TASKS}
    return {
        "m_avg": sum(item["accuracy"] for item in tasks.values()) / len(TASKS),
        "questions": sum(item["total"] for item in tasks.values()),
        "by_task": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    verdicts = load_judge_verdicts(args.log_dir)
    if verdicts is None:
        raise FileNotFoundError(f"No judge results under {args.log_dir}; run evaluation.scoring.judge first")
    metrics = compute_metrics(pd.read_parquet(args.data).to_dict("records"), load_predictions(args.log_dir), verdicts)
    output_path = args.log_dir / "mlvu_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
