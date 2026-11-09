#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from evaluation.adapters.common import benchmark_output_root
from evaluation.scoring.answers import boxed_content


def load_predictions(log_dir: Path) -> dict[str, dict]:
    predictions: dict[str, dict] = {}
    files = sorted(log_dir.glob("sample.eval_iter0_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No evaluation JSONL files under {log_dir}")
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Ignoring malformed JSON at {path}:{line_number}")
                    continue
                predictions.setdefault(str(record["qids"]), record)
    return predictions


JUDGE_CACHES = ("gemini_judgments.jsonl", "llm_judgments.jsonl")


def normalize(value: Any) -> str:
    text = str(value or "").strip()
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"^\\text\s*\{(.*)\}$", r"\1", text, flags=re.DOTALL).strip()
    text = text.replace("$", "").strip()
    text = re.sub(r"\s+", " ", text).lower()
    text = text.strip(" .;:!\"'`")
    match = re.fullmatch(r"\(?([a-z])\)?[.)]?", text)
    if match:
        return match.group(1)
    number = text.replace(",", "")
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", number):
        return repr(float(number))
    return text


def exact_match(final_answer: Any, reference: Any) -> bool:
    content = boxed_content(final_answer)
    return content is not None and normalize(content) == normalize(reference)


def load_judge_verdicts(log_dir: Path) -> dict[tuple[str, str, str], int] | None:
    metrics_path = log_dir / "llm_judge_metrics.json"
    if not metrics_path.is_file():
        return None
    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
    verdicts: dict[tuple[str, str, str], int] = {}
    for name in JUDGE_CACHES:
        path = log_dir / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if (
                    entry.get("judge_model") != summary.get("judge_model")
                    or entry.get("judge_thinking_level") != summary.get("judge_thinking_level")
                ):
                    continue
                key = (str(entry["question"]), str(entry["reference"]), str(entry["candidate"]))
                verdicts[key] = int(entry["verdict"])
    return verdicts or None


def breakdown(rows: dict[str, dict], predictions: dict[str, dict], verdicts) -> dict:
    exact: dict[str, list[float]] = defaultdict(list)
    judged: dict[str, list[float]] = defaultdict(list)
    unjudged = 0
    for qid, row in rows.items():
        record = predictions[qid]
        category = str(row["category"])
        exact[category].append(float(exact_match(record.get("final_answer"), row["answer"])))
        if verdicts is not None:
            key = (str(record.get("questions")), str(record.get("gts")), str(record.get("final_answer")))
            if key in verdicts:
                judged[category].append(float(verdicts[key]))
            else:
                unjudged += 1

    def summary(values: dict[str, list[float]]) -> dict:
        flat = [value for items in values.values() for value in items]
        return {
            "overall": 100.0 * sum(flat) / len(flat) if flat else 0.0,
            "by_category": {
                name: {"accuracy": 100.0 * sum(items) / len(items), "total": len(items)}
                for name, items in sorted(values.items())
            },
        }

    result = {"questions": len(rows), "exact_match": summary(exact)}
    if verdicts is not None:
        result["llm_judge"] = {**summary(judged), "unjudged": unjudged}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=benchmark_output_root() / "perceptionbench.parquet")
    args = parser.parse_args()

    predictions = load_predictions(args.log_dir)
    rows = {str(row["qid"]): row for row in pd.read_parquet(args.data).to_dict("records")}
    missing = sorted(set(rows) - set(predictions))
    unexpected = sorted(set(predictions) - set(rows))
    if missing or unexpected:
        raise ValueError(f"Prediction coverage mismatch: missing={len(missing)}, unexpected={len(unexpected)}")

    metrics = breakdown(rows, predictions, load_judge_verdicts(args.log_dir))
    output_path = args.log_dir / "perceptionbench_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
