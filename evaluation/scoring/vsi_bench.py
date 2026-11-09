#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.scoring.answers import strip_think


DEFAULT_DATA = str(Path(__file__).resolve().parents[2] / "outputs/benchmarks/vsi_bench.parquet")
BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^{}]*)\}")
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}


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


def boxed_content(value: object) -> str:
    text = str(value or "").strip()
    matches = BOXED_PATTERN.findall(text)
    return matches[-1].strip() if matches else strip_think(text)


def extract_choice(value: object) -> str:
    text = boxed_content(value).strip().upper()
    match = re.fullmatch(r"(?:OPTION\s+)?\(?([A-D])\)?[.)]?", text)
    if match:
        return match.group(1)
    match = re.search(r"(?:ANSWER|OPTION)(?:\s+IS|:)\s*\(?([A-D])\)?", text)
    return match.group(1) if match else ""


def extract_number(value: object) -> float | None:
    match = NUMBER_PATTERN.search(boxed_content(value).replace(",", ""))
    return float(match.group(0)) if match else None


def mean_relative_accuracy(prediction: float | None, target: float) -> float:
    if prediction is None:
        return 0.0
    if target == 0:
        return float(prediction == 0)
    relative_error = abs(prediction - target) / abs(target)
    thresholds = np.linspace(0.5, 0.95, 10)
    return float(np.mean(relative_error <= 1.0 - thresholds))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate_scores(
    by_type: dict[str, list[float]],
) -> tuple[float, dict[str, dict[str, float | int | str]]]:
    expected = MCA_QUESTION_TYPES | NA_QUESTION_TYPES
    missing = expected - set(by_type)
    if missing:
        raise ValueError(f"Missing VSI-Bench question types: {sorted(missing)}")

    type_scores = {name: mean(scores) for name, scores in by_type.items()}
    direction_types = (
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
    )
    direction_score = mean([type_scores.pop(name) for name in direction_types])
    type_scores["object_rel_direction"] = direction_score

    diagnostics = {}
    for name, score in sorted(type_scores.items()):
        if name == "object_rel_direction":
            count = sum(len(by_type[item]) for item in direction_types)
            metric = "accuracy"
        else:
            count = len(by_type[name])
            metric = "accuracy" if name in MCA_QUESTION_TYPES else "MRA:.5:.95:.05"
        diagnostics[name] = {
            "questions": count,
            "score": 100.0 * score,
            "metric": metric,
        }
    return 100.0 * mean(list(type_scores.values())), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--data", default=DEFAULT_DATA)
    args = parser.parse_args()

    predictions = load_predictions(args.log_dir)
    metadata = pd.read_parquet(args.data)
    rows = {str(row["qid"]): row for row in metadata.to_dict("records")}
    missing = sorted(set(rows) - set(predictions))
    unexpected = sorted(set(predictions) - set(rows))
    if missing or unexpected:
        raise ValueError(
            f"Prediction coverage mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    by_type: dict[str, list[float]] = defaultdict(list)
    tool_responses = 0
    tool_calls = 0
    execution_errors = 0
    for qid, row in rows.items():
        prediction = predictions[qid]
        final_answer = prediction.get("final_answer", "")
        question_type = str(row["category"])
        if question_type in MCA_QUESTION_TYPES:
            score = float(extract_choice(final_answer) == extract_choice(row["answer"]))
        elif question_type in NA_QUESTION_TYPES:
            target = extract_number(row["answer"])
            if target is None:
                raise ValueError(f"Invalid numerical answer for {qid}: {row['answer']!r}")
            score = mean_relative_accuracy(extract_number(final_answer), target)
        else:
            raise ValueError(f"Unknown VSI-Bench question type: {question_type}")
        by_type[question_type].append(score)

        solution = str(prediction.get("solutions", ""))
        calls = len(re.findall(r"<tool_call>", solution))
        tool_calls += calls
        tool_responses += int(calls > 0)
        execution_errors += int("Execution Error:" in solution)

    overall, diagnostics = aggregate_scores(by_type)
    metrics = {
        "questions": len(rows),
        "overall": overall,
        "tool_response_rate": 100.0 * tool_responses / len(rows),
        "tool_responses": tool_responses,
        "tool_calls": tool_calls,
        "execution_errors": execution_errors,
        "by_question_type": diagnostics,
    }

    output_path = args.log_dir / "vsi_bench_official_metrics.json"
    output_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
