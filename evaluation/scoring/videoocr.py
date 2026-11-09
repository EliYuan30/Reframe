#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from evaluation.scoring.answers import official_choice, strip_think


DEFAULT_DATA = str(Path(__file__).resolve().parents[2] / "outputs/benchmarks/videoocr.parquet")

BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^{}]*)\}")


def load_predictions(log_dir: Path) -> dict[str, dict]:
    predictions: dict[str, dict] = {}
    duplicate_rows = 0
    malformed_rows = 0
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
                    malformed_rows += 1
                    print(f"Ignoring malformed JSON at {path}:{line_number}")
                    continue
                qid = str(record["qids"])
                if qid in predictions:
                    duplicate_rows += 1
                    continue
                predictions[qid] = record

    if duplicate_rows:
        print(f"Ignored {duplicate_rows} distributed-padding rows")
    if malformed_rows:
        print(f"Ignored {malformed_rows} malformed rows")
    return predictions


def boxed_content(value) -> str:
    text = str(value or "")
    matches = BOXED_PATTERN.findall(text)
    return matches[-1].strip() if matches else strip_think(text).strip()


def normalize_text(value: str) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def letter_correct(prediction: str, reference: str) -> bool:
    expected = re.search(r"[A-Z]", boxed_content(reference).upper())
    if expected is None:
        raise ValueError(f"Invalid multiple-choice reference: {reference!r}")
    return official_choice(prediction, "ABCDEFGH") == expected.group(0)


def containment_correct(prediction: str, reference: str) -> bool:
    predicted = normalize_text(boxed_content(prediction))
    if not predicted:
        return False
    targets = [
        normalize_text(part)
        for part in boxed_content(reference).split(";")
        if normalize_text(part)
    ]
    if not targets:
        raise ValueError(f"Invalid containment reference: {reference!r}")
    return all(target in predicted for target in targets)


def is_correct(record: dict, row: dict) -> bool:
    prediction = record.get("final_answer")
    if str(row["eval_method"]) == "multiple_choice":
        return letter_correct(prediction, row["answer"])
    return containment_correct(prediction, row["answer"])


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--data", default=DEFAULT_DATA)
    args = parser.parse_args()

    predictions = load_predictions(args.log_dir)
    metadata = pd.read_parquet(args.data)
    metadata_by_qid = {str(row["qid"]): row for row in metadata.to_dict("records")}

    missing = sorted(set(metadata_by_qid) - set(predictions))
    unexpected = sorted(set(predictions) - set(metadata_by_qid))
    if missing or unexpected:
        raise ValueError(
            f"Prediction coverage mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    results: list[tuple[dict, bool]] = []
    tool_responses = 0
    tool_calls = 0
    execution_errors = 0
    for qid, row in metadata_by_qid.items():
        record = predictions[qid]
        solution = str(record.get("solutions", ""))
        calls = len(re.findall(r"<tool_call>", solution))
        tool_calls += calls
        tool_responses += int(calls > 0)
        execution_errors += int(
            bool(re.search(r"Execution Error:", solution, re.IGNORECASE))
        )
        results.append((row, is_correct(record, row)))

    by_task: dict[str, list[bool]] = defaultdict(list)
    by_method: dict[str, list[bool]] = defaultdict(list)
    for row, correct in results:
        by_task[str(row["category"])].append(correct)
        by_method[str(row["eval_method"])].append(correct)

    metrics = {
        "questions": len(results),
        "overall_accuracy": 100.0 * mean([float(correct) for _, correct in results]),
        "tool_response_rate": 100.0 * tool_responses / len(results),
        "tool_responses": tool_responses,
        "tool_calls": tool_calls,
        "execution_errors": execution_errors,
        "by_eval_method": {
            method: {
                "questions": len(scores),
                "accuracy": 100.0 * mean([float(value) for value in scores]),
            }
            for method, scores in sorted(by_method.items())
        },
        "by_task_type": {
            task: {
                "questions": len(scores),
                "accuracy": 100.0 * mean([float(value) for value in scores]),
            }
            for task, scores in sorted(by_task.items())
        },
    }

    output_path = args.log_dir / "videoocr_official_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
