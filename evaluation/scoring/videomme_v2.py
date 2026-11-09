#!/usr/bin/env python3
import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

from evaluation.scoring.answers import official_choice


DEFAULT_METADATA = (
    "/path/to/your/data/"
    "datasets--MME-Benchmarks--Video-MME-v2/test.parquet"
)


def relevance_score(scores: list[bool]) -> float:
    correct = sum(scores)
    return {0: 0.0, 1: 6.25, 2: 25.0, 3: 56.25, 4: 100.0}[correct]


def logic_score(scores: list[bool], structure: str) -> float:
    structure_value = ast.literal_eval(structure)
    last_correct = -1
    for index, correct in enumerate(scores):
        if not correct:
            break
        last_correct = index

    if structure_value == [1, 2, 3, 4]:
        score_map = {0: 0.0, 1: 6.25, 2: 25.0, 3: 56.25, 4: 100.0}
    elif structure_value == [1, [2, 3], 4]:
        score_map = {
            0: 0.0,
            1: 100.0 / 12,
            2: 400.0 / 12,
            3: 700.0 / 12,
            4: 100.0,
        }
        if last_correct == 0 and scores[2]:
            last_correct += 1
    elif structure_value == [[1, 2], 3, 4]:
        score_map = {0: 0.0, 1: 10.0, 2: 20.0, 3: 50.0, 4: 100.0}
        if last_correct == -1 and scores[1]:
            last_correct += 1
    else:
        raise ValueError(f"Unknown group structure: {structure_value}")

    return score_map[last_correct + 1]


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_predictions(log_dir: Path) -> dict[str, dict]:
    predictions = {}
    duplicate_rows = 0
    malformed_rows = 0
    files = sorted(log_dir.glob("sample.eval_iter0_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No evaluation JSONL files under {log_dir}")

    for path in files:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    malformed_rows += 1
                    print(f"Ignoring malformed JSON at {path}:{line_number}")
                    continue
                qid = record["qids"]
                if qid in predictions:
                    duplicate_rows += 1
                    continue
                predictions[qid] = record

    if duplicate_rows:
        print(f"Ignored {duplicate_rows} distributed-padding rows")
    if malformed_rows:
        print(f"Ignored {malformed_rows} malformed rows")
    return predictions


ANSWER_PREFIXES = (
    "Final Answer:",
    "The best answer is",
    "The correct answer is",
    "The answer is",
    "The answer",
    "The best option is",
    "The correct option is",
    "Best answer:",
    "Best option:",
    "Answer:",
    "Option:",
)


BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]*)\}")


def extract_answer(value: str) -> str:
    value = str(value).strip()
    boxed = BOXED_PATTERN.findall(value)
    if boxed:
        match = re.search(r"[A-H]", boxed[-1].strip().upper())
        return match.group(0) if match else ""
    for prefix in ANSWER_PREFIXES:
        value = value.replace(prefix, "")
    if len(value.split()) > 10 and not re.search(r"[A-H]", value):
        return ""
    match = re.search(r"[A-H]", value)
    return match.group(0) if match else ""


def final_correctness(record: dict, expected: str) -> bool:
      return official_choice(record.get("solutions"), "ABCDEFGH") == str(expected).strip().upper()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    args = parser.parse_args()

    predictions = load_predictions(args.log_dir)
    metadata = pd.read_parquet(args.metadata)
    metadata_by_qid = {
        f"videomme_v2-{row.question_id}": row
        for row in metadata.itertuples(index=False)
    }

    missing = sorted(set(metadata_by_qid) - set(predictions))
    unexpected = sorted(set(predictions) - set(metadata_by_qid))
    if missing or unexpected:
        raise ValueError(
            f"Prediction coverage mismatch: missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    question_results = []
    tool_responses = 0
    tool_calls = 0
    execution_errors = 0
    for qid, row in metadata_by_qid.items():
        record = predictions[qid]
        solution = str(record.get("solutions", ""))
        calls = len(re.findall(r"<tool_call>", solution))
        tool_calls += calls
        tool_responses += int(calls > 0)
        execution_errors += int(bool(re.search(r"Execution Error:", solution, re.IGNORECASE)))
        question_results.append((row, final_correctness(record, row.answer)))

    groups = defaultdict(list)
    for row, correct in question_results:
        groups[str(row.video_id)].append((row, correct))

    group_results = []
    for video_id, items in groups.items():
        items.sort(key=lambda item: int(str(item[0].question_id).split("-")[-1]))
        scores = [correct for _, correct in items]
        row = items[-1][0]
        if len(scores) != 4:
            raise ValueError(f"Video {video_id} has {len(scores)} questions")
        if row.group_type == "relevance":
            score = relevance_score(scores)
        elif row.group_type == "logic":
            score = logic_score(scores, row.group_structure)
        else:
            raise ValueError(f"Unknown group type: {row.group_type}")
        group_results.append((row, score))

    metrics = {
        "questions": len(question_results),
        "videos": len(group_results),
        "overall_accuracy": 100.0
        * mean([float(correct) for _, correct in question_results]),
        "overall_group_score": mean([score for _, score in group_results]),
        "relevance_group_score": mean(
            [score for row, score in group_results if row.group_type == "relevance"]
        ),
        "logic_group_score": mean(
            [score for row, score in group_results if row.group_type == "logic"]
        ),
        "tool_response_rate": 100.0 * tool_responses / len(question_results),
        "tool_responses": tool_responses,
        "tool_calls": tool_calls,
        "execution_errors": execution_errors,
    }
    for level in ("1", "2", "3"):
        metrics[f"level_{level}_group_score"] = mean(
            [
                score
                for row, score in group_results
                if str(row.level) == level
            ]
        )

    output_path = args.log_dir / "videomme_v2_official_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()