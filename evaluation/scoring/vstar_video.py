#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import re
from statistics import mean
import time
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm
from evaluation.scoring import JUDGE_BACKEND, JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_THINKING_LEVEL


TASKS = {"vqa", "temporal_1", "spatial_1", "temporal_2", "spatial_2"}
IOU_THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.9)
JUDGE_INSTRUCTION = """You evaluate a candidate answer against a reference answer.
Use only semantic correctness for the given question and ignore grammatical differences.
Return one integer and nothing else:
0 = entirely incorrect
1 = largely incorrect
2 = largely correct
3 = completely correct"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(str(Path(__file__).resolve().parents[2] / "outputs/benchmarks/vstar_video.parquet")),
    )
    parser.set_defaults(judge_model=JUDGE_MODEL)
    parser.add_argument("--judge-workers", type=int, default=8)
    return parser.parse_args()


def load_predictions(log_dir: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for path in sorted(log_dir.glob("sample.eval_iter0_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    records.setdefault(str(record["qids"]), record)
    if not records:
        raise ValueError(f"No predictions found under {log_dir}")
    return records


def boxed_content(text: Any) -> str:
    text = "" if text is None else str(text)
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    if not starts:
        return text.strip()
    start = starts[-1].end()
    depth = 1
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start:index].strip()
    return text[start:].strip()


def parse_literal(text: Any) -> Any:
    value = boxed_content(text)
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL)
    if fenced:
        value = fenced.group(1)
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        match = re.search(r"(\[[\s\S]*\])", value)
        if not match:
            return None
        try:
            return ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            return None


def parse_temporal(text: Any) -> list[float] | None:
    value = parse_literal(text)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or not math.isfinite(end) or end < start:
        return None
    return [start, end]


def parse_spatial(text: Any) -> dict[int, list[list[float]]]:
    value = parse_literal(text)
    if not isinstance(value, (list, tuple)):
        return {}
    result: dict[int, list[list[float]]] = {}
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) != 5:
            continue
        try:
            second_value = float(entry[0])
            second = int(round(second_value))
            box = [float(coordinate) for coordinate in entry[1:]]
        except (TypeError, ValueError):
            continue
        if abs(second_value - second) > 1e-6:
            continue
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            continue
        result.setdefault(second, []).append(box)
    return result


def temporal_iou(target: list[float], prediction: list[float] | None) -> float:
    if prediction is None:
        return 0.0
    intersection = max(0.0, min(target[1], prediction[1]) - max(target[0], prediction[0]))
    union = max(target[1], prediction[1]) - min(target[0], prediction[0])
    return intersection / union if union > 0 else 0.0


def box_iou(target: list[float], prediction: list[float]) -> float:
    x1 = max(target[0], prediction[0])
    y1 = max(target[1], prediction[1])
    x2 = min(target[2], prediction[2])
    y2 = min(target[3], prediction[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    prediction_area = (prediction[2] - prediction[0]) * (prediction[3] - prediction[1])
    union = target_area + prediction_area - intersection
    return intersection / union if union > 0 else 0.0


def spatial_ious(row: dict[str, Any], prediction: dict[int, list[list[float]]]) -> list[float]:
    width, height = float(row["width"]), float(row["height"])
    values = []
    for target in json.loads(row["gt_bboxes"]):
        target_box = [
            1000 * float(target["xmin"]) / width,
            1000 * float(target["ymin"]) / height,
            1000 * float(target["xmax"]) / width,
            1000 * float(target["ymax"]) / height,
        ]
        candidates = prediction.get(int(target["timestamp"]), [])
        values.append(max((box_iou(target_box, box) for box in candidates), default=0.0))
    return values


JUDGE_RATING = re.compile(r"\s*:?\s*([0-3])\s*")


def parse_judge_rating(text: Any) -> int:
    match = JUDGE_RATING.fullmatch("" if text is None else str(text))
    return int(match.group(1)) if match else -1


def gemini_score(
    api_key: str,
    model: str,
    question: str,
    reference: str,
    candidate: str,
) -> tuple[int, str]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = (
        f"Question: {question}\nReference answer: {reference}\n"
        f"Candidate answer: {candidate}\nScore:"
    )
    if JUDGE_BACKEND == "openai":
        return openai_score(api_key, model, prompt)
    payload = {
        "system_instruction": {"parts": [{"text": JUDGE_INSTRUCTION}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 256,
            "thinkingConfig": {"thinkingLevel": JUDGE_THINKING_LEVEL},
        },
    }
    response = None
    for attempt in range(6):
        try:
            response = requests.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
        except requests.RequestException:
            time.sleep(2**attempt)
            continue
        if response.status_code == 200:
            body = response.json()
            candidates = body.get("candidates") or []
            parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
            text = "".join(part.get("text", "") for part in parts if not part.get("thought", False)).strip()
            return parse_judge_rating(text), text
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
        time.sleep(2**attempt)
    if response is not None:
        response.raise_for_status()
    raise RuntimeError(f"Gemini judge unreachable after six attempts: {url}")


def openai_score(api_key: str, model: str, prompt: str) -> tuple[int, str]:
    endpoint = JUDGE_BASE_URL.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": JUDGE_INSTRUCTION}, {"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 256,
    }
    response = None
    for attempt in range(6):
        try:
            response = requests.post(endpoint, headers=headers, json=payload, timeout=180)
        except requests.RequestException:
            time.sleep(2**attempt)
            continue
        if response.status_code == 200:
            text = (response.json()["choices"][0]["message"].get("content") or "").strip()
            return parse_judge_rating(text), text
        if response.status_code not in {429, 500, 502, 503, 504}:
            response.raise_for_status()
        time.sleep(2**attempt)
    if response is not None:
        response.raise_for_status()
    raise RuntimeError(f"Judge endpoint unreachable: {endpoint}")


def load_judge_cache(path: Path) -> dict[int, dict[str, Any]]:
    cache = {}
    if not path.is_file():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                cache[int(item["annotation_id"])] = item
    return cache


def judge_vqa(
    items: list[dict[str, Any]],
    api_key: str,
    model: str,
    workers: int,
    cache_path: Path,
) -> dict[int, int]:
    backend = JUDGE_BACKEND
    thinking = "default" if backend == "openai" else JUDGE_THINKING_LEVEL
    cache = load_judge_cache(cache_path)
    pending = [
        item
        for item in items
        if int(item["annotation_id"]) not in cache
        or cache[int(item["annotation_id"])].get("candidate") != item["candidate"]
        or cache[int(item["annotation_id"])].get("judge_model") != model
        or cache[int(item["annotation_id"])].get("judge_backend", "gemini") != backend
        or cache[int(item["annotation_id"])].get("judge_thinking", "default") != thinking
        or type(cache[int(item["annotation_id"])].get("score")) is not int
        or cache[int(item["annotation_id"])]["score"] not in range(4)
    ]

    def evaluate(item: dict[str, Any]) -> dict[str, Any]:
        score, response = -1, ""
        for _ in range(6):
            try:
                score, response = gemini_score(
                    api_key,
                    model,
                    item["question"],
                    item["reference"],
                    item["candidate"],
                )
            except RuntimeError as exc:
                score, response = -1, f"JUDGE_FAILED: {exc}"[:500]
                break
            if type(score) is int and score in range(4):
                break
        result = {**item, "score": score, "judge_response": response, "judge_model": model,
                  "judge_backend": backend, "judge_thinking": thinking}
        if type(score) is not int or score not in range(4):
            # Six attempts gave no 0-3 rating (e.g. blocked by a safety filter): count it as incorrect
            # (score -1, as the official eval.py does) and keep going; the next scoring pass retries it.
            result["score"] = -1
            result["judge_failed"] = True
        return result

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for result in tqdm(
            executor.map(evaluate, pending),
            total=len(pending),
            desc="Judging V-STaR VQA",
        ):
            cache[int(result["annotation_id"])] = result
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    failed = sum(1 for item in cache.values() if item.get("judge_failed"))
    if failed:
        print(f"{failed} V-STaR answers got no judge rating after six attempts and count as incorrect; "
              "rerun the scoring pass to retry them")
    return {annotation_id: int(item["score"]) for annotation_id, item in cache.items()}


def duration_group(row: dict[str, Any]) -> str:
    duration = float(row["frame_count"]) / float(row["fps"])
    if duration < 60:
        return "short"
    if duration < 180:
        return "medium"
    return "long"


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(samples)
    vqa_accuracy = mean(float(sample["vqa_correct"]) for sample in samples)
    temporal_1 = mean(sample["temporal_iou_1"] for sample in samples)
    temporal_2 = mean(sample["temporal_iou_2"] for sample in samples)
    spatial_1 = mean(sample["spatial_miou_1"] for sample in samples)
    spatial_2 = mean(sample["spatial_miou_2"] for sample in samples)

    def chain_metrics(temporal: float, spatial: float) -> tuple[float, float]:
        am = (vqa_accuracy + temporal + spatial) / 3
        lgm = -mean(math.log(max(1 - value, 1e-12)) for value in (vqa_accuracy, temporal, spatial))
        return am, lgm

    am1, lgm1 = chain_metrics(temporal_1, spatial_1)
    am2, lgm2 = chain_metrics(temporal_2, spatial_2)
    valid_vqa_scores = [sample["vqa_score"] for sample in samples if sample["vqa_score"] >= 0]
    metrics: dict[str, Any] = {
        "samples": count,
        "vqa_average_rating": mean(max(sample["vqa_score"], 0) for sample in samples),
        "vqa_average_valid_rating": mean(valid_vqa_scores) if valid_vqa_scores else 0.0,
        "vqa_accuracy": vqa_accuracy,
        "chain_1": {
            "temporal_mean_iou": temporal_1,
            "spatial_mean_iou": spatial_1,
            "am": am1,
            "lgm": lgm1,
        },
        "chain_2": {
            "temporal_mean_iou": temporal_2,
            "spatial_mean_iou": spatial_2,
            "am": am2,
            "lgm": lgm2,
        },
        "mAM": (am1 + am2) / 2,
        "mLGM": (lgm1 + lgm2) / 2,
    }
    for chain in (1, 2):
        temporal_values = [sample[f"temporal_iou_{chain}"] for sample in samples]
        spatial_values = [sample[f"spatial_ious_{chain}"] for sample in samples]
        metrics[f"chain_{chain}"].update(
            {
                f"temporal_r1_iou_{threshold}": mean(
                    float(value >= threshold) for value in temporal_values
                )
                for threshold in (0.3, 0.5, 0.7)
            }
        )
        metrics[f"chain_{chain}"].update(
            {
                f"spatial_map_{threshold}": mean(
                    mean(float(value >= threshold) for value in values) if values else 0.0
                    for values in spatial_values
                )
                for threshold in IOU_THRESHOLDS
            }
        )
        metrics[f"chain_{chain}"].update(
            {
                "vqa_and_temporal": mean(
                    float(sample["vqa_correct"] and sample[f"temporal_iou_{chain}"] >= 0.3)
                    for sample in samples
                ),
                "vqa_and_spatial": mean(
                    float(sample["vqa_correct"] and sample[f"spatial_miou_{chain}"] >= 0.1)
                    for sample in samples
                ),
                "temporal_and_spatial": mean(
                    float(
                        sample[f"temporal_iou_{chain}"] >= 0.3
                        and sample[f"spatial_miou_{chain}"] >= 0.1
                    )
                    for sample in samples
                ),
                "vqa_temporal_spatial": mean(
                    float(
                        sample["vqa_correct"]
                        and sample[f"temporal_iou_{chain}"] >= 0.3
                        and sample[f"spatial_miou_{chain}"] >= 0.1
                    )
                    for sample in samples
                ),
            }
        )
    return metrics


def group_predictions(
    rows: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> dict[int, dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
    missing = sorted({str(row["qid"]) for row in rows} - set(predictions))
    if missing:
        raise ValueError(f"Missing {len(missing)} V-STaR predictions, e.g. {missing[:5]}")
    grouped: dict[int, dict[str, tuple[dict[str, Any], dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(int(row["annotation_id"]), {})[str(row["task"])] = (
            row,
            predictions[str(row["qid"])],
        )
    for annotation_id, tasks in grouped.items():
        if set(tasks) != TASKS:
            raise ValueError(f"Annotation {annotation_id} has tasks {sorted(tasks)}")
    return grouped


def build_judge_items(grouped: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for annotation_id, tasks in grouped.items():
        row, prediction = tasks["vqa"]
        items.append(
            {
                "annotation_id": annotation_id,
                "question": str(row.get("source_question", row["question"])),
                "reference": str(row["answer"]),
                "candidate": boxed_content(prediction["final_answer"]),
            }
        )
    return items


def compute_metrics(
    grouped: dict[int, dict[str, Any]], scores: dict[int, int], judge_model: str
) -> dict[str, Any]:
    samples = []
    for annotation_id, tasks in grouped.items():
        base_row = tasks["vqa"][0]
        sample: dict[str, Any] = {
            "annotation_id": annotation_id,
            "domain": str(base_row["domain"]),
            "duration": duration_group(base_row),
            "vqa_score": scores[annotation_id],
            "vqa_correct": scores[annotation_id] >= 2,
        }
        target_temporal = json.loads(base_row["gt_timestamps"])
        for chain in (1, 2):
            temporal_prediction = tasks[f"temporal_{chain}"][1]["final_answer"]
            sample[f"temporal_iou_{chain}"] = temporal_iou(
                target_temporal, parse_temporal(temporal_prediction)
            )
            spatial_row, spatial_prediction = tasks[f"spatial_{chain}"]
            ious = spatial_ious(
                spatial_row, parse_spatial(spatial_prediction["final_answer"])
            )
            sample[f"spatial_ious_{chain}"] = ious
            sample[f"spatial_miou_{chain}"] = mean(ious) if ious else 0.0
        samples.append(sample)
    return {
        "judge_model": judge_model,
        "overall": summarize(samples),
        "by_duration": {
            duration: summarize([sample for sample in samples if sample["duration"] == duration])
            for duration in sorted({sample["duration"] for sample in samples})
        },
        "by_domain": {
            domain: summarize([sample for sample in samples if sample["domain"] == domain])
            for domain in sorted({sample["domain"] for sample in samples})
        },
    }


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("LLM_JUDGE_API_KEY", "")
    if JUDGE_BACKEND == "gemini" and not api_key:
        raise ValueError("Set LLM_JUDGE_API_KEY before scoring V-STaR with the Gemini judge")
    if args.judge_workers < 1:
        raise ValueError("--judge-workers must be at least 1")

    rows = pd.read_parquet(args.data).to_dict("records")
    grouped = group_predictions(rows, load_predictions(args.log_dir))
    scores = judge_vqa(
        build_judge_items(grouped),
        api_key,
        args.judge_model,
        args.judge_workers,
        args.log_dir / "vstar_video_judgments.jsonl",
    )
    metrics = compute_metrics(grouped, scores, args.judge_model)
    metrics["judge_failed"] = sum(scores[annotation_id] == -1 for annotation_id in grouped)
    output_path = args.log_dir / "vstar_video_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics["overall"], indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

