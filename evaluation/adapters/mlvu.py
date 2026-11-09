#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
from string import ascii_uppercase

import pandas as pd
from tqdm import tqdm

try:
    from .common import benchmark_output_root, eval_raw_root, raw_dataset_root, sample_video
except ImportError:
    from evaluation.adapters.common import benchmark_output_root, eval_raw_root, raw_dataset_root, sample_video


SOURCE_ROOT = raw_dataset_root("datasets--MLVU--MVLU") / "MLVU"
METADATA_DIR = SOURCE_ROOT / "json"
VIDEO_DIR = SOURCE_ROOT / "video"
MLVU_ROOT_ENV = "REFRAME_MLVU_ROOT"
LOCAL_DEFAULT_NAME = "MLVU_dev"
OUTPUT_ROOT = benchmark_output_root()
FRAME_DIR = OUTPUT_ROOT / "mlvu_frames"
OUTPUT_PATH = OUTPUT_ROOT / "mlvu.parquet"
NUM_FRAMES = 64
NUM_WORKERS = int(os.environ.get("REFRAME_PREPARE_WORKERS", "64"))


def format_question(item: dict) -> tuple[str, str]:
    question = "These are uniformly sampled frames of a video.\nQuestion: " + str(item["question"]).strip()
    candidates = item.get("candidates")
    if candidates is not None and len(candidates):
        choices = [str(value).strip() for value in candidates]
        labels = ascii_uppercase[: len(choices)]
        answer_text = str(item["answer"]).strip()
        if answer_text not in choices:
            raise ValueError(f"MLVU answer {answer_text!r} is not in candidates")
        prompt = (
            question
            + "\nOptions:\n"
            + "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices))
            + "\nSelect the single best option and put its letter in \\boxed{}."
        )
        return prompt, labels[choices.index(answer_text)]
    return question + "\nPut the final answer in \\boxed{}.", str(item["answer"]).strip()


def has_local_layout(root: Path) -> bool:
    return (root / "videos").is_dir() and any((root / "mlvu").glob("test-*.parquet"))


def local_layout_root() -> Path | None:
    value = os.environ.get(MLVU_ROOT_ENV, "").strip()
    if value:
        root = Path(value)
        if not has_local_layout(root):
            raise FileNotFoundError(
                f"{MLVU_ROOT_ENV}={root} must contain mlvu/test-*.parquet and videos/"
            )
        return root
    if any(METADATA_DIR.glob("*.json")):
        return None
    candidate = eval_raw_root() / LOCAL_DEFAULT_NAME
    return candidate if has_local_layout(candidate) else None


def question_stem(question: object) -> str:
    return re.split(r"\n\s*\(A\)", str(question), maxsplit=1)[0].strip()


def local_records(metadata: pd.DataFrame, video_dir: Path) -> list[tuple[str, str, dict, Path]]:
    records = []
    for row in metadata.to_dict("records"):
        candidates = [str(value).strip() for value in row["candidates"]]
        letter = str(row["answer"]).strip().upper()
        index = ascii_uppercase.find(letter) if len(letter) == 1 else -1
        if not 0 <= index < len(candidates):
            raise ValueError(f"MLVU {row['question_id']} answer {row['answer']!r} is outside its candidates")
        task = str(row["task_type"])
        item = {
            "question": question_stem(row["question"]),
            "candidates": candidates,
            "answer": candidates[index],
            "question_type": task,
            "duration": float(row["duration"]),
        }
        records.append((task, str(row["question_id"]), item, video_dir / str(row["video_name"])))
    return records


def json_records() -> list[tuple[str, str, dict, Path]]:
    records = []
    for metadata_path in sorted(METADATA_DIR.glob("*.json")):
        task = metadata_path.stem
        for index, item in enumerate(json.loads(metadata_path.read_text(encoding="utf-8"))):
            records.append((task, str(index), item, VIDEO_DIR / task / str(item["video"])))
    if not records:
        raise FileNotFoundError(
            f"No MLVU annotations under {METADATA_DIR}; set {MLVU_ROOT_ENV} to an "
            "lmms-eval MLVU root (mlvu/test-*.parquet + videos/) instead"
        )
    return records


def main() -> None:
    local_root = local_layout_root()
    if local_root is None:
        records = json_records()
    else:
        paths = sorted((local_root / "mlvu").glob("test-*.parquet"))
        print(f"Using lmms-eval MLVU layout under {local_root}", flush=True)
        metadata = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        records = local_records(metadata, local_root / "videos")
    missing = [path for _, _, _, path in records if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MLVU download is incomplete: {len(missing)} videos missing; first={missing[0]}")

    unique_videos = sorted({(task, path) for task, _, _, path in records})
    frames_by_video = {}

    def prepare_video(entry: tuple[str, Path]) -> tuple[Path, list[str]]:
        task, path = entry
        frames, _ = sample_video(path, FRAME_DIR / task / path.stem, NUM_FRAMES)
        return path, frames

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = executor.map(prepare_video, unique_videos)
        for path, frames in tqdm(results, total=len(unique_videos), desc="Sampling MLVU"):
            frames_by_video[path] = frames

    rows = []
    for task, index, item, video_path in records:
        question, answer = format_question(item)
        rows.append({
            "qid": f"mlvu-{task}-{index}",
            "question": question,
            "image": frames_by_video[video_path],
            "answer": answer,
            "is_video": True,
            "category": str(item["question_type"]),
            "duration": float(item["duration"]),
        })
    output = pd.DataFrame(rows)
    if output["qid"].duplicated().any():
        raise ValueError("MLVU qid values are not unique")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(output)} questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
