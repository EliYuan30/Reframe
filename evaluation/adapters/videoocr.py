#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import ascii_uppercase

import decord
import pandas as pd
from PIL import Image
from tqdm import tqdm


try:
    from .common import benchmark_output_root, raw_dataset_root
except ImportError:
    from evaluation.adapters.common import benchmark_output_root, raw_dataset_root


SOURCE_ROOT = raw_dataset_root("datasets--DogNeverSleep--MME-VideoOCR_Dataset")
METADATA_PATH = SOURCE_ROOT / "data" / "train-00000-of-00001.parquet"
VIDEO_DIR = SOURCE_ROOT / "Video"
OUTPUT_ROOT = benchmark_output_root()
FRAME_DIR = OUTPUT_ROOT / "videoocr_frames"
OUTPUT_PATH = OUTPUT_ROOT / "videoocr.parquet"
NUM_FRAMES = 64
JPEG_QUALITY = 95
NUM_WORKERS = 64
KEPT_EVAL_METHODS = ("multiple_choice", "containment_match")


def sample_video(video_path: Path) -> list[str]:
    output_dir = FRAME_DIR / video_path.stem
    frame_paths = [
        output_dir / f"frame_n{NUM_FRAMES}_{index:04d}.jpg"
        for index in range(NUM_FRAMES)
    ]
    if all(path.is_file() for path in frame_paths):
        return [str(path) for path in frame_paths]

    output_dir.mkdir(parents=True, exist_ok=True)
    reader = decord.VideoReader(str(video_path))
    frame_count = len(reader)
    if frame_count <= 0:
        raise RuntimeError(f"Cannot read video: {video_path}")

    step_size = frame_count / (NUM_FRAMES + 1)
    frame_indices = [int(index * step_size) for index in range(1, NUM_FRAMES + 1)]
    frames = reader.get_batch(frame_indices).asnumpy()
    for output_path, frame in zip(frame_paths, frames):
        Image.fromarray(frame).save(output_path, quality=JPEG_QUALITY)
    return [str(path) for path in frame_paths]


def format_multiple_choice(question: str, options: list[str]) -> str:
    labels = list(ascii_uppercase[: len(options)])
    lines = [f"{label}. {text.strip()}" for label, text in zip(labels, options)]
    return (
        "These are the frames of a video.\n"
        f"Question: {str(question).strip()}\n"
        "Options:\n" + "\n".join(lines) + "\n"
        "Reason about the video frames, then select the single best option from "
        f"{labels[0]} through {labels[-1]} and put the final answer letter in "
        "\\boxed{}, for example \\boxed{A}."
    )


def format_containment(question: str) -> str:
    return (
        "These are the frames of a video.\n"
        f"Question: {str(question).strip()}\n"
        "Reason about the video frames, then put only the exact answer text in "
        "\\boxed{}. If the answer contains several items, separate them with "
        "semicolons, for example \\boxed{FIRST;SECOND}."
    )


def main() -> None:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(METADATA_PATH)
    metadata = pd.read_parquet(METADATA_PATH)
    metadata = metadata[metadata["eval_method"].isin(KEPT_EVAL_METHODS)]
    if metadata.empty:
        raise ValueError("No MME-VideoOCR questions left after filtering")

    video_ids = sorted({str(value) for value in metadata["video_index"]})
    frames_by_video: dict[str, list[str]] = {}

    def prepare_video(video_id: str) -> tuple[str, list[str]]:
        video_path = VIDEO_DIR / f"{video_id}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        return video_id, sample_video(video_path)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for video_id, frame_paths in tqdm(
            executor.map(prepare_video, video_ids),
            total=len(video_ids),
            desc="Sampling MME-VideoOCR",
        ):
            frames_by_video[video_id] = frame_paths

    rows = []
    for row in metadata.to_dict("records"):
        video_id = str(row["video_index"])
        eval_method = str(row["eval_method"])
        options = [str(value) for value in (row["option"] if row["option"] is not None else [])]

        if eval_method == "multiple_choice":
            if len(options) < 2:
                raise ValueError(f"VideoOCR index {row['index']} has fewer than 2 options")
            labels = list(ascii_uppercase[: len(options)])
            answer = str(row["answer"]).strip().upper()
            if answer not in labels:
                raise ValueError(
                    f"VideoOCR index {row['index']} answer {answer!r} is not a valid option"
                )
            question = format_multiple_choice(row["question"], options)
        else:
            answer = str(row["answer"]).strip()
            if not answer:
                raise ValueError(f"VideoOCR index {row['index']} has an empty answer")
            question = format_containment(row["question"])

        rows.append(
            {
                "qid": f"videoocr-{row['index']}",
                "question": question,
                "image": frames_by_video[video_id],
                "answer": f"\\boxed{{{answer}}}",
                "is_video": True,
                "category": str(row["task_type"]),
                "task": str(row["task"]),
                "eval_method": eval_method,
            }
        )

    output = pd.DataFrame(rows)
    if output["qid"].duplicated().any():
        raise ValueError("MME-VideoOCR qid values are not unique")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(output)} questions to {OUTPUT_PATH}")
    print(output["eval_method"].value_counts().to_string())
    print(f"Wrote {len(video_ids) * NUM_FRAMES} frames under {FRAME_DIR}")


if __name__ == "__main__":
    main()
