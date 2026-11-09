#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import decord
import pandas as pd
from PIL import Image
from tqdm import tqdm


try:
    from .common import benchmark_output_root, raw_dataset_root
except ImportError:
    from evaluation.adapters.common import benchmark_output_root, raw_dataset_root


SOURCE_ROOT = raw_dataset_root("datasets--lmms-lab--Video-MME")
METADATA_PATH = SOURCE_ROOT / "videomme" / "test-00000-of-00001.parquet"
VIDEO_DIR = SOURCE_ROOT / "data"
OUTPUT_ROOT = benchmark_output_root()
FRAME_DIR = OUTPUT_ROOT / "videomme_frames"
OUTPUT_PATH = OUTPUT_ROOT / "videomme.parquet"
NUM_FRAMES = 64
JPEG_QUALITY = 95
NUM_WORKERS = 64


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


def format_question(question: str, options) -> str:
    lines = [str(option).strip() for option in options]
    return (
        "These are the frames of a video.\n"
        f"Question: {str(question).strip()}\n"
        + "\n".join(lines)
        + "\nReason about the video frames, then select the single best option "
        "from A through D and put the final answer letter in \\boxed{}, "
        "for example \\boxed{A}."
    )


def main() -> None:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(METADATA_PATH)
    metadata = pd.read_parquet(METADATA_PATH)

    video_ids = sorted(metadata["videoID"].astype(str).unique())
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
            desc="Sampling Video-MME",
        ):
            frames_by_video[video_id] = frame_paths

    rows = []
    for row in metadata.to_dict("records"):
        answer = str(row["answer"]).strip().upper()
        if answer not in {"A", "B", "C", "D"}:
            raise ValueError(f"Unexpected Video-MME answer: {row['answer']!r}")
        rows.append(
            {
                "qid": f"videomme-{row['question_id']}",
                "question": format_question(row["question"], row["options"]),
                "image": frames_by_video[str(row["videoID"])],
                "answer": f"\\boxed{{{answer}}}",
                "is_video": True,
                "category": str(row["task_type"]),
                "duration": str(row["duration"]),
                "domain": str(row["domain"]),
                "sub_category": str(row["sub_category"]),
            }
        )

    output = pd.DataFrame(rows)
    if output["qid"].duplicated().any():
        raise ValueError("Video-MME question_id values are not unique")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(output)} questions to {OUTPUT_PATH}")
    print(f"Wrote {len(video_ids) * NUM_FRAMES} frames under {FRAME_DIR}")


if __name__ == "__main__":
    main()
