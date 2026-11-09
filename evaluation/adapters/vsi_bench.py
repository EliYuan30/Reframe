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


SOURCE_ROOT = raw_dataset_root("datasets--nyu-visionx--VSI-Bench")
METADATA_PATH = SOURCE_ROOT / "test.jsonl"
OUTPUT_ROOT = benchmark_output_root()
FRAME_DIR = OUTPUT_ROOT / "vsi_bench_frames"
OUTPUT_PATH = OUTPUT_ROOT / "vsi_bench.parquet"
NUM_FRAMES = 64
JPEG_QUALITY = 95
NUM_WORKERS = 64


def sample_video(video_path: Path, dataset: str, scene_name: str) -> list[str]:
    output_dir = FRAME_DIR / dataset / scene_name
    frame_paths = [
        output_dir / f"frame_n{NUM_FRAMES}_{index:04d}.jpg"
        for index in range(NUM_FRAMES)
    ]
    if all(path.is_file() for path in frame_paths):
        return [str(path) for path in frame_paths]

    output_dir.mkdir(parents=True, exist_ok=True)
    reader = decord.VideoReader(str(video_path), num_threads=1)
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
    prompt = (
        "These are the frames of an egocentric video.\n"
        f"Question: {str(question).strip()}"
    )
    if isinstance(options, (list, tuple)) and options:
        labels = [str(option).split(".", 1)[0].strip() for option in options]
        prompt += (
            "\nOptions:\n"
            + "\n".join(str(option).strip() for option in options)
            + "\nReason about the video frames, then select the single best "
            f"option from {labels[0]} through {labels[-1]} and put the final "
            "answer letter in \\boxed{}, for example \\boxed{A}."
        )
    else:
        prompt += (
            "\nReason about the video frames, then give the concise numerical "
            "answer in \\boxed{}."
        )
    return prompt


def main() -> None:
    if not METADATA_PATH.is_file():
        raise FileNotFoundError(METADATA_PATH)
    metadata = pd.read_json(METADATA_PATH, lines=True)

    videos = sorted(
        {
            (str(row.dataset), str(row.scene_name))
            for row in metadata.itertuples(index=False)
        }
    )
    frames_by_video: dict[tuple[str, str], list[str]] = {}

    def prepare_video(key: tuple[str, str]) -> tuple[tuple[str, str], list[str]]:
        dataset, scene_name = key
        video_path = SOURCE_ROOT / dataset / f"{scene_name}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        return key, sample_video(video_path, dataset, scene_name)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for key, frame_paths in tqdm(
            executor.map(prepare_video, videos),
            total=len(videos),
            desc="Sampling VSI-Bench",
        ):
            frames_by_video[key] = frame_paths

    rows = []
    for row in metadata.itertuples(index=False):
        options = row.options
        answer = str(row.ground_truth).strip()
        if isinstance(options, (list, tuple)) and options:
            labels = {str(option).split(".", 1)[0].strip() for option in options}
            if answer not in labels:
                raise ValueError(f"Invalid VSI-Bench answer {answer!r} for id {row.id}")

        key = (str(row.dataset), str(row.scene_name))
        rows.append(
            {
                "qid": f"vsi_bench-{row.id}",
                "question": format_question(row.question, options),
                "image": frames_by_video[key],
                "answer": f"\\boxed{{{answer}}}",
                "is_video": True,
                "category": str(row.question_type),
                "answer_type": "multiple_choice" if options else "numerical",
                "source_dataset": str(row.dataset),
                "scene_name": str(row.scene_name),
            }
        )

    output = pd.DataFrame(rows)
    if output["qid"].duplicated().any():
        raise ValueError("VSI-Bench id values are not unique")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(output)} questions to {OUTPUT_PATH}")
    print(f"Wrote {len(videos) * NUM_FRAMES} frames under {FRAME_DIR}")


if __name__ == "__main__":
    main()
