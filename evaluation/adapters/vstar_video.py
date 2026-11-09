#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import warnings

import cv2
import decord
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


try:
    from .common import benchmark_output_root, raw_dataset_root
except ImportError:
    from evaluation.adapters.common import benchmark_output_root, raw_dataset_root


SOURCE_ROOT = raw_dataset_root("datasets--V-STaR-Bench--V-STaR")
ANNOTATION_PATH = SOURCE_ROOT / "V_STaR_test.json"
VIDEO_DIR = SOURCE_ROOT / "videos"
OUTPUT_ROOT = benchmark_output_root()
FRAME_DIR = OUTPUT_ROOT / "vstar_video_frames"
OUTPUT_PATH = OUTPUT_ROOT / "vstar_video.parquet"
NUM_FRAMES = 64
NUM_WORKERS = 32
JPEG_QUALITY = 95


def sample_video_opencv(video_path: Path, outputs: list[Path]) -> list[str]:
    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened() or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Cannot read video with OpenCV: {video_path}")

    indices = np.rint(np.linspace(0, frame_count - 1, NUM_FRAMES)).astype(int)
    cursor = 0
    frame_index = 0
    last_frame = None
    while cursor < NUM_FRAMES:
        ok, frame = capture.read()
        if not ok:
            break
        last_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        while cursor < NUM_FRAMES and indices[cursor] <= frame_index:
            Image.fromarray(last_frame).save(outputs[cursor], quality=JPEG_QUALITY)
            cursor += 1
        frame_index += 1
    capture.release()

    if last_frame is None:
        raise RuntimeError(f"Cannot decode any frames with OpenCV: {video_path}")
    if cursor < NUM_FRAMES:
        warnings.warn(
            f"{video_path} ended after {frame_index} decodable frames; "
            f"reusing the last frame for {NUM_FRAMES - cursor} sampled positions."
        )
        while cursor < NUM_FRAMES:
            Image.fromarray(last_frame).save(outputs[cursor], quality=JPEG_QUALITY)
            cursor += 1
    return [str(path) for path in outputs]


def sample_video(video_path: Path) -> list[str]:
    output_dir = FRAME_DIR / video_path.stem
    outputs = [output_dir / f"{index:02d}.jpg" for index in range(NUM_FRAMES)]
    if all(path.is_file() for path in outputs):
        return [str(path) for path in outputs]

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        reader = decord.VideoReader(str(video_path))
        if len(reader) <= 0:
            return sample_video_opencv(video_path, outputs)
        indices = np.rint(np.linspace(0, len(reader) - 1, NUM_FRAMES)).astype(int)
        frames = reader.get_batch(indices).asnumpy()
    except decord.DECORDError as error:
        warnings.warn(f"Decord failed for {video_path}; using OpenCV fallback: {error}")
        return sample_video_opencv(video_path, outputs)

    for output_path, frame in zip(outputs, frames):
        Image.fromarray(frame).save(output_path, quality=JPEG_QUALITY)
    return [str(path) for path in outputs]


def normalized_box(box: dict, width: int, height: int) -> list[int]:
    return [
        round(1000 * float(box["xmin"]) / width),
        round(1000 * float(box["ymin"]) / height),
        round(1000 * float(box["xmax"]) / width),
        round(1000 * float(box["ymax"]) / height),
    ]


def spatial_target(item: dict) -> list[list[int]]:
    width, height = int(item["width"]), int(item["height"])
    return [
        [int(box["timestamp"]), *normalized_box(box, width, height)]
        for box in item["bboxes"]
    ]


def prompts(item: dict) -> dict[str, str]:
    duration = round(float(item["frame_count"]) / float(item["fps"]), 2)
    start = math.ceil(float(item["timestamps"][0]))
    end = math.floor(float(item["timestamps"][1]))
    seconds = list(range(start, end + 1))
    boxes = [entry[1:] for entry in spatial_target(item)]
    output_format = (
        "Use coordinates normalized to integers from 0 to 1000. Output only "
        "\\boxed{[[second, x1, y1, x2, y2], ...]} as the final answer."
    )
    return {
        "vqa": (
            f"Answer the question about the video: {item['question']}\n"
            "If the answer is a person, you do not need to identify the person. "
            "Give a concise answer in \\boxed{}."
        ),
        "temporal_1": (
            f"This video is {duration} seconds long. Answer the temporal grounding "
            f"question: {item['temporal_question']}\nOutput only the start and end "
            "timestamps in seconds as \\boxed{[start, end]}."
        ),
        "spatial_1": (
            f"Answer the spatial grounding question: {item['spatial_question']}\n"
            f"For every whole second in {seconds}, localize the relevant object. "
            f"{output_format}"
        ),
        "temporal_2": (
            f"This video is {duration} seconds long. Answer the temporal grounding "
            f"question: {item['temporal_question']}\nThe key object has the following "
            f"{len(boxes)} locations in xyxy_0_1000 coordinates, but their timestamps "
            f"are unknown: {json.dumps(boxes, separators=(',', ':'))}. Output only "
            "the start and end timestamps in seconds as \\boxed{[start, end]}."
        ),
        "spatial_2": (
            f"Answer the spatial grounding question: {item['spatial_question_2']}\n"
            f"Determine the relevant whole seconds and output exactly {len(boxes)} "
            f"locations. {output_format}"
        ),
    }


def task_answer(item: dict, task: str) -> str:
    if task == "vqa":
        return str(item["answer"]).strip()
    if task.startswith("temporal"):
        return json.dumps(item["timestamps"], separators=(",", ":"))
    return json.dumps(spatial_target(item), separators=(",", ":"))


def main() -> None:
    if not ANNOTATION_PATH.is_file():
        raise FileNotFoundError(ANNOTATION_PATH)
    with ANNOTATION_PATH.open(encoding="utf-8") as handle:
        annotations = json.load(handle)

    video_paths = {path.stem: path for path in VIDEO_DIR.rglob("*.mp4")}
    required_videos = sorted({str(item["vid"]) for item in annotations})
    missing = sorted(set(required_videos) - set(video_paths))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} V-STaR videos: {missing[:10]}")

    frames_by_video: dict[str, list[str]] = {}

    def prepare_video(video_id: str) -> tuple[str, list[str]]:
        return video_id, sample_video(video_paths[video_id])

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for video_id, frames in tqdm(
            executor.map(prepare_video, required_videos),
            total=len(required_videos),
            desc="Sampling V-STaR",
        ):
            frames_by_video[video_id] = frames

    rows = []
    for index, item in enumerate(annotations):
        video_id = str(item["vid"])
        gt_bboxes = json.dumps(item["bboxes"], separators=(",", ":"))
        gt_timestamps = json.dumps(item["timestamps"], separators=(",", ":"))
        for task, question in prompts(item).items():
            rows.append(
                {
                    "qid": f"vstar-video-{index}-{task}",
                    "annotation_id": index,
                    "task": task,
                    "question": question,
                    "source_question": str(item["question"]),
                    "image": frames_by_video[video_id],
                    "answer": task_answer(item, task),
                    "is_video": True,
                    "video_id": video_id,
                    "domain": str(item["domain"]),
                    "fps": float(item["fps"]),
                    "width": int(item["width"]),
                    "height": int(item["height"]),
                    "frame_count": int(item["frame_count"]),
                    "gt_timestamps": gt_timestamps,
                    "gt_bboxes": gt_bboxes if task.startswith("spatial") else "[]",
                }
            )

    output = pd.DataFrame(rows)
    if output["qid"].duplicated().any():
        raise ValueError("V-STaR qid values are not unique")
    expected_rows = 5 * len(annotations)
    if len(output) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(output)}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(output)} requests for {len(annotations)} samples to {OUTPUT_PATH}")
    print(f"Wrote frames under {FRAME_DIR}")


if __name__ == "__main__":
    main()

