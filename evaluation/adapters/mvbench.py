#!/usr/bin/env python3
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from string import ascii_uppercase
import json
import shutil
import tempfile
import zipfile

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


try:
    from .common import benchmark_output_root, raw_dataset_root
except ImportError:
    from evaluation.adapters.common import benchmark_output_root, raw_dataset_root


SOURCE_ROOT = raw_dataset_root("datasets--OpenGVLab--MVBench")
METADATA_DIR = SOURCE_ROOT / "json"
VIDEO_DIR = SOURCE_ROOT / "video"
OUTPUT_ROOT = benchmark_output_root()
FRAME_DIR = OUTPUT_ROOT / "mvbench_frames"
OUTPUT_PATH = OUTPUT_ROOT / "mvbench.parquet"
NUM_FRAMES = 64
JPEG_QUALITY = 95
PATCH_ARCHIVE = "data0613.zip"
PATCH_PREFIX = "data0613"

TASK_SOURCES = {
    "action_sequence": ("star.zip", "star/Charades_v1_480", "video"),
    "action_prediction": ("star.zip", "star/Charades_v1_480", "video"),
    "action_antonym": ("ssv2_video.zip", "ssv2_video", "video"),
    "fine_grained_action": (
        "Moments_in_Time_Raw.zip",
        "Moments_in_Time_Raw/videos",
        "video",
    ),
    "unexpected_action": ("FunQA_test.zip", "FunQA_test/test", "video"),
    "object_existence": ("clevrer.zip", "clevrer/video_validation", "video"),
    "object_interaction": ("star.zip", "star/Charades_v1_480", "video"),
    "object_shuffle": ("perception.zip", "perception/videos", "video"),
    "moving_direction": ("clevrer.zip", "clevrer/video_validation", "video"),
    "action_localization": ("sta.zip", "sta/sta_video", "video"),
    "scene_transition": ("scene_qa.zip", "scene_qa/video", "video"),
    "action_count": ("perception.zip", "perception/videos", "video"),
    "moving_count": ("clevrer.zip", "clevrer/video_validation", "video"),
    "moving_attribute": ("clevrer.zip", "clevrer/video_validation", "video"),
    "state_change": ("perception.zip", "perception/videos", "video"),
    "character_order": ("perception.zip", "perception/videos", "video"),
    "egocentric_navigation": ("vlnqa.zip", "vlnqa", "video"),
    "episodic_reasoning": ("tvqa.zip", "tvqa/frames_fps3_hq", "frames"),
    "counterfactual_inference": (
        "clevrer.zip",
        "clevrer/video_validation",
        "video",
    ),
}


def frame_paths(output_dir: Path) -> list[Path]:
    return [output_dir / f"{index:02d}.jpg" for index in range(NUM_FRAMES)]


def sampled_indices(first: int, last: int) -> np.ndarray:
    if last < first:
        raise ValueError(f"Invalid frame range: {first}..{last}")
    return np.rint(np.linspace(first, last, NUM_FRAMES)).astype(np.int64)


def decode_frame_sequentially(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    frame = None
    for _ in range(frame_index + 1):
        ok, decoded = capture.read()
        if not ok:
            break
        frame = decoded
    capture.release()
    if frame is None:
        raise RuntimeError(f"Cannot decode any frames from {video_path}")
    return frame


def sample_video(
    video_path: Path,
    output_dir: Path,
    start: float | None,
    end: float | None,
) -> list[str]:
    outputs = frame_paths(output_dir)
    if all(path.is_file() for path in outputs):
        return [str(path) for path in outputs]

    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not capture.isOpened() or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Cannot read video: {video_path}")

    first = 0
    last = frame_count - 1
    if start is not None or end is not None:
        if fps <= 0:
            capture.release()
            raise RuntimeError(f"Cannot determine FPS: {video_path}")
        if start is not None:
            first = max(0, int(round(start * fps)))
        if end is not None:
            last = min(last, int(round(end * fps)))

    output_dir.mkdir(parents=True, exist_ok=True)
    for output_path, frame_index in zip(outputs, sampled_indices(first, last)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok:
            frame = decode_frame_sequentially(video_path, int(frame_index))
        if not cv2.imwrite(
            str(output_path),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
        ):
            capture.release()
            raise RuntimeError(f"Cannot write frame: {output_path}")

    capture.release()
    return [str(path) for path in outputs]


def sample_archived_video(
    archive: zipfile.ZipFile,
    member: str,
    output_dir: Path,
    start: float | None,
    end: float | None,
) -> list[str]:
    outputs = frame_paths(output_dir)
    if all(path.is_file() for path in outputs):
        return [str(path) for path in outputs]

    with tempfile.TemporaryDirectory() as temporary_dir:
        video_path = Path(temporary_dir) / Path(member).name
        with archive.open(member) as source, video_path.open("wb") as target:
            shutil.copyfileobj(source, target)
        return sample_video(video_path, output_dir, start, end)


def sample_archived_frames(
    archive: zipfile.ZipFile,
    members: list[str],
    output_dir: Path,
    start: float | None,
    end: float | None,
    fps: float | None,
) -> list[str]:
    outputs = frame_paths(output_dir)
    if all(path.is_file() for path in outputs):
        return [str(path) for path in outputs]
    if not members:
        raise FileNotFoundError(f"No source frames for {output_dir}")

    first = 0
    last = len(members) - 1
    if start is not None or end is not None:
        if fps is None or fps <= 0:
            raise ValueError(f"Missing FPS for bounded frame sequence: {output_dir}")
        if start is not None:
            first = max(0, int(round(start * fps)))
        if end is not None:
            last = min(last, int(round(end * fps)))

    output_dir.mkdir(parents=True, exist_ok=True)
    for output_path, frame_index in zip(outputs, sampled_indices(first, last)):
        output_path.write_bytes(archive.read(members[int(frame_index)]))
    return [str(path) for path in outputs]


def format_question(question: str, candidates: list[str]) -> str:
    labels = ascii_uppercase[:len(candidates)]
    options = "\n".join(
        f"{label}. {candidate}" for label, candidate in zip(labels, candidates)
    )
    return (
        "These are the frames of a video.\n"
        f"Question: {question.strip()}\n"
        f"Options:\n{options}\n"
        "Reason about the video frames, then select the single best option from "
        f"{labels[0]} through {labels[-1]} and put the final answer letter in "
        "\\boxed{}, for example \\boxed{A}."
    )


def load_metadata() -> dict[str, list[dict]]:
    metadata = {}
    for task in TASK_SOURCES:
        path = METADATA_DIR / f"{task}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            items = json.load(handle)
        for index, item in enumerate(items):
            candidates = item.get("candidates") or []
            if item.get("answer") not in candidates:
                raise ValueError(f"Invalid answer in {path} row {index}")
            if len(candidates) > len(ascii_uppercase):
                raise ValueError(f"Too many candidates in {path} row {index}")
        metadata[task] = items
    return metadata


def archive_member(prefix: str, media: str) -> str:
    return f"{prefix.rstrip('/')}/{media.lstrip('/')}"


def resolve_archived_media(
    archives: dict[str, zipfile.ZipFile],
    archive_name: str,
    member: str,
) -> tuple[zipfile.ZipFile, str]:
    archive = archives[archive_name]
    if member in archive.NameToInfo:
        return archive, member
    patch_member = f"{PATCH_PREFIX}/{member}"
    patch_archive = archives[PATCH_ARCHIVE]
    if patch_member in patch_archive.NameToInfo:
        return patch_archive, patch_member
    raise KeyError(member)


def frame_members_by_directory(archive: zipfile.ZipFile) -> dict[str, list[str]]:
    directories: dict[str, list[str]] = {}
    for member in archive.namelist():
        if member.lower().endswith((".jpg", ".jpeg", ".png")):
            directory = member.rsplit("/", 1)[0]
            directories.setdefault(directory, []).append(member)
    for members in directories.values():
        members.sort()
    return directories


def validate_sources(
    metadata: dict[str, list[dict]],
    archives: dict[str, zipfile.ZipFile],
    frame_directories: dict[str, dict[str, list[str]]],
) -> None:
    missing = []
    for task, items in metadata.items():
        archive_name, prefix, media_type = TASK_SOURCES[task]
        for item in items:
            media = str(item["video"])
            if media_type == "frames":
                directory = archive_member(prefix, media)
                if directory not in frame_directories[archive_name]:
                    missing.append(f"{archive_name}:{directory}/")
            else:
                member = archive_member(prefix, media)
                try:
                    resolve_archived_media(archives, archive_name, member)
                except KeyError:
                    missing.append(f"{archive_name}:{member}")

    if missing:
        examples = "\n".join(f"  {path}" for path in missing[:10])
        suffix = "\n  ..." if len(missing) > 10 else ""
        raise FileNotFoundError(
            f"Missing {len(missing)} MVBench media files:\n{examples}{suffix}"
        )


def optional_float(item: dict, key: str) -> float | None:
    value = item.get(key)
    return None if value is None else float(value)


def main() -> None:
    metadata = load_metadata()
    archive_names = sorted(
        {source[0] for source in TASK_SOURCES.values() if source[0] is not None}
        | {PATCH_ARCHIVE}
    )

    with ExitStack() as stack:
        archives = {}
        for archive_name in archive_names:
            archive_path = VIDEO_DIR / archive_name
            if not archive_path.is_file():
                raise FileNotFoundError(archive_path)
            archives[archive_name] = stack.enter_context(zipfile.ZipFile(archive_path))

        frame_directories = {
            archive_name: frame_members_by_directory(archive)
            for archive_name, archive in archives.items()
            if any(
                source[0] == archive_name and source[2] == "frames"
                for source in TASK_SOURCES.values()
            )
        }
        validate_sources(metadata, archives, frame_directories)

        rows = []
        total = sum(len(items) for items in metadata.values())
        with tqdm(total=total, desc="Sampling MVBench") as progress:
            for task, items in metadata.items():
                archive_name, prefix, media_type = TASK_SOURCES[task]
                for index, item in enumerate(items):
                    qid = f"mvbench-{task}-{index:04d}"
                    output_dir = FRAME_DIR / task / f"{index:04d}"
                    media = str(item["video"])
                    start = optional_float(item, "start")
                    end = optional_float(item, "end")

                    if media_type == "frames":
                        directory = archive_member(prefix, media)
                        images = sample_archived_frames(
                            archives[archive_name],
                            frame_directories[archive_name][directory],
                            output_dir,
                            start,
                            end,
                            optional_float(item, "fps"),
                        )
                    else:
                        source_archive, source_member = resolve_archived_media(
                            archives,
                            archive_name,
                            archive_member(prefix, media),
                        )
                        images = sample_archived_video(
                            source_archive,
                            source_member,
                            output_dir,
                            start,
                            end,
                        )

                    candidates = [str(value) for value in item["candidates"]]
                    answer_index = candidates.index(str(item["answer"]))
                    rows.append(
                        {
                            "qid": qid,
                            "question": format_question(str(item["question"]), candidates),
                            "image": images,
                            "answer": f"\\boxed{{{ascii_uppercase[answer_index]}}}",
                            "is_video": True,
                            "category": task,
                        }
                    )
                    progress.update()

    output = pd.DataFrame(rows)
    if output["qid"].duplicated().any():
        raise ValueError("MVBench qid values are not unique")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(output)} questions to {OUTPUT_PATH}")
    print(f"Wrote {len(output) * NUM_FRAMES} frames under {FRAME_DIR}")


if __name__ == "__main__":
    main()
