#!/usr/bin/env python3
from __future__ import annotations

import io
import os
from pathlib import Path


JPEG_QUALITY = 95

RAW_ROOT_ENV = "REFRAME_EVAL_RAW_ROOT"
OUTPUT_ROOT_ENV = "REFRAME_EVAL_OUTPUT_ROOT"
DEFAULT_RAW_ROOT = Path("/path/to/your/data")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "outputs/benchmarks"


def eval_raw_root() -> Path:
    value = os.environ.get(RAW_ROOT_ENV, "").strip()
    return Path(value) if value else DEFAULT_RAW_ROOT


def raw_dataset_root(name: str) -> Path:
    return eval_raw_root() / name


def benchmark_output_root() -> Path:
    value = os.environ.get(OUTPUT_ROOT_ENV, "").strip()
    return Path(value) if value else DEFAULT_OUTPUT_ROOT


def export_image(payload, output_path: Path) -> str:
    if output_path.is_file():
        return str(output_path)
    from PIL import Image

    data = payload.get("bytes") if isinstance(payload, dict) else payload
    if not isinstance(data, bytes) or not data:
        raise ValueError(f"Missing image bytes for {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").save(output_path, quality=JPEG_QUALITY)
    return str(output_path)


def sample_video(
    video_path: Path,
    output_dir: Path,
    num_frames: int = 64,
    official_lvb_duration: float | None = None,
) -> tuple[list[str], list[float]]:
    import decord
    import numpy as np
    from PIL import Image

    try:
        reader = decord.VideoReader(str(video_path), num_threads=1)
    except decord.DECORDError:
        if official_lvb_duration is not None:
            raise
        return _sample_video_opencv(video_path, output_dir, num_frames)
    frame_count = len(reader)
    if frame_count <= 0:
        raise RuntimeError(f"Cannot read video: {video_path}")
    fps = float(reader.get_avg_fps())
    if fps <= 0:
        raise RuntimeError(f"Cannot determine FPS: {video_path}")

    if official_lvb_duration is None:
        indices = np.rint(np.linspace(0, frame_count - 1, num_frames)).astype(int)
    else:
        count = min(num_frames, max(1, int(official_lvb_duration)))
        valid_frames = min(frame_count, max(1, int(official_lvb_duration * fps)))
        step = max(1, valid_frames // count)
        indices = np.asarray([min(frame_count - 1, step * i) for i in range(count)])

    outputs = [output_dir / f"{i:02d}.jpg" for i in range(len(indices))]
    if not all(path.is_file() for path in outputs):
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            frames = reader.get_batch(indices).asnumpy()
        except decord.DECORDError:
            frames, indices = _read_frames_with_tail_fallback(video_path, indices, frame_count)
        for path, frame in zip(outputs, frames):
            Image.fromarray(frame).save(path, quality=JPEG_QUALITY)
    return [str(path) for path in outputs], [float(index) / fps for index in indices]


def _sample_video_opencv(video_path: Path, output_dir: Path, num_frames: int) -> tuple[list[str], list[float]]:
    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot read video with decord or OpenCV: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    if frame_count <= 0 or fps <= 0:
        cap.release()
        raise RuntimeError(f"Cannot determine frame count/FPS with OpenCV: {video_path}")
    indices = np.rint(np.linspace(0, frame_count - 1, num_frames)).astype(int)
    outputs = [output_dir / f"{i:02d}.jpg" for i in range(len(indices))]
    used = []
    if not all(path.is_file() for path in outputs):
        output_dir.mkdir(parents=True, exist_ok=True)
        last = None
        for path, index in zip(outputs, indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                if last is None:
                    cap.release()
                    raise RuntimeError(f"OpenCV failed to decode frame {index} of {video_path}")
                frame, index = last
            last = (frame, index)
            used.append(int(index))
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(path, quality=JPEG_QUALITY)
    else:
        used = [int(i) for i in indices]
    cap.release()
    return [str(path) for path in outputs], [float(index) / fps for index in used]


def _read_frames_with_tail_fallback(video_path: Path, indices, frame_count: int):
    import decord
    import numpy as np

    backoff = max(1, frame_count // 100)
    frames, used = [], []
    reader = decord.VideoReader(str(video_path), num_threads=1)
    for index in (int(i) for i in indices):
        candidate = index
        for _ in range(20):
            try:
                frames.append(reader[candidate].asnumpy())
                used.append(candidate)
                break
            except decord.DECORDError:
                candidate = max(0, candidate - backoff)
                reader = decord.VideoReader(str(video_path), num_threads=1)
        else:
            raise RuntimeError(f"Cannot decode frame {index} (or earlier fallbacks) of {video_path}")
    return frames, np.asarray(used)

