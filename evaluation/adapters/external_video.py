from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from evaluation.inference.protocols import PATCH, smart_resize

VIDEO_BENCHMARKS = ("mvbench", "videomme", "videomme_v2", "mlvu", "videoocr", "vsi_bench",
                    "vstar_video")
FRAME_TOKENS = 128


def resize_frame(job: tuple[str, str, int]) -> None:
    source, target, max_pixels = job
    target_path = Path(target)
    if target_path.is_file():
        return
    with Image.open(source) as image:
        image = image.convert("RGB")
        height, width = smart_resize(image.height, image.width, max_pixels=max_pixels)
        if (width, height) != image.size:
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        partial = target_path.with_name(target_path.name + ".partial")
        image.save(partial, format="JPEG", quality=95)
        partial.replace(target_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmarks", nargs="+", default=list(VIDEO_BENCHMARKS), choices=VIDEO_BENCHMARKS)
    parser.add_argument("--frame-tokens", type=int, default=FRAME_TOKENS)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    source_root, output_root = args.source_root.resolve(), args.output_root.resolve()
    max_pixels = args.frame_tokens * PATCH * PATCH
    for name in args.benchmarks:
        table = pq.read_table(source_root / f"{name}.parquet")
        if "tool_video_frames" in table.schema.names:
            raise ValueError(f"{name}: source rows already list tool_video_frames")
        frames = table.column("image").to_pylist()
        mapping: dict[str, str] = {}
        for row_frames in frames:
            for frame in row_frames:
                if frame not in mapping:
                    relative = Path(frame).resolve().relative_to(source_root)
                    mapping[frame] = str(output_root / "frames" / relative.with_suffix(".jpg"))
        with ProcessPoolExecutor(args.workers) as pool:
            list(pool.map(resize_frame, [(s, t, max_pixels) for s, t in mapping.items()], chunksize=256))
        table = table.set_column(table.schema.get_field_index("image"), "image",
                                 pa.array([[mapping[f] for f in row] for row in frames]))
        table = table.append_column("tool_video_frames", pa.array([json.dumps(row) for row in frames]))
        output_root.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, output_root / f"{name}.parquet")
        print(f"{name}: {len(frames)} rows, {len(mapping)} frames at <= {args.frame_tokens} tokens", flush=True)


if __name__ == "__main__":
    main()
