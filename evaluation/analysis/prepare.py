#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

CROP_TOOL = "crop"
SCALE = 1000.0
RELAX = 1.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", type=Path, required=True,
                        help="V* benchmark parquet")
    parser.add_argument("--output", type=Path, required=True,
                        help="destination probe parquet")
    return parser.parse_args()


def to_grid(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> list[int]:
    grid = [
        round(x1 / width * SCALE),
        round(y1 / height * SCALE),
        round(x2 / width * SCALE),
        round(y2 / height * SCALE),
    ]
    grid = [max(0, min(1000, int(value))) for value in grid]
    if grid[2] <= grid[0]:
        grid[2] = min(1000, grid[0] + 1)
    if grid[3] <= grid[1]:
        grid[3] = min(1000, grid[1] + 1)
    return grid


def main() -> None:
    args = parse_args()
    frame = pd.read_parquet(args.bench)
    rows = []
    mismatches = []
    areas_raw = []
    areas_relaxed = []
    for _, row in frame.iterrows():
        image_path = Path(row["image"])
        sidecar = image_path.with_suffix(".json")
        annotation = json.loads(sidecar.read_text())
        boxes = annotation["bbox"]
        if not boxes:
            raise ValueError(f"No bbox for {image_path}")
        first_line = str(row["question"]).splitlines()[0].strip()
        if annotation.get("question", "").strip() != first_line:
            mismatches.append((str(row["qid"]), first_line, annotation.get("question")))
        with Image.open(image_path) as image:
            width, height = image.size
        xs1 = min(box[0] for box in boxes)
        ys1 = min(box[1] for box in boxes)
        xs2 = max(box[0] + box[2] for box in boxes)
        ys2 = max(box[1] + box[3] for box in boxes)
        cx, cy = (xs1 + xs2) / 2.0, (ys1 + ys2) / 2.0
        half_w = (xs2 - xs1) * RELAX / 2.0
        half_h = (ys2 - ys1) * RELAX / 2.0
        rx1, ry1 = max(0.0, cx - half_w), max(0.0, cy - half_h)
        rx2, ry2 = min(float(width), cx + half_w), min(float(height), cy + half_h)
        relaxed = to_grid(rx1, ry1, rx2, ry2, width, height)
        raw = to_grid(xs1, ys1, xs2, ys2, width, height)
        areas_raw.append((raw[2] - raw[0]) * (raw[3] - raw[1]) / SCALE**2)
        areas_relaxed.append((relaxed[2] - relaxed[0]) * (relaxed[3] - relaxed[1]) / SCALE**2)
        metadata = {
            "schema_version": 2,
            "source": "vstar",
            "domain": str(row["category"]),
            "preferred_tool": CROP_TOOL,
            "tool_applicable": True,
            "canonical_tool_plan": [
                {"name": CROP_TOOL, "arguments": {"target_image": 1, "bbox_2d": relaxed}}
            ],
            "evidence_bbox": relaxed,
            "original_evidence_bbox": raw,
            "target_objects": list(annotation.get("target_object") or []),
            "gt_relaxation": "bbox_side_1.5",
            "image_size": [width, height],
        }
        rows.append({
            "qid": str(row["qid"]),
            "question": row["question"],
            "image": row["image"],
            "answer": row["answer"],
            "category": row["category"],
            "tool_metadata": json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        })

    if mismatches:
        for qid, expected, actual in mismatches[:5]:
            print(f"question mismatch qid={qid}: parquet={expected!r} sidecar={actual!r}")
        raise ValueError(f"{len(mismatches)} sidecar/question mismatches")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(args.output, index=False)

    def pct(values: list[float], q: float) -> float:
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))]

    by_category = pd.DataFrame(rows).groupby("category")["qid"].count().to_dict()
    print(f"wrote {len(rows)} rows -> {args.output}")
    print(f"categories: {by_category}")
    print(
        "gt area ratio raw p10/p50/p90: "
        f"{pct(areas_raw, 0.1):.4f}/{pct(areas_raw, 0.5):.4f}/{pct(areas_raw, 0.9):.4f}"
    )
    print(
        "gt area ratio relaxed p10/p50/p90: "
        f"{pct(areas_relaxed, 0.1):.4f}/{pct(areas_relaxed, 0.5):.4f}/{pct(areas_relaxed, 0.9):.4f}"
    )


if __name__ == "__main__":
    main()
