#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm

try:
    from .common import benchmark_output_root, raw_dataset_root
except ImportError:
    from evaluation.adapters.common import benchmark_output_root, raw_dataset_root


SOURCE_ROOT = raw_dataset_root("datasets--moonshotai--PerceptionBench")
METADATA_PATH = SOURCE_ROOT / "PerceptionBench.jsonl"
OUTPUT_ROOT = benchmark_output_root()
IMAGE_DIR = OUTPUT_ROOT / "perceptionbench_images"
OUTPUT_PATH = OUTPUT_ROOT / "perceptionbench.parquet"
DATA_URI_PATTERN = re.compile(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", re.DOTALL)
PLACEHOLDER_PATTERN = re.compile(r"<\|image_(\d+)\|>")
EXTENSIONS = {"jpeg": "jpg", "jpg": "jpg", "png": "png", "webp": "webp", "gif": "gif", "bmp": "bmp"}


def decode_data_uri(value: str) -> tuple[str, bytes]:
    match = DATA_URI_PATTERN.match(str(value))
    if not match:
        raise ValueError(f"Unsupported PerceptionBench image payload: {str(value)[:40]!r}")
    subtype = match.group(1).lower()
    if subtype not in EXTENSIONS:
        raise ValueError(f"Unsupported PerceptionBench image type: {subtype}")
    return EXTENSIONS[subtype], base64.b64decode(match.group(2))


def export_data_uri(value: str, stem: Path) -> str:
    extension, data = decode_data_uri(value)
    path = stem.with_suffix(f".{extension}")
    if not path.is_file():
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    return str(path)


def format_question(problem: str, hint: str, image_count: int) -> str:
    placeholders = [int(value) for value in PLACEHOLDER_PATTERN.findall(problem)]
    if sorted(placeholders) != list(range(1, image_count + 1)):
        raise ValueError(f"PerceptionBench placeholders {placeholders} do not match {image_count} images")
    if image_count == 1:
        text = PLACEHOLDER_PATTERN.sub("", problem)
        prefix = ""
    else:
        text = PLACEHOLDER_PATTERN.sub(lambda match: f"[Image {match.group(1)}]", problem)
        prefix = f"The {image_count} images are given in order as Image 1 to Image {image_count}.\n"
    parts = [prefix + "Question: " + text.strip()]
    if str(hint or "").strip():
        parts.append("Hint: " + str(hint).strip())
    parts.append(
        "Reason about the visual input carefully, then give a short final answer "
        "(an option letter, number, word or short phrase as the question requires) in \\boxed{}."
    )
    return "\n".join(parts)


def main() -> None:
    rows = []
    with METADATA_PATH.open(encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Preparing PerceptionBench"):
            if not line.strip():
                continue
            item = json.loads(line)
            index = int(item["index"])
            images = [
                export_data_uri(value, IMAGE_DIR / f"{index}_{position}")
                for position, value in enumerate(item["image"], start=1)
            ]
            if not images:
                raise ValueError(f"PerceptionBench {index} has no images")
            rows.append({
                "qid": f"perceptionbench-{index}",
                "question": format_question(str(item["problem"]), str(item.get("hint") or ""), len(images)),
                "image": images,
                "answer": str(item["answer"]).strip(),
                "is_video": False,
                "category": str(item["error_category"]),
                "source_benchmark": str(item.get("source_bmk") or ""),
            })
    output = pd.DataFrame(rows)
    if output["qid"].duplicated().any():
        raise ValueError("PerceptionBench index values are not unique")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(output)} questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
