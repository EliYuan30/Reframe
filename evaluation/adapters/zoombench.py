#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

try:
    from .common import benchmark_output_root, export_image, raw_dataset_root
except ImportError:
    from evaluation.adapters.common import benchmark_output_root, export_image, raw_dataset_root


SOURCE_ROOT = raw_dataset_root("datasets--inclusionAI--ZoomBench")
METADATA_PATH = SOURCE_ROOT / "data/test.parquet"
OUTPUT_ROOT = benchmark_output_root()
IMAGE_DIR = OUTPUT_ROOT / "zoombench_images"
OUTPUT_PATH = OUTPUT_ROOT / "zoombench.parquet"


def main() -> None:
    metadata = pd.read_parquet(METADATA_PATH)
    rows = []
    for row in tqdm(metadata.to_dict("records"), desc="Preparing ZoomBench"):
        qid = str(row["id"])
        rows.append({
            "qid": f"zoombench-{qid}",
            "question": str(row["query"]).strip() + "\nPut the final answer or option letter in \\boxed{}.",
            "image": export_image(row["image"], IMAGE_DIR / f"{qid}.jpg"),
            "answer": str(row["response"]).strip(),
            "is_video": False,
            "category": str(row["question_type"]),
        })
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(rows)} questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
