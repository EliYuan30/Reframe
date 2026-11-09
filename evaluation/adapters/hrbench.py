#!/usr/bin/env python3
from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
from tqdm import tqdm


try:
    from .common import benchmark_output_root, raw_dataset_root
except ImportError:
    from evaluation.adapters.common import benchmark_output_root, raw_dataset_root


SOURCE_ROOT = raw_dataset_root("datasets--DreamMr--HR-Bench")
OUTPUT_ROOT = benchmark_output_root()
OPTION_COLUMNS = ("A", "B", "C", "D")


def format_question(question: str, options: list[tuple[str, str]]) -> str:
    lines = [f"Question: {str(question).strip()}", "Options:"]
    lines.extend(f"{label}. {text}" for label, text in options)
    lines.append(
        "Reason about the image, then select the single best option from A through D "
        "and put the final answer letter in \\boxed{}, for example \\boxed{A}."
    )
    return "\n".join(lines)


def export_image(version: str, index: int, encoded: str) -> str:
    image_dir = OUTPUT_ROOT / f"hrbench_{version}_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    output_path = image_dir / f"{index}.jpg"
    if not output_path.is_file():
        output_path.write_bytes(base64.b64decode(encoded, validate=True))
    return str(output_path)


def prepare(version: str) -> None:
    if version not in {"4k", "8k"}:
        raise ValueError(f"Unsupported HR-Bench version: {version}")
    source_path = SOURCE_ROOT / f"hr_bench_{version}.parquet"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    metadata = pd.read_parquet(source_path)
    rows = []
    for row in tqdm(
        metadata.to_dict("records"),
        total=len(metadata),
        desc=f"Preparing HR-Bench {version.upper()}",
    ):
        index = int(row["index"])
        answer = str(row["answer"]).strip().upper()
        if answer not in OPTION_COLUMNS:
            raise ValueError(f"Invalid HR-Bench answer {answer!r} at index {index}")
        options = [(label, str(row[label]).strip()) for label in OPTION_COLUMNS]
        rows.append(
            {
                "qid": f"hrbench-{version}-{index}",
                "question": format_question(row["question"], options),
                "image": export_image(version, index, row["image"]),
                "answer": f"\\boxed{{{answer}}}",
                "is_video": False,
                "category": str(row["category"]),
                "cycle_category": int(row["cycle_category"]),
            }
        )

    output = pd.DataFrame(rows)
    if output["qid"].duplicated().any():
        raise ValueError(f"HR-Bench {version} qid values are not unique")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"hrbench_{version}.parquet"
    output.to_parquet(output_path, index=False)
    print(f"Wrote {len(output)} questions to {output_path}")

