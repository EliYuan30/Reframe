#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import re
from pathlib import Path
from string import ascii_uppercase

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


try:
    from .common import benchmark_output_root, raw_dataset_root
except ImportError:
    from evaluation.adapters.common import benchmark_output_root, raw_dataset_root


SOURCE_ROOT = raw_dataset_root("datasets--yifanzhang114--MME-RealWorld-Lmms-eval")
METADATA_DIR = SOURCE_ROOT / "data"
OUTPUT_ROOT = benchmark_output_root()
IMAGE_DIR = OUTPUT_ROOT / "mme_realworld_images"
OUTPUT_PATH = OUTPUT_ROOT / "mme_realworld.parquet"
READ_COLUMNS = (
    "bytes",
    "index",
    "question",
    "multi-choice options",
    "answer",
    "category",
    "l2-category",
)
OPTION_PATTERN = re.compile(r"^\s*\(([A-Z])\)\s*(.+?)\s*$", re.DOTALL)
FORMAT_SUFFIXES = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}


def normalize_options(values: list[str]) -> list[tuple[str, str]]:
    options = []
    for value in values:
        match = OPTION_PATTERN.match(str(value))
        if match is None:
            raise ValueError(f"Invalid MME-RealWorld option: {value!r}")
        options.append((match.group(1), match.group(2)))
    if len(options) < 2:
        raise ValueError("MME-RealWorld question has fewer than two options")
    if len(options) == 5:
        labels = ascii_uppercase[: len(options)]
        options = [(label, text) for label, (_, text) in zip(labels, options)]
    return options


def format_question(question: str, options: list[tuple[str, str]]) -> str:
    labels = list(dict.fromkeys(label for label, _ in options))
    lines = [f"Question: {str(question).strip()}", "Options:"]
    lines.extend(f"{label}. {text}" for label, text in options)
    lines.append(
        "Reason about the image, then select the single best option using one of "
        f"{', '.join(labels)} and put the final answer letter in \\boxed{{}}, for "
        "example \\boxed{A}."
    )
    return "\n".join(lines)


def existing_image(index: int) -> Path | None:
    matches = [
        IMAGE_DIR / f"{index}{suffix}"
        for suffix in FORMAT_SUFFIXES.values()
        if (IMAGE_DIR / f"{index}{suffix}").is_file()
    ]
    if len(matches) > 1:
        raise ValueError(f"Multiple exported images for MME-RealWorld index {index}")
    return matches[0] if matches else None


def export_image(index: int, encoded: str) -> str:
    existing = existing_image(index)
    if existing is not None:
        return str(existing)

    data = base64.b64decode(str(encoded), validate=True)
    with Image.open(io.BytesIO(data)) as image:
        suffix = FORMAT_SUFFIXES.get(image.format or "")
    if suffix is None:
        raise ValueError(f"Unsupported image format for MME-RealWorld index {index}")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = IMAGE_DIR / f"{index}{suffix}"
    output_path.write_bytes(data)
    return str(output_path)


def load_shards() -> list[Path]:
    shards = sorted(METADATA_DIR.glob("train-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No MME-RealWorld shards under {METADATA_DIR}")
    return shards


def main() -> None:
    shards = load_shards()
    total = sum(pq.ParquetFile(path).metadata.num_rows for path in shards)
    rows = []
    duplicate_option_label_rows = 0

    with tqdm(total=total, desc="Preparing MME-RealWorld") as progress:
        for path in shards:
            parquet = pq.ParquetFile(path)
            missing = set(READ_COLUMNS) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")

            for batch in parquet.iter_batches(
                batch_size=1,
                columns=READ_COLUMNS,
            ):
                row = batch.to_pylist()[0]
                index = int(row["index"])
                options = normalize_options(row["multi-choice options"])
                labels = [label for label, _ in options]
                answer = str(row["answer"]).strip().upper()
                if answer not in labels:
                    raise ValueError(
                        f"MME-RealWorld index {index} answer {answer!r} is not an option"
                    )
                duplicate_option_label_rows += int(len(labels) != len(set(labels)))

                rows.append(
                    {
                        "qid": f"mme-realworld-{index}",
                        "question": format_question(row["question"], options),
                        "image": export_image(index, row["bytes"]),
                        "answer": f"\\boxed{{{answer}}}",
                        "is_video": False,
                        "category": str(row["category"]),
                        "l2_category": str(row["l2-category"]),
                    }
                )
                progress.update()

    output = pd.DataFrame(rows)
    if len(output) != total:
        raise ValueError(f"Expected {total} rows, built {len(output)}")
    if output["qid"].duplicated().any():
        raise ValueError("MME-RealWorld qid values are not unique")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(output)} questions to {OUTPUT_PATH}")
    print(f"Wrote images under {IMAGE_DIR}")
    if duplicate_option_label_rows:
        print(
            "Warning: preserved duplicate upstream option labels in "
            f"{duplicate_option_label_rows} rows"
        )


if __name__ == "__main__":
    main()
