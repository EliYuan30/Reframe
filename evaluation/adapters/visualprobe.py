import json
from pathlib import Path
import pandas as pd
from .common import benchmark_output_root, raw_dataset_root


def prepare(split: str) -> None:
    if split not in {"easy", "medium", "hard"}:
        raise ValueError(split)
    source = raw_dataset_root(f"datasets--Mini-o3--VisualProbe_{split.title()}")
    rows = []
    for row in json.loads((source / "val.json").read_text(encoding="utf-8")):
        images = row.get("images") or []
        if len(images) != 1:
            raise ValueError(f"Expected one VisualProbe image for {row['doc_id']}")
        path = source / images[0]
        if not path.is_file():
            path = source / "data" / Path(images[0]).name
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({
            "qid": f"visualprobe-{split}-{row['doc_id']}",
            "question": str(row["problem"]).replace("<image>", "").strip()
                        + "\nPut the final short answer in \\boxed{}.",
            "image": str(path), "answer": str(row["solution"]).strip(), "is_video": False,
        })
    output = benchmark_output_root() / f"visualprobe_{split}.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output, index=False)
    print(f"Wrote {len(rows)} questions to {output}")
