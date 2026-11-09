#!/usr/bin/env python3
from pathlib import Path

import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm


try:
    from .common import benchmark_output_root, raw_dataset_root
except ImportError:
    from evaluation.adapters.common import benchmark_output_root, raw_dataset_root


SOURCE_ROOT = raw_dataset_root("datasets--craigwu--vstar_bench")
METADATA_PATH = SOURCE_ROOT / "test_questions.jsonl"
OUTPUT_ROOT = benchmark_output_root()
OUTPUT_PATH = OUTPUT_ROOT / "vstar.parquet"


OLD_ANSWER_INSTRUCTION = "Answer with the option's letter from the given choices directly."


def normalize_option_labels(question: str) -> str:
    return re.sub(r"(?m)^\(([A-H])\)\s*", r"\1 ", question).strip()


def format_question(question: str) -> str:
    labels = re.findall(r"(?m)^\(([A-H])\)\s*", question)
    if not labels:
        raise ValueError("VStar question has no labeled options")
    question = normalize_option_labels(question)
    if question.endswith(OLD_ANSWER_INSTRUCTION):
        question = question[:-len(OLD_ANSWER_INSTRUCTION)].rstrip()
    return (
        f"{question}\n"
        "Reason about the image, then select the single best option from "
        f"{labels[0]} through {labels[-1]} and put the final answer letter in "
        "\\boxed{}, for example \\boxed{A}."
    )


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "r") as f:
        metadata = [json.loads(line) for line in f.readlines()]
    
    for qid, item in enumerate(metadata):
        item["qid"] = item.pop("question_id")
        item["question"] = format_question(item.pop("text"))
        item["image"] = str(SOURCE_ROOT) + "/" + item.pop("image")
        item["answer"] = f"\\boxed{{{item.pop('label')}}}"
        item["category"] = item.pop("category")
        pass
            

    df = pd.DataFrame(metadata)
    df.to_parquet(OUTPUT_PATH, index=False)
