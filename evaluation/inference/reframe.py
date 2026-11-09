#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import re
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "ms-swift")]

from evaluation.benchmarks import DATASETS

from reframe.plugin import (  # noqa: E402
    _correctness,
    _tool_instructions,
    _tool_schemas,
    _valid_output,
)


NO_TOOL_INSTRUCTIONS = (
    "Reason about the visual input and answer the question without calling tools. "
    "Put the final answer in \\boxed{}."
)





def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("image", "video"), required=True)
    parser.add_argument(
        "--benchmark",
        choices=DATASETS,
        required=True,
    )
    parser.add_argument("--with-tools", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def answer(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def boxed_answer(value: Any) -> str:
    value = answer(value).strip()
    match = re.fullmatch(r"\\boxed\s*\{([^{}]*)\}", value)
    if match:
        value = match.group(1).strip()
    return f"\\boxed{{{value}}}" if value else ""


def target_answer(value: Any, benchmark: str) -> str:
    return boxed_answer(value)




def score_answer(output: str, target: str, benchmark: str) -> float:
    return float(_correctness(output, target))


def valid_answer(output: str, benchmark: str) -> float:
    return float(_valid_output(output))


def build_request(
    row: dict[str, Any], mode: str, with_tools: bool, benchmark: str
) -> dict[str, Any]:
    if mode not in {"image", "video"}:
        raise ValueError(f"Unsupported modality: {mode}")
    is_video = bool(row.get("is_video", mode == "video"))
    media = row.get("image")
    media_items = list(media) if isinstance(media, (list, tuple)) else [media]
    metadata = row.get("tool_metadata") or {}
    if not isinstance(metadata, str):
        metadata = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))

    system_content = _tool_instructions(mode) if with_tools else NO_TOOL_INSTRUCTIONS
    visual_prefix = "<video>" if is_video else "<image>" * len(media_items)
    user_content = visual_prefix + str(row["question"])

    request = {
        "messages": [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "tools": _tool_schemas(mode) if with_tools else [],
        "data_dict": {
            "qid": str(row["qid"]),
            "solution": target_answer(row["answer"], benchmark),
            "tool_metadata": metadata,
            "teacher_route": "video" if is_video else "image",
            "tools_enabled": with_tools,
        },
    }
    if is_video:
        request["videos"] = [media_items]
        tool_frames = row.get("tool_video_frames")
        if tool_frames is not None and not (isinstance(tool_frames, float) and tool_frames != tool_frames):
            if not isinstance(tool_frames, str):
                tool_frames = json.dumps([str(x) for x in tool_frames], separators=(",", ":"))
            if json.loads(tool_frames):
                request["data_dict"]["tool_video_frames"] = tool_frames
    else:
        request["images"] = media_items
    return request


def assistant_contents(messages: list[dict[str, Any]]) -> list[str]:
    return [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "assistant"
    ]


def trajectory_text(messages: list[dict[str, Any]]) -> str:
    parts = []
    for message in messages:
        content = str(message.get("content", ""))
        if message.get("role") == "assistant" or content.startswith("<tool_response>"):
            parts.append(content)
    return "\n".join(parts)


def build_record(
    row: dict[str, Any], output: dict[str, Any], with_tools: bool, benchmark: str
) -> dict[str, Any]:
    messages = output.get("messages") or []
    assistants = assistant_contents(messages)
    if not assistants:
        raise ValueError(f"No assistant message returned for qid={row['qid']}")
    target = target_answer(row["answer"], benchmark)
    first = assistants[0]
    final = assistants[-1]
    final_correctness = score_answer(final, target, benchmark)
    rollout_infos = output.get("rollout_infos") or {}
    return {
        "qids": str(row["qid"]),
        "questions": str(row["question"]),
        "solutions": trajectory_text(messages),
        "gts": target,
        "round0_correctness": score_answer(first, target, benchmark),
        "round1_correctness": final_correctness,
        "match": final_correctness,
        "validity": valid_answer(final, benchmark),
        "final_answer": final,
        "tool_calls": rollout_infos.get("tool_calls") or [],
        "tools_enabled": with_tools,
    }


def write_metrics(log_dir: Path) -> None:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(log_dir.glob("sample.eval_iter0_*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                records.setdefault(str(record["qids"]), record)
    if not records:
        raise ValueError(f"No evaluation records found under {log_dir}")
    correctness = []
    for record in records.values():
        value = record.get("match")
        if value is None:
            value = record.get("round1_correctness", record["round0_correctness"])
        correctness.append(float(value))
    correct = sum(correctness)
    metrics = {
        "accuracy": correct / len(records),
        "correct": correct,
        "total": len(records),
    }
    output = log_dir / "metrics_iter0.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(metrics, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    rows = pq.read_table(args.data).to_pylist()
    if args.max_rows is not None:
        rows = rows[:args.max_rows]
    if not rows:
        print(f"No remaining rows in {args.data}")
        return
    qids = [str(row["qid"]) for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError(f"Duplicate qids in {args.data}")

    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / "sample.eval_iter0_0.jsonl"
    done_qids = set()
    if log_path.is_file():
        valid_lines, torn = [], 0
        with log_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    done_qids.add(str(json.loads(line)["qids"]))
                    valid_lines.append(line if line.endswith("\n") else line + "\n")
                except (ValueError, KeyError):
                    torn += 1
        if torn:
            temporary = log_path.with_suffix(".jsonl.tmp")
            temporary.write_text("".join(valid_lines), encoding="utf-8")
            temporary.replace(log_path)
            print(f"resume: dropped {torn} torn line(s)", flush=True)
    if done_qids:
        rows = [row for row in rows if str(row["qid"]) not in done_qids]
        print(f"resume: {len(done_qids)} qids already recorded, {len(rows)} remaining", flush=True)
        if not rows:
            write_metrics(args.log_dir)
            return
    request_config = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }

    endpoints = [f"http://{args.host}:{args.port}/infer/"]
    batches = queue.Queue()
    for start in range(0, len(rows), args.batch_size):
        batches.put(rows[start:start + args.batch_size])
    lock = threading.Lock()
    progress = {"completed": 0}

    def run_rows(endpoint: str, batch_rows: list) -> list:
        infer_requests = [build_request(row, args.mode, args.with_tools, args.benchmark) for row in batch_rows]
        response = requests.post(
            endpoint,
            json={"infer_requests": infer_requests, "request_config": request_config},
            timeout=86400,
        )
        if response.status_code >= 500:
            if len(batch_rows) > 1:
                half = len(batch_rows) // 2
                return run_rows(endpoint, batch_rows[:half]) + run_rows(endpoint, batch_rows[half:])
            row = batch_rows[0]
            print(f"server error for qid={row['qid']}: recorded as incorrect", flush=True)
            target = target_answer(row["answer"], args.benchmark)
            return [{
                "qids": str(row["qid"]), "questions": str(row["question"]), "solutions": "", "gts": target,
                "round0_correctness": 0.0, "round1_correctness": 0.0, "match": 0.0, "validity": 0.0,
                "final_answer": "", "tool_calls": [], "tools_enabled": args.with_tools,
                "server_error": response.text[:500],
            }]
        response.raise_for_status()
        outputs = response.json()
        if len(outputs) != len(batch_rows):
            raise RuntimeError(f"Server returned {len(outputs)} outputs for {len(batch_rows)} requests")
        return [build_record(row, output, args.with_tools, args.benchmark) for row, output in zip(batch_rows, outputs)]

    def serve(endpoint: str) -> None:
        while True:
            try:
                batch_rows = batches.get_nowait()
            except queue.Empty:
                return
            lines = [
                json.dumps(record, ensure_ascii=False) + "\n"
                for record in run_rows(endpoint, batch_rows)
            ]
            with lock:
                writer.writelines(lines)
                writer.flush()
                progress["completed"] += len(batch_rows)
                print(f"completed {progress['completed']}/{len(rows)}", flush=True)

    with log_path.open("a", encoding="utf-8") as writer:
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            for future in [pool.submit(serve, endpoint) for endpoint in endpoints]:
                future.result()

    write_metrics(args.log_dir)


if __name__ == "__main__":
    main()
