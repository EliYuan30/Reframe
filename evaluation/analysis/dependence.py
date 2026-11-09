#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import hashlib
import json
import os
import queue
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import numpy as np
import pyarrow.parquet as pq
import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "ms-swift")]

from reframe.tool_rewards import (  # noqa: E402
    CROP_BBOX_SCALE,
    CROP_TOOL,
    crop_token_budget,
    parse_tool_bbox,
    parse_tool_metadata,
)
from evaluation.analysis.report import observation_dependence  # noqa: E402
from evaluation.inference.tools import ONE_CALL_NOTE  # noqa: E402
from reframe.plugin import (  # noqa: E402
    ObservationGatedScheduler,
    _correctness,
    _format_tool_call,
    _resize_to_visual_grid,
    _to_pil,
    _tool_instructions,
    _tool_schemas,
    _valid_output,
)
def qid_seed(qid: str, salt: str = "visdep-v1") -> int:
    return int(hashlib.sha256(f"{salt}:{qid}".encode("utf-8")).hexdigest()[:16], 16)


NO_TOOL_INSTRUCTIONS = (
    "Reason about the visual input and answer the question without calling tools. "
    "Put the final answer in \\boxed{}."
)
BRANCHES = ("direct", "natural", "gt", "random", "blank")
INTERVENTIONS = ("gt", "random", "blank")
PAIRED_EFFECTS = (
    ("natural", "direct"),
    ("natural", "blank"),
    ("natural", "random"),
    ("gt", "natural"),
    ("gt", "blank"),
    ("gt", "random"),
)
PROBE_VERSION = 2
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": -1}
GRAY = (128, 128, 128)
RANDOM_SALT = "visdep-v1"
RANDOM_MAX_IOU = 0.1
RANDOM_MAX_DRAWS = 100
BUDGET_ENV_DEFAULTS = {
    "REFRAME_CROP_MAX_TOKEN_NUM": "1960",
    "REFRAME_SELECT_MAX_TOKEN_NUM": "400",
    "REFRAME_MAX_TOOL_VISUAL_TOKENS": "4096",
    "VIDEO_MAX_TOKEN_NUM": "128",
    "MAX_PIXELS": "",
    "MIN_PIXELS": "",
    "IMAGE_MIN_TOKEN_NUM": "",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("rollout", "summarize", "dry_build", "precompute"), default="rollout"
    )
    parser.add_argument(
        "--intervention-cache",
        type=Path,
        help="directory of pre-rendered gt/random/blank images shared by all checkpoints; requests then "
        "carry file paths (the exact JPEG bytes the scheduler would produce) instead of data URLs",
    )
    parser.add_argument("--workers", type=int, default=16, help="precompute stage: worker processes")
    parser.add_argument("--modality", choices=("image",), default="image")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", help="checkpoint path (rollout manifest identity)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--ports", type=int, nargs="+", help="rollout-server ports, served round-robin")
    parser.add_argument("--branches", nargs="+", choices=BRANCHES, default=list(BRANCHES))
    parser.add_argument("--max-rows", type=int, help="probe only the first N rows")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, help="per-round completion limit (default: eval value)")
    parser.add_argument(
        "--intervention-tools",
        choices=("schemas", "none"),
        default="schemas",
        help="gt/random/blank: keep the tool schemas in the prompt (identical prefix to natural) "
        "or drop them; tool execution is disabled either way",
    )
    parser.add_argument("--retry-errors", action="store_true", help="re-run records saved with a server error")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--name", help="checkpoint label for the markdown row")
    parser.add_argument("--dry-build-rows", type=int, default=5)
    args = parser.parse_args(argv)
    if args.max_tokens is None:
        args.max_tokens = 8192
    if args.stage == "precompute" and args.intervention_cache is None:
        parser.error("--intervention-cache is required for the precompute stage")
    if args.stage not in ("dry_build", "precompute") and args.output_dir is None:
        parser.error("--output-dir is required")
    if args.stage == "rollout" and (not args.ports or not args.model):
        parser.error("--ports and --model are required for the rollout stage")
    return args


def boxed_answer(value: Any) -> str:
    import re

    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    value = "" if value is None else str(value).strip()
    match = re.fullmatch(r"\\boxed\s*\{([^{}]*)\}", value)
    if match:
        value = match.group(1).strip()
    return f"\\boxed{{{value}}}" if value else ""


def media_values(row: dict[str, Any]) -> list[Any]:
    media = row.get("image")
    if media is None:
        media = row.get("images")
    return list(media) if isinstance(media, (list, tuple)) else [media]


def metadata_json(row: dict[str, Any]) -> str:
    metadata = row.get("tool_metadata") or {}
    if not isinstance(metadata, str):
        metadata = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
    return metadata




def canonical_call(row: dict[str, Any], modality: str) -> dict[str, Any] | None:
    metadata = parse_tool_metadata(row.get("tool_metadata"))
    for call in metadata.get("canonical_tool_plan") or []:
        arguments = call.get("arguments") or {}
        if modality == "image" and call.get("name") == CROP_TOOL:
            bbox = parse_tool_bbox(arguments.get("bbox_2d"))
            target = arguments.get("target_image")
            if bbox is not None and type(target) is int and 1 <= target <= len(media_values(row)):
                return {"name": CROP_TOOL, "arguments": {"target_image": target, "bbox_2d": list(bbox)}}
    return None


_EXECUTOR = object.__new__(ObservationGatedScheduler)


def real_crop(images: list[Any], call: dict[str, Any]) -> tuple[str, str]:
    request = SimpleNamespace(images=list(images))
    state = {
        "frames": [],
        "image_sources": [
            {"value": value, "bbox": [0, 0, 1000, 1000], "original_target_image": index}
            for index, value in enumerate(images, 1)
        ],
        "original_drop": None,
    }
    response, _ = _EXECUTOR._crop(request, state, call["arguments"])
    return response, request.images[-1]




def data_url_size(url: str) -> tuple[int, int]:
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size


def box_iou(first: list[float], second: list[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = width * height
    union = (
        (first[2] - first[0]) * (first[3] - first[1])
        + (second[2] - second[0]) * (second[3] - second[1])
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def random_crop_origin(
    qid: str,
    image_size: tuple[int, int],
    gt_pixels: list[int],
    salt: str = RANDOM_SALT,
    max_iou: float = RANDOM_MAX_IOU,
    max_draws: int = RANDOM_MAX_DRAWS,
) -> dict[str, Any]:
    width, height = image_size
    crop_width, crop_height = gt_pixels[2] - gt_pixels[0], gt_pixels[3] - gt_pixels[1]
    rng = random.Random(qid_seed(qid, salt))

    def iou(origin: tuple[int, int]) -> float:
        x, y = origin
        return box_iou([x, y, x + crop_width, y + crop_height], gt_pixels)

    draws = []
    for draw in range(1, max_draws + 1):
        origin = (rng.randint(0, width - crop_width), rng.randint(0, height - crop_height))
        value = iou(origin)
        if value <= max_iou:
            return {"origin": origin, "iou": value, "draws": draw, "fallback": False}
        draws.append(origin)
    corners = [
        (0, 0),
        (width - crop_width, 0),
        (0, height - crop_height),
        (width - crop_width, height - crop_height),
    ]
    origin = min(corners + draws, key=iou)
    return {"origin": origin, "iou": iou(origin), "draws": max_draws, "fallback": True}


def image_intervention(row: dict[str, Any]) -> dict[str, Any] | None:
    call = canonical_call(row, "image")
    if call is None:
        return None
    images = media_values(row)
    response, gt_url = real_crop(images, call)
    target = call["arguments"]["target_image"]
    source = _to_pil(images[target - 1])
    width, height = source.size
    bbox = call["arguments"]["bbox_2d"]
    gt_pixels = [
        round(bbox[0] / CROP_BBOX_SCALE * width),
        round(bbox[1] / CROP_BBOX_SCALE * height),
        round(bbox[2] / CROP_BBOX_SCALE * width),
        round(bbox[3] / CROP_BBOX_SCALE * height),
    ]
    crop_width, crop_height = gt_pixels[2] - gt_pixels[0], gt_pixels[3] - gt_pixels[1]
    budget = crop_token_budget()
    placement = random_crop_origin(str(row["qid"]), (width, height), gt_pixels)
    x, y = placement["origin"]
    random_url = _resize_to_visual_grid(source.crop((x, y, x + crop_width, y + crop_height)), budget)[0]
    blank_url = _resize_to_visual_grid(Image.new("RGB", (crop_width, crop_height), GRAY), budget)[0]
    sizes = {data_url_size(url) for url in (gt_url, random_url, blank_url)}
    if len(sizes) != 1:
        raise RuntimeError(f"qid={row['qid']}: intervention images differ in size: {sizes}")
    random_bbox = [
        round(x / width * CROP_BBOX_SCALE),
        round(y / height * CROP_BBOX_SCALE),
        round((x + crop_width) / width * CROP_BBOX_SCALE),
        round((y + crop_height) / height * CROP_BBOX_SCALE),
    ]
    return {
        "call": call,
        "tool_response": response,
        "images": {"gt": [gt_url], "random": [random_url], "blank": [blank_url]},
        "info": {
            "gt_bbox": bbox,
            "random_bbox": random_bbox,
            "random_iou": round(placement["iou"], 6),
            "random_draws": placement["draws"],
            "random_fallback": placement["fallback"],
            "crop_size_px": [crop_width, crop_height],
            "returned_size_px": list(sizes.pop()),
        },
    }




def intervention(row: dict[str, Any], modality: str) -> dict[str, Any] | None:
    if modality != "image":
        raise ValueError("Observation dependence is evaluated on V* images only")
    return image_intervention(row)


def build_request(
    row: dict[str, Any],
    modality: str,
    branch: str,
    prepared: dict[str, Any] | None,
    intervention_tools: str = "schemas",
) -> dict[str, Any]:
    media = media_values(row)
    tools_enabled = branch == "natural"
    if branch == "direct":
        system, tools = NO_TOOL_INSTRUCTIONS, []
    elif branch == "natural":
        system, tools = _tool_instructions(modality), _tool_schemas(modality)
    elif branch in INTERVENTIONS:
        if prepared is None:
            raise ValueError(f"{branch} requires a canonical tool plan (qid={row['qid']})")
        system = _tool_instructions(modality)
        tools = _tool_schemas(modality) if intervention_tools == "schemas" else []
    else:
        raise ValueError(f"unknown branch {branch!r}")
    if modality != "image":
        raise ValueError("Observation dependence is evaluated on V* images only")
    visual_prefix = "<image>" * len(media)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": visual_prefix + str(row["question"])},
    ]
    request: dict[str, Any] = {
        "messages": messages,
        "tools": tools,
        "data_dict": {
            "qid": str(row["qid"]),
            "solution": boxed_answer(row["answer"]),
            "tool_metadata": metadata_json(row),
            "teacher_route": modality,
            "tools_enabled": tools_enabled,
        },
    }
    request["images"] = list(media)
    if branch in INTERVENTIONS:
        messages.append({"role": "assistant", "content": _format_tool_call(prepared["call"])})
        messages.append({"role": "user", "content": single_call_response(prepared["tool_response"])})
        returned = list(prepared["images"][branch])
        request["images"] = request["images"] + returned
    return request


def single_call_response(response: str) -> str:
    if not response.endswith("</tool_response>"):
        raise ValueError("Inserted tool response must end with </tool_response>")
    return response[:-len("</tool_response>")] + "\n" + ONE_CALL_NOTE + "</tool_response>"


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["qid"]), str(record["branch"])


def final_assistant(messages: list[dict[str, Any]]) -> str:
    assistants = [str(m.get("content", "")) for m in messages if m.get("role") == "assistant"]
    if not assistants:
        raise ValueError("Rollout returned no assistant message")
    return assistants[-1]


def trajectory_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "assistant" or str(message.get("content", "")).startswith("<tool_response>")
    )


def base_record(job: dict[str, Any], modality: str) -> dict[str, Any]:
    row = job["row"]
    record = {
        "qid": str(row["qid"]),
        "branch": job["branch"],
        "modality": modality,
        "target": boxed_answer(row["answer"]),
        "error": None,
    }
    prepared = job.get("prepared")
    if prepared is not None:
        info = prepared["info"]
        record["gt_bbox"] = info["gt_bbox"]
        if job["branch"] in INTERVENTIONS:
            record["intervention"] = info
    return record


def build_record(job: dict[str, Any], output: dict[str, Any], modality: str) -> dict[str, Any]:
    record = base_record(job, modality)
    messages = output.get("messages") or []
    final = final_assistant(messages)
    calls = (output.get("rollout_infos") or {}).get("tool_calls") or []
    natural = job["branch"] == "natural"
    record.update({
        "final_answer": final,
        "trajectory": trajectory_text(messages),
        "valid_output": _valid_output(final),
        "tool_calls": calls,
        "natural_called_tool": bool(calls) if natural else None,
        "natural_successful_tool": any(call.get("success") for call in calls) if natural else None,
        "continuation_tool_call": ("<tool_call>" in final) if job["branch"] in INTERVENTIONS else None,
        "correctness": float(_correctness(final, record["target"])),
    })
    return record


def error_record(job: dict[str, Any], modality: str, error: str) -> dict[str, Any]:
    record = base_record(job, modality)
    record.update({"error": error[:500], "final_answer": "", "trajectory": "", "tool_calls": [],
                   "correctness": None})
    return record


def load_records(path: Path, retry_errors: bool = False) -> tuple[list[dict[str, Any]], set]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.is_file():
        return [], set()
    valid_lines, torn = [], 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = record_key(record)
            except (ValueError, KeyError, TypeError):
                torn += 1
                continue
            valid_lines.append(line if line.endswith("\n") else line + "\n")
            existing = by_key.get(key)
            if existing is None or (existing.get("error") and not record.get("error")):
                by_key[key] = record
    if torn:
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(valid_lines), encoding="utf-8")
        temporary.replace(path)
        print(f"resume: dropped {torn} torn line(s) from {path}", flush=True)
    records = list(by_key.values())
    completed = {key for key, record in by_key.items() if not (retry_errors and record.get("error"))}
    return records, completed


def manifest_for(args: argparse.Namespace, data_rows: int) -> dict[str, Any]:
    return {
        "probe": "visual_dependence",
        "version": PROBE_VERSION,
        "modality": args.modality,
        "model": str(Path(args.model).resolve()) if args.model else None,
        "data": str(args.data.resolve()),
        "data_rows": data_rows,
        "max_tokens": args.max_tokens,
        **GREEDY,
        "intervention_tools": args.intervention_tools,
        "random_salt": RANDOM_SALT,
        "env": {name: os.environ.get(name, default) for name, default in BUDGET_ENV_DEFAULTS.items()},
    }


def check_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    path = output_dir / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) == manifest:
        return
    if path.exists() or (output_dir / "records.jsonl").exists():
        print(f"Settings differ from the previous run in {output_dir}; deleting its results and starting over.",
              flush=True)
        for name in ("records.jsonl", "summary.json"):
            (output_dir / name).unlink(missing_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_rows(path: Path, max_rows: int | None) -> tuple[list[dict[str, Any]], int]:
    rows = pq.read_table(path).to_pylist()
    qids = [str(row["qid"]) for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError(f"Duplicate qids in {path}")
    return (rows[:max_rows] if max_rows else rows), len(rows)


def plan_jobs(rows: list[dict[str, Any]], modality: str, branches: Iterable[str]) -> tuple[list, int]:
    jobs, skipped = [], 0
    for row in rows:
        supported = BRANCHES if canonical_call(row, modality) is not None else ("direct", "natural")
        available = [branch for branch in branches if branch in supported]
        skipped += len([branch for branch in branches if branch not in supported])
        for branch in available:
            jobs.append({"row": row, "branch": branch})
    return jobs, skipped


CACHE_VERSION = 1


def cache_identity(modality: str) -> dict[str, Any]:
    return {
        "cache_version": CACHE_VERSION,
        "probe_version": PROBE_VERSION,
        "modality": modality,
        "random_salt": RANDOM_SALT,
        "env": {name: os.environ.get(name, default) for name, default in BUDGET_ENV_DEFAULTS.items()},
    }


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def cached_intervention(row: dict[str, Any], modality: str, cache_root: Path) -> dict[str, Any] | None:
    if canonical_call(row, modality) is None:
        return None
    qid = str(row["qid"])
    safe = "".join(char if char.isalnum() or char in "-_." else "_" for char in qid)
    directory = cache_root / modality / safe
    meta_path = directory / "meta.json"
    identity = cache_identity(modality)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (meta.get("identity") == identity and meta.get("qid") == qid
                and all(Path(path).is_file() for paths in meta["images"].values() for path in paths)):
            return {key: meta[key] for key in ("call", "tool_response", "images", "info")}
    prepared = intervention(row, modality)
    if prepared is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    images: dict[str, list[str]] = {}
    for branch, urls in prepared["images"].items():
        paths = []
        for index, url in enumerate(urls):
            header, payload = url.split(",", 1)
            extension = "jpg" if "jpeg" in header else "png"
            path = directory / f"{branch}_{index:02d}.{extension}"
            _atomic_write(path, base64.b64decode(payload))
            paths.append(str(path))
        images[branch] = paths
    meta = {
        "qid": qid,
        "identity": identity,
        "call": prepared["call"],
        "tool_response": prepared["tool_response"],
        "images": images,
        "info": prepared["info"],
    }
    _atomic_write(meta_path, (json.dumps(meta, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return {key: meta[key] for key in ("call", "tool_response", "images", "info")}


def attach_interventions(
    batch: list[dict[str, Any]], modality: str, cache_root: Path | None = None
) -> None:
    cache: dict[str, Any] = {}
    for job in batch:
        qid = str(job["row"]["qid"])
        if qid not in cache:
            if cache_root is not None:
                cache[qid] = cached_intervention(job["row"], modality, cache_root)
            else:
                cache[qid] = intervention(job["row"], modality) if canonical_call(job["row"], modality) else None
        job["prepared"] = cache[qid]


def _precompute_one(task: tuple[dict[str, Any], str, str]) -> tuple[str, bool]:
    row, modality, cache_root = task
    return str(row["qid"]), cached_intervention(row, modality, Path(cache_root)) is not None


def run_precompute(args: argparse.Namespace) -> None:
    import multiprocessing

    rows, _ = load_rows(args.data, args.max_rows)
    tasks = [(row, args.modality, str(args.intervention_cache)) for row in rows]
    start, built = time.time(), 0
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for count, (_, ok) in enumerate(pool.imap_unordered(_precompute_one, tasks, chunksize=4), 1):
            built += ok
            if count % 200 == 0 or count == len(tasks):
                print(f"precompute {count}/{len(tasks)} rows ({built} with interventions) "
                      f"{time.time() - start:.0f}s", flush=True)
    print(json.dumps({"rows": len(tasks), "with_intervention": built,
                      "identity": cache_identity(args.modality)}, indent=1), flush=True)


def make_runner(
    modality: str,
    request_config: dict[str, Any],
    intervention_tools: str,
    post: Callable[..., Any] = requests.post,
    retry_sleep: float = 10.0,
) -> Callable[[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    def run(endpoint: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        infer_requests = [
            build_request(job["row"], modality, job["branch"], job.get("prepared"), intervention_tools)
            for job in batch
        ]
        response = None
        for attempt in range(3):
            try:
                response = post(
                    endpoint,
                    json={"infer_requests": infer_requests, "request_config": request_config},
                    timeout=86400,
                )
                break
            except requests.RequestException as exc:
                if attempt == 2:
                    raise RuntimeError(f"Rollout request to {endpoint} failed three times: {exc}") from exc
                time.sleep(retry_sleep)
        if response.status_code >= 500:
            if len(batch) > 1:
                half = len(batch) // 2
                return run(endpoint, batch[:half]) + run(endpoint, batch[half:])
            job = batch[0]
            print(f"server error for qid={job['row']['qid']} branch={job['branch']}: saved as error", flush=True)
            return [error_record(job, modality, f"HTTP {response.status_code}: {response.text}")]
        response.raise_for_status()
        outputs = response.json()
        if len(outputs) != len(batch):
            raise RuntimeError(f"Server returned {len(outputs)} outputs for {len(batch)} requests")
        return [build_record(job, output, modality) for job, output in zip(batch, outputs)]

    return run


def run_rollout(args: argparse.Namespace) -> None:
    rows, data_rows = load_rows(args.data, args.max_rows)
    check_manifest(args.output_dir, manifest_for(args, data_rows))
    records_path = args.output_dir / "records.jsonl"
    records, completed = load_records(records_path, args.retry_errors)
    jobs, skipped = plan_jobs(rows, args.modality, args.branches)
    if skipped:
        print(f"{skipped} (row, branch) slots skipped for rows without a canonical tool plan", flush=True)
    pending = [job for job in jobs if (str(job["row"]["qid"]), job["branch"]) not in completed]
    print(f"{len(jobs)} jobs planned, {len(jobs) - len(pending)} already recorded, {len(pending)} pending", flush=True)
    request_config = {"max_tokens": args.max_tokens, **GREEDY}
    runner = make_runner(args.modality, request_config, args.intervention_tools)
    batches: queue.Queue = queue.Queue()
    for start in range(0, len(pending), args.batch_size):
        batches.put(pending[start:start + args.batch_size])
    lock = threading.Lock()
    progress = {"done": 0, "errors": 0}
    endpoints = [f"http://{args.host}:{port}/infer/" for port in args.ports]

    def take() -> list[dict[str, Any]] | None:
        try:
            batch = batches.get_nowait()
        except queue.Empty:
            return None
        attach_interventions(batch, args.modality, args.intervention_cache)
        return batch

    def serve(endpoint: str) -> None:
        with ThreadPoolExecutor(max_workers=1) as prefetch:
            upcoming = prefetch.submit(take)
            while True:
                batch = upcoming.result()
                if batch is None:
                    return
                upcoming = prefetch.submit(take)
                process(endpoint, batch)

    def process(endpoint: str, batch: list[dict[str, Any]]) -> None:
        new_records = runner(endpoint, batch)
        for job in batch:
            job.pop("prepared", None)
        lines = [json.dumps(record, ensure_ascii=False) + "\n" for record in new_records]
        with lock:
            writer.writelines(lines)
            writer.flush()
            records.extend(new_records)
            progress["done"] += len(batch)
            progress["errors"] += sum(bool(record.get("error")) for record in new_records)
            print(f"completed {progress['done']}/{len(pending)} (server errors {progress['errors']})", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with records_path.open("a", encoding="utf-8") as writer:
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            for future in [pool.submit(serve, endpoint) for endpoint in endpoints]:
                future.result()

    records, _ = load_records(records_path)
    write_summary(args, records, "correctness")


def mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def bootstrap(values: list[float], samples: int, seed: int = 42) -> dict[str, Any]:
    if not values:
        return {"mean": None, "ci95": [None, None], "n_queries": 0}
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    estimates = []
    chunk = max(1, min(samples, 2_000_000 // max(1, len(array))))
    remaining = samples
    while remaining > 0:
        size = min(chunk, remaining)
        estimates.append(array[rng.integers(0, len(array), size=(size, len(array)))].mean(axis=1))
        remaining -= size
    estimates = np.concatenate(estimates)
    return {
        "mean": float(array.mean()),
        "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "n_queries": int(len(array)),
    }


def rollout_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {"records": len(records), "by_branch": {}}
    for branch in BRANCHES:
        subset = [record for record in records if record["branch"] == branch]
        if not subset:
            continue
        ok = [record for record in subset if not record.get("error")]
        entry = {"records": len(subset), "server_errors": len(subset) - len(ok)}
        if branch == "natural":
            entry["call_rate"] = mean([float(bool(record.get("natural_called_tool"))) for record in ok])
            entry["successful_call_rate"] = mean([float(bool(record.get("natural_successful_tool"))) for record in ok])
        if branch in INTERVENTIONS:
            entry["continuation_tool_call_rate"] = mean([float(bool(record.get("continuation_tool_call"))) for record in ok])
        entry["valid_output_rate"] = mean([float(bool(record.get("valid_output"))) for record in ok])
        stats["by_branch"][branch] = entry
    return stats


def summarize(
    records: list[dict[str, Any]],
    value_key: str,
    bootstrap_samples: int = 10000,
    branches: Iterable[str] = BRANCHES,
) -> dict[str, Any]:
    branches = list(branches)
    usable = [record for record in records if not record.get("error") and record.get(value_key) is not None]
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in usable:
        grouped[str(record["qid"])][record["branch"]].append(float(record[value_key]))
    qmeans = {qid: {branch: mean(values) for branch, values in by_branch.items()} for qid, by_branch in grouped.items()}

    branch_accuracy = {}
    for branch in branches:
        values = [scores[branch] for scores in qmeans.values() if branch in scores]
        result = bootstrap(values, bootstrap_samples)
        branch_accuracy[branch] = {
            "accuracy": result["mean"],
            "ci95": result["ci95"],
            "n_queries": result["n_queries"],
            "n_rollouts": sum(record["branch"] == branch for record in usable),
            "n_server_errors": sum(record["branch"] == branch and bool(record.get("error")) for record in records),
        }

    paired = {}
    for left, right in PAIRED_EFFECTS:
        values = [
            scores[left] - scores[right]
            for scores in qmeans.values()
            if left in scores and right in scores
        ]
        result = bootstrap(values, bootstrap_samples)
        result["wins_ties_losses"] = {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }
        paired[f"{left}_minus_{right}"] = result

    natural = [record for record in usable if record["branch"] == "natural"]
    called = [record for record in natural if record.get("natural_called_tool")]
    not_called = [record for record in natural if not record.get("natural_called_tool")]
    natural_routing = {
        "call_rate": mean([float(bool(record.get("natural_called_tool"))) for record in natural]),
        "successful_call_rate": mean([float(bool(record.get("natural_successful_tool"))) for record in natural]),
        "n_rollouts": len(natural),
        "n_called": len(called),
        "accuracy_called": mean([float(record[value_key]) for record in called]),
        "accuracy_not_called": mean([float(record[value_key]) for record in not_called]),
    }
    interventions = {}
    for branch in INTERVENTIONS:
        subset = [record for record in usable if record["branch"] == branch]
        if subset:
            interventions[branch] = {
                "continuation_tool_call_rate": mean([float(bool(record.get("continuation_tool_call"))) for record in subset]),
                "valid_output_rate": mean([float(bool(record.get("valid_output"))) for record in subset]),
            }
    random_records = [record for record in usable if record["branch"] == "random" and record.get("intervention")]
    random_info = {}
    if random_records:
        per_qid = {str(record["qid"]): record["intervention"] for record in random_records}
        if "random_iou" in next(iter(per_qid.values())):
            ious = [info["random_iou"] for info in per_qid.values()]
            random_info = {
                "n_queries": len(per_qid),
                "mean_iou_to_gt": mean(ious),
                "max_iou_to_gt": max(ious),
                "fallback_queries": sum(bool(info.get("random_fallback")) for info in per_qid.values()),
            }
        else:
            random_info = {
                "n_queries": len(per_qid),
                "overlap_queries": sum(info.get("random_overlap", 0) > 0 for info in per_qid.values()),
                "overlap_frames": sum(info.get("random_overlap", 0) for info in per_qid.values()),
                "gt_subsampled_queries": sum(bool(info.get("gt_subsampled")) for info in per_qid.values()),
            }
    return {
        "value_key": value_key,
        "n_queries": len(qmeans),
        "branch_accuracy": branch_accuracy,
        "paired_effects": paired,
        "natural_routing": natural_routing,
        "interventions": interventions,
        "random_control": random_info,
        "server_errors": sum(bool(record.get("error")) for record in records),
    }


def markdown_row(name: str, accuracy: dict[str, float | None]) -> str:
    def pct(value: float | None) -> str:
        return "-" if value is None else f"{100 * value:.1f}"

    gt, random_, blank = (accuracy.get(b) for b in ("gt", "random", "blank"))
    dependence = observation_dependence(gt, random_, blank)
    return "| " + " | ".join([name, *(pct(accuracy.get(b)) for b in BRANCHES), pct(dependence)]) + " |"


MARKDOWN_HEADER = (
    "| ckpt | direct | natural | gt | random | blank | dependence |\n"
    "|---|---|---|---|---|---|---|"
)


def write_summary(args: argparse.Namespace, records: list[dict[str, Any]], value_key: str,
                  extra: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = summarize(records, value_key, args.bootstrap_samples, args.branches)
    summary["modality"] = args.modality
    summary["rollout"] = rollout_stats(records)
    if extra:
        summary.update(extra)
    name = args.name or args.output_dir.name
    branch = summary.get("branch_accuracy", {})
    summary["markdown_row"] = markdown_row(name, {b: (branch.get(b) or {}).get("accuracy") for b in BRANCHES})
    path = args.output_dir / "summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print(MARKDOWN_HEADER)
    print(summary["markdown_row"], flush=True)
    print(f"Full summary: {path}", flush=True)
    return summary












def run_summarize(args: argparse.Namespace) -> None:
    records, _ = load_records(args.output_dir / "records.jsonl")
    write_summary(args, records, "correctness")


def describe_request(request: dict[str, Any]) -> dict[str, Any]:
    def image_desc(value: Any) -> str:
        if isinstance(value, str) and value.startswith("data:"):
            width, height = data_url_size(value)
            return f"data:{width}x{height}"
        return Path(str(value)).name

    return {
        "roles": [message["role"] for message in request["messages"]],
        "placeholders": {
            "image": sum(message["content"].count("<image>") for message in request["messages"]),
            "video": sum(message["content"].count("<video>") for message in request["messages"]),
        },
        "images": [image_desc(value) for value in request.get("images", [])],
        "videos": [len(video) for video in request.get("videos", [])],
        "tools": [tool["function"]["name"] for tool in request.get("tools", [])],
        "tools_enabled": request["data_dict"]["tools_enabled"],
        "tool_video_frames": "tool_video_frames" in request["data_dict"],
        "request_bytes": len(json.dumps(request)),
    }


def run_dry_build(args: argparse.Namespace) -> None:
    rows, _ = load_rows(args.data, args.dry_build_rows)
    for row in rows:
        prepared = intervention(row, args.modality) if canonical_call(row, args.modality) else None
        print(f"=== qid={row['qid']}")
        if prepared is not None:
            print("intervention:", json.dumps(prepared["info"]))
            print("assistant tool call:", _format_tool_call(prepared["call"]).replace("\n", "\\n"))
            print("tool response:", prepared["tool_response"].replace("\n", "\\n")[:600])
        texts = {}
        for branch in args.branches:
            if branch in INTERVENTIONS and prepared is None:
                continue
            request = build_request(row, args.modality, branch, prepared, args.intervention_tools)
            texts[branch] = json.dumps(request["messages"], ensure_ascii=False)
            print(f"  {branch:7s}", json.dumps(describe_request(request)))
        intervention_texts = {texts[b] for b in INTERVENTIONS if b in texts}
        if intervention_texts:
            print("  gt/random/blank identical message text:", len(intervention_texts) == 1)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    {
        "rollout": run_rollout,
        "summarize": run_summarize,
        "dry_build": run_dry_build,
        "precompute": run_precompute,
    }[args.stage](args)


if __name__ == "__main__":
    main()
