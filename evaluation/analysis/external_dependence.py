#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import importlib.metadata
import json
import os
import random
import re
import socket
import sys
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq
import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "ms-swift")]

from evaluation.analysis.report import observation_dependence  # noqa: E402
from evaluation.inference import tools as native  # noqa: E402

from evaluation.inference import service
from evaluation.inference.service import (
    digest, endpoint_for, file_digest, infer_controlled_batch,
    model_identity, observed_engine_identity, output_lock,
    process_start_ticks, swift_model_name, unpack_single_turn,
)

VERSION = 6
SERVICE_HEADER = "X-External-Visdep-Instance"
IDENTITY_HEADER = "X-External-Visdep-Identity"
SWIFT_ROOT = REPO_ROOT / "ms-swift" / "swift"
PAIRS = (("gt", "natural"), ("natural", "random"), ("natural", "blank"))
SAMPLING = {"max_tokens": 8192, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "n": 1,
            "stop": ["</tool_call>"]}


def sampling_for(protocol: str) -> dict[str, Any]:
    config = copy.deepcopy(SAMPLING)
    if protocol == "thyme":
        config["stop"] = ["</code>", "</answer>", "</tool_call>"]
    elif protocol == "deepeyesv2":
        config["stop"] = ["</code>", "<|im_end|>", "</tool_call>"]
    return config






def image_values(row: dict[str, Any]) -> list[Any]:
    value = row.get("image", row.get("images"))
    return list(value) if isinstance(value, (list, tuple)) else [value]


def load_image(row: dict[str, Any]) -> Image.Image:
    values = image_values(row)
    if len(values) != 1 or values[0] is None or row.get("is_video"):
        raise ValueError("This probe requires exactly one original image and no video")
    value = values[0]
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            with Image.open(io.BytesIO(value["bytes"])) as image:
                return image.convert("RGB")
        value = value.get("path")
    if not isinstance(value, str) or "://" in value or value.startswith("data:"):
        raise ValueError("Only trusted local data images (or embedded bytes) are accepted")
    with Image.open(value) as image:
        return image.convert("RGB")


def canonical_bbox(row: dict[str, Any]) -> list[float]:
    metadata = row.get("tool_metadata") or {}
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    plans = metadata.get("canonical_tool_plan") or []
    if len(plans) != 1:
        raise ValueError("Exactly one canonical GT crop is required; no row is silently skipped")
    plan = plans[0]
    arguments = plan.get("arguments") or {}
    bbox = arguments.get("bbox_2d")
    if plan.get("name") != "crop" or type(arguments.get("target_image")) is not int or arguments["target_image"] != 1:
        raise ValueError("GT plan must crop the only original image")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or any(type(v) not in (int, float) or not 0 <= v <= 1000 for v in bbox):
        raise ValueError("GT bbox must contain four finite 0-1000 coordinates")
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        raise ValueError("Empty GT bbox")
    return list(bbox)


def box_iou(a: list[float], b: list[float]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0.0


def qid_seed(qid: str) -> int:
    return int(hashlib.sha256(f"{native.RANDOM_SALT}:{qid}".encode()).hexdigest()[:16], 16)


def random_crop_origin(qid: str, image_size: tuple[int, int], gt_pixels: list[int]) -> dict[str, Any]:
    width, height = image_size
    cw, ch = gt_pixels[2]-gt_pixels[0], gt_pixels[3]-gt_pixels[1]
    rng = random.Random(qid_seed(qid))
    def iou(origin):
        x, y = origin
        return box_iou([x, y, x+cw, y+ch], gt_pixels)
    draws = []
    for draw in range(1, native.RANDOM_MAX_DRAWS+1):
        origin = (rng.randint(0, width-cw), rng.randint(0, height-ch))
        value = iou(origin)
        if value <= native.RANDOM_MAX_IOU:
            return {"origin": origin, "iou": value, "draws": draw, "fallback": False}
        draws.append(origin)
    corners = [(0, 0), (width-cw, 0), (0, height-ch), (width-cw, height-ch)]
    origin = min(corners+draws, key=iou)
    return {"origin": origin, "iou": iou(origin), "draws": native.RANDOM_MAX_DRAWS, "fallback": True}


def random_diagnostics(infos: list[dict[str, Any]]) -> dict[str, Any]:
    values = sorted(info["random_iou"] for info in infos)
    return {"qid_count": len(values), "target_max_iou": native.RANDOM_MAX_IOU,
            "max_draws": native.RANDOM_MAX_DRAWS, "salt": native.RANDOM_SALT,
            "fallback_qids": sum(info["random_fallback"] for info in infos),
            "above_target_qids": sum(value > native.RANDOM_MAX_IOU for value in values),
            "iou_min": values[0] if values else None, "iou_max": values[-1] if values else None,
            "iou_mean": sum(values)/len(values) if values else None,
            "iou_quantiles": {str(q): values[round((len(values)-1)*q)] if values else None for q in (.25, .5, .75, .9, .95)},
            "interpretation": "same-size target <=0.1 with minimum sampled/corner IoU fallback; above-target rows retained"}


def prepare_interventions(row: dict[str, Any], protocol: str, image: Image.Image,
                          crop_tokens: int = 1960) -> dict[str, Any]:
    call = native.native_call(protocol, image, canonical_bbox(row))
    gt, actual_box = native.execute_native(protocol, [image], call)
    rounded_box = [round(v) for v in actual_box]
    width, height = gt.size
    placement = random_crop_origin(str(row["qid"]), image.size, rounded_box)
    left, top = placement["origin"]
    random_box = [left, top, left + width, top + height]
    crops = {"gt": gt, "random": image.crop(tuple(random_box)), "blank": Image.new("RGB", gt.size, native.GRAY)}
    return {
        "assistant": native.forced_assistant(protocol, call),
        "observation": native.observation(protocol, gt, stdout=native.RESULT_ALIAS + "\n" if protocol == "thyme" else ""),
        "images": {branch: native.data_url(native.budget_crop(crop, crop_tokens)) for branch, crop in crops.items()},
        "info": {"native_call": call, "gt_source_box": actual_box, "gt_rounded_box": rounded_box,
                 "random_source_box": random_box, "random_iou": box_iou(random_box, rounded_box),
                 "native_size": list(gt.size), "returned_size": list(native.budget_crop(gt, crop_tokens).size),
                 "random_seed": qid_seed(str(row["qid"])), "random_draws": placement["draws"],
                 "random_fallback": placement["fallback"], "random_above_target": placement["iou"] > native.RANDOM_MAX_IOU},
    }


def build_request(row: dict[str, Any], protocol: str, branch: str,
                  prepared: dict[str, Any] | None = None) -> dict[str, Any]:
    native.require_supported(protocol)
    if branch not in native.BRANCHES:
        raise ValueError(branch)
    tools = branch != "direct"
    request = {"messages": [
        {"role": "system", "content": native.system_prompt(protocol, tools)},
        {"role": "user", "content": native.user_prompt(str(row["question"]), protocol, tools,
            size=load_image(row).size if protocol == "thyme" else None)},
    ], "images": image_values(row), "tools": []}
    if branch in native.INTERVENTIONS:
        if prepared is None:
            raise ValueError("Missing intervention")
        native.append_assistant(request, protocol, prepared["assistant"])
        native.append_observation(request, protocol, native.single_call_observation(protocol, prepared["observation"]))
        request["images"] = request["images"] + [prepared["images"][branch]]
    return request


def answer_letter(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})*)\}", text or "")
    tagged = re.findall(r"<answer>(.*?)</answer>", text or "", flags=re.S | re.I)
    extracted = boxed[-1] if boxed else tagged[-1] if tagged else ""
    extracted = re.sub(r"\\text\s*\{([^{}]*)\}", r"\1", extracted).strip()
    for pattern in (r"([A-Za-z])\.?", r"[\(\[]\s*([A-Za-z])\s*[\)\]](?:\s.*)?",
                    r"([A-Za-z])\s*[.):](?:\s.*)?", r"(?:option|choice)\s*:?\s*[\(\[]?\s*([A-Za-z])\s*[\)\]]?\.?"):
        match = re.fullmatch(pattern, extracted, flags=re.S | re.I)
        if match:
            return match.group(1).upper()
    return None


def reference_letter(row: dict[str, Any]) -> str:
    target = row["answer"]
    if isinstance(target, (list, tuple)):
        target = target[0] if target else ""
    text = str(target).strip()
    letter = answer_letter(text) or (text.upper() if re.fullmatch(r"[A-Za-z]", text) else None)
    if letter is None:
        raise ValueError(f"Non-MCQ reference for qid={row['qid']}")
    return letter


def run_job(row: dict[str, Any], protocol: str, branch: str,
            infer: Callable[[dict[str, Any]], tuple[str, str | None]], crop_tokens: int = 1960,
            prepared: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {"qid": str(row["qid"]), "branch": branch,
              "status": "ok", "correctness": None, "calls": [], "invalid": False,
              "unsupported": False, "re_call": False, "natural_called_tool": False,
              "messages": [], "final_answer": "", "error": None}
    try:
        native.require_supported(protocol)
        try:
            target = reference_letter(row)
            image = load_image(row)
            if branch in native.INTERVENTIONS and prepared is None:
                prepared = prepare_interventions(row, protocol, image, crop_tokens)
        except (ValueError, OSError) as exc:
            record.update(status="data_unavailable", error=f"{type(exc).__name__}: {exc}")
            return record
        request = build_request(row, protocol, branch, prepared)
        if prepared is not None and branch in native.INTERVENTIONS:
            record["intervention"] = prepared["info"]
        current_images = [image]
        interpreter = native.ImageInterpreter(image, protocol) if protocol in native.CODE_PROTOCOLS else None
        tools_enabled = branch == "natural"
        operations = 0
        for turn in range(native.MAX_TURNS if tools_enabled else 1):
            text, finish_reason = infer(copy.deepcopy(request))
            record["final_answer"] = text
            # No code/tool execution in any direct or intervention continuation,
            # even when an apparent correct final answer accompanies the re-call.
            if not tools_enabled:
                record["re_call"] = native.has_operation(text)
                record["invalid"] = record["re_call"] or finish_reason == "length" or answer_letter(text) is None
                native.append_assistant(request, protocol, text)
                break
            attempted = native.has_operation(text)
            record["natural_called_tool"] |= attempted
            try:
                bounded, call = native.bounded_action(text, protocol)
            except (ValueError, native.UnsupportedOperation) as exc:
                record["unsupported"] |= isinstance(exc, native.UnsupportedOperation)
                record["invalid"] = True
                bounded, call = text, None
                if not attempted:
                    raise
                native.append_assistant(request, protocol, bounded)
                if operations >= native.MAX_OPERATIONS or turn == native.MAX_TURNS - 1:
                    record["re_call"] = True
                    break
                operations += 1
                record["calls"].append({"success": False, "error": str(exc), "unsupported": isinstance(exc, native.UnsupportedOperation)})
                native.append_observation(request, protocol, native.observation(protocol, None, error=str(exc), count=0))
                continue
            native.append_assistant(request, protocol, bounded)
            record["final_answer"] = bounded
            if call is None:
                record["invalid"] |= finish_reason == "length" or answer_letter(bounded) is None
                break
            if operations >= native.MAX_OPERATIONS or turn == native.MAX_TURNS - 1:
                record["invalid"] = record["re_call"] = True
                break
            operations += 1
            trace = {"call": call, "success": False}
            try:
                if protocol in native.CODE_PROTOCOLS:
                    result = native.execute_code(protocol, interpreter, call)
                    frames, stdout = result.frames, result.stdout
                else:
                    crop, box = native.execute_native(protocol, current_images, call)
                    frames, stdout = [native.Frame(crop, tuple(box))], ""
                per_image_tokens = max(1, crop_tokens // max(1, len(frames)))
                if len(frames) > crop_tokens:
                    raise native.UnsupportedOperation("Too many outputs for crop token budget")
                returned = [native.budget_crop(frame.image, per_image_tokens) for frame in frames]
                observation = native.observation(protocol, frames[0].image if frames else None, stdout=stdout, count=len(frames))
                current_images.extend(frame.image for frame in frames)
                request["images"].extend(native.data_url(value) for value in returned)
                trace.update(success=True, stdout=stdout, image_count=len(frames),
                             source_boxes=[list(frame.box) for frame in frames],
                             native_sizes=[list(frame.image.size) for frame in frames],
                             returned_sizes=[list(value.size) for value in returned])
                if len(frames) == 1:
                    trace.update(source_box=list(frames[0].box), native_size=list(frames[0].image.size), returned_size=list(returned[0].size))
            except (ValueError, native.UnsupportedOperation) as exc:
                observation = native.observation(protocol, None, error=str(exc), count=0)
                trace.update(error=str(exc), unsupported=isinstance(exc, native.UnsupportedOperation))
                record["unsupported"] |= trace["unsupported"]
                record["invalid"] = True
            record["calls"].append(trace)
            native.append_observation(request, protocol, observation)
        record["messages"] = request["messages"]
        record["correctness"] = float(not record["re_call"] and finish_reason != "length" and answer_letter(record["final_answer"]) == target)
    except native.ProtocolBlocked as exc:
        record.update(status="blocked", error=str(exc))
    except Exception as exc:
        record.update(status="infra_error", error=f"{type(exc).__name__}: {exc}")
    return record


def plan_jobs(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], str]]:
    return [(row, branch) for row in rows for branch in native.BRANCHES]


def key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["qid"]), record["branch"]


def summarize(rows: list[dict[str, Any]], records: list[dict[str, Any]],
              protocol: str, total_rows: int | None = None) -> dict[str, Any]:
    expected = {(str(row["qid"]), branch) for row, branch in plan_jobs(rows)}
    by_key = {}
    for record in records:
        if key(record) not in expected:
            raise ValueError(f"Record outside requested experiment: {key(record)}")
        by_key[key(record)] = record
    groups = {}
    means = {}
    for branch in native.BRANCHES:
        current = [record for record in by_key.values() if record["branch"] == branch]
        scored = [record for record in current if record["status"] == "ok"]
        qid_values = defaultdict(list)
        for record in scored:
            qid_values[record["qid"]].append(record["correctness"])
        observed = {qid: sum(values) / len(values) for qid, values in qid_values.items()}
        means[branch] = observed
        expected_count = len(rows)
        rate = lambda field: sum(bool(record.get(field)) for record in scored) / len(scored) if scored else None
        complete = len(scored) == expected_count
        groups[branch] = {
            "expected": expected_count, "recorded": len(current), "scored": len(scored),
            "missing": expected_count - len(current),
            "infra_errors": sum(record["status"] == "infra_error" for record in current),
            "data_unavailable": sum(record["status"] == "data_unavailable" for record in current),
            "blocked": sum(record["status"] == "blocked" for record in current),
            "coverage": len(scored) / expected_count if expected_count else 0,
            "accuracy": sum(observed.values()) / len(observed) if complete and observed else None,
            "accuracy_observed_qid_mean": sum(observed.values()) / len(observed) if observed else None,
            "complete_qids": len(means[branch]),
            "invalid_rate": rate("invalid"), "unsupported_rate": rate("unsupported"),
            "re_call_rate": rate("re_call"),
            "natural_call_rate": rate("natural_called_tool") if branch == "natural" else None,
            "diagnostic_rate_denominator": len(scored),
        }
    paired = {}
    for left, right in PAIRS:
        shared = sorted(means[left].keys() & means[right].keys())
        paired[f"{left}-{right}"] = {
            "qid_count": len(shared), "expected_qids": len(rows),
            "difference": sum(means[left][qid] - means[right][qid] for qid in shared) / len(shared) if shared else None,
            "complete": len(shared) == len(rows),
        }
    complete_requested = all(group["scored"] == group["expected"] for group in groups.values()) and bool(rows) and protocol not in native.BLOCKED
    infos = {record["qid"]: record["intervention"] for record in by_key.values() if record.get("intervention")}
    return {"protocol": native.protocol_status(protocol), "groups": groups, "paired_effects": paired,
            "random_control": random_diagnostics(list(infos.values())),
            "expected_records": len(expected), "recorded": len(by_key),
            "complete_requested": complete_requested,
            "complete_experiment": complete_requested and len(rows) == 191 and total_rows == 191,
            "accuracy_unit": "mean over qid of correctness; infra is unscored, never zero",
            "scope": "image-only controlled crop-budget probe; not full native Python/search environments"}


def load_rows(path: Path, limit: int | None) -> tuple[list[dict[str, Any]], int]:
    all_rows = pq.read_table(path).to_pylist()
    qids = [str(row["qid"]) for row in all_rows]
    if len(set(qids)) != len(qids):
        raise ValueError("Duplicate qids in data")
    return (all_rows[:limit] if limit else all_rows), len(all_rows)




def source_identity() -> dict[str, Any]:
    tree = {str(path.relative_to(SWIFT_ROOT)): file_digest(path) for path in sorted(SWIFT_ROOT.rglob('*.py'))}
    if not tree or 'infer_engine/grpo_vllm_engine.py' not in tree:
        raise ValueError('Checked-out Swift sources unavailable')
    return {'swift_root': str(SWIFT_ROOT.resolve()), 'swift_py_count': len(tree), 'swift_tree_sha256': digest(tree),
            'client': file_digest(Path(__file__)), 'protocol': file_digest(Path(native.__file__)),
            'service': file_digest(Path(service.__file__)),
            'launcher': file_digest(REPO_ROOT / 'evaluation/launchers/external_dependence.sh')}






def service_contract(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = model_identity(args.model)
    if not checkpoint or not checkpoint.get('configs', {}).get('config.json') or not checkpoint.get('shard_size_mtime_ns'):
        raise ValueError('Controlled service requires a local checkpoint with config.json and weight shards')
    packages = {}
    for package in ('ms-swift', 'torch', 'vllm', 'transformers', 'Pillow', 'fastapi', 'uvicorn'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {'schema': 1, 'endpoint': endpoint_for(args.host, args.port), 'protocol': args.protocol,
            'checkpoint': checkpoint, 'sources': source_identity(),
            'python': {'executable': str(Path(sys.executable).resolve()), 'version': sys.version}, 'packages': packages,
            'engine_class': 'swift.infer_engine.grpo_vllm_engine.GRPOVllmEngine', 'model_type': 'qwen2_5_vl',
            'template_type': 'qwen2_5_vl', 'template_backend': 'swift', 'response_prefix': None,
            'torch_dtype': 'bfloat16', 'single_turn': True, 'scheduler': None, 'gym': False,
            'tensor_parallel_size': args.tensor_parallel_size, 'data_parallel_size': args.data_parallel_size,
            'max_model_len': args.max_model_len, 'gpu_memory_utilization': args.gpu_memory_utilization,
            'processor': {name: os.environ.get(name, default) for name, default in (('MIN_PIXELS', '3136'), ('MAX_PIXELS', '4014080'))}}




def validate_local_listener(identity: dict[str, Any], expected_pid: int | None = None) -> None:
    pid = identity['pid']
    if type(pid) is not int or (expected_pid is not None and pid != expected_pid):
        raise ValueError('Service PID differs from the process started by this launcher')
    proc = Path(f'/proc/{pid}')
    if proc.stat().st_uid != os.getuid() or process_start_ticks(pid) != identity['process_start_ticks']:
        raise ValueError('Stale or foreign service process identity')
    command = (proc / 'cmdline').read_bytes().split(b'\0')
    decoded = [part.decode(errors='replace') for part in command if part]
    stage_positions = [index for index, part in enumerate(decoded[:-1]) if part == '--stage']
    if len(decoded) < 2 or Path(decoded[1]).resolve() != Path(__file__).resolve() or not stage_positions or decoded[stage_positions[-1]+1] != 'serve':
        raise ValueError('Listener is not the controlled probe serve process')
    if str((proc / 'exe').resolve()) != identity['contract']['python']['executable']:
        raise ValueError('Service Python executable mismatch')
    socket_inodes = set()
    for entry in (proc / 'fd').iterdir():
        try:
            target = os.readlink(entry)
        except FileNotFoundError:
            continue
        if target.startswith('socket:['):
            socket_inodes.add(target[8:-1])
    port = int(identity['contract']['endpoint'].rsplit(':', 1)[1])
    expected_local = f'0100007F:{port:04X}'
    entries = (proc / 'net/tcp').read_text().splitlines()[1:]
    if not any(parts[1] == expected_local and parts[3] == '0A' and parts[9] in socket_inodes
               for parts in (line.split() for line in entries)):
        raise ValueError('Requested endpoint listener is not owned by the identified service PID')








def serve_controlled(args: argparse.Namespace) -> int:
    if args.service_identity is None:
        raise ValueError('--service-identity is required for controlled serve')
    contract = service_contract(args)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(('127.0.0.1', args.port))
    listener.listen(128)
    sys.path.insert(0, str(SWIFT_ROOT.parent))
    import swift
    if Path(swift.__file__).resolve().parent != SWIFT_ROOT.resolve():
        raise ValueError('Imported Swift is not the fingerprinted checkout')
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request
    globals()['Request'] = Request
    from fastapi.responses import JSONResponse
    import uvicorn
    from swift.arguments import RolloutArguments
    from swift.infer_engine.protocol import RequestConfig, RolloutInferRequest
    from swift.pipelines.infer.rollout import SwiftRolloutDeploy, safe_set_start_method
    safe_set_start_method()
    swift_args = RolloutArguments(model=args.model, model_type='qwen2_5_vl', template='qwen2_5_vl',
                                 template_backend='swift', response_prefix=None, torch_dtype='bfloat16',
                                 vllm_tensor_parallel_size=args.tensor_parallel_size,
                                 vllm_data_parallel_size=args.data_parallel_size, vllm_max_model_len=args.max_model_len,
                                 vllm_gpu_memory_utilization=args.gpu_memory_utilization,
                                 vllm_use_async_engine=True, multi_turn_scheduler=None, max_turns=1, use_gym_env=False)
    state = {}
    @asynccontextmanager
    async def lifespan(app):
        engine = SwiftRolloutDeploy.get_infer_engine(swift_args, template=swift_args.get_template())
        runtime = observed_engine_identity(engine, contract, swift_args)
        if service_contract(args) != contract:
            raise ValueError('Checkpoint/source/runtime changed while the engine initialized')
        identity = {'contract': contract, 'runtime': runtime, 'instance': uuid.uuid4().hex,
                    'pid': os.getpid(), 'process_start_ticks': process_start_ticks(os.getpid())}
        validate_local_listener(identity, os.getpid())
        state.update(engine=engine, identity=identity, identity_sha256=digest(identity))
        args.service_identity.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.service_identity.with_name(args.service_identity.name + f'.{os.getpid()}.partial')
        with temporary.open('x', encoding='utf-8') as output:
            json.dump(identity, output, sort_keys=True, indent=2)
            output.flush(); os.fsync(output.fileno())
        os.replace(temporary, args.service_identity)
        yield
    app = FastAPI(lifespan=lifespan)
    def response(payload, status=200):
        return JSONResponse(payload, status_code=status, headers={SERVICE_HEADER: state['identity']['instance'], IDENTITY_HEADER: state['identity_sha256']})
    @app.get('/health/')
    async def health():
        return response({'ready': True})
    @app.post('/external_visdep_identity/')
    async def identity_route(request: Request):
        body = await request.json()
        return response({'challenge': body.get('challenge'), 'identity': state['identity']})
    @app.post('/get_engine_type/')
    async def engine_type():
        return response({'engine_type': 'GRPOVllmEngine', 'enable_multi_turn': False, 'use_gym_env': False})
    @app.post('/infer/')
    async def infer_route(request: Request):
        if request.headers.get(SERVICE_HEADER) != state['identity']['instance'] or request.headers.get(IDENTITY_HEADER) != state['identity_sha256']:
            return response({'error': 'Service identity headers required'}, 409)
        body = await request.json()
        if body.get('request_config') != sampling_for(args.protocol) or len(body.get('infer_requests', [])) != 1:
            return response({'error': 'Only the fixed single-request sampling protocol is accepted'}, 400)
        output = await infer_controlled_batch(state['engine'], [RolloutInferRequest(**body['infer_requests'][0])], RequestConfig(**body['request_config']))
        return response(output)
    uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=args.port, log_level='info')).run(sockets=[listener])
    return 0


def manifest_for(args: argparse.Namespace, rows: list[dict[str, Any]], total: int, service: dict[str, Any] | None = None) -> dict[str, Any]:
    media = []
    for row in rows:
        value = image_values(row)[0]
        if isinstance(value, dict):
            value = value.get("path") if not value.get("bytes") else None
        if isinstance(value, str):
            path = Path(value)
            media.append({"path": str(path.resolve()), "sha256": file_digest(path)})
    return {"version": VERSION, "protocol": native.protocol_status(args.protocol), "sampling_per_round": sampling_for(args.protocol),
            "branches": native.BRANCHES,
            "max_turns": native.MAX_TURNS, "max_operations": native.MAX_OPERATIONS,
            "crop_tokens": args.crop_tokens, "processor": {"MIN_PIXELS": os.environ.get("MIN_PIXELS", "3136"), "MAX_PIXELS": os.environ.get("MAX_PIXELS", "4014080")},
            "data_path": str(args.data.resolve()), "data_sha256": file_digest(args.data),
            "row_fingerprint": digest(rows), "data_rows": total, "selected_qids": [str(row["qid"]) for row in rows],
            "media": media, "model": model_identity(args.model), "random_salt": native.RANDOM_SALT,
            "endpoint": endpoint_for(args.host, args.port), "service": service,
            "code": source_identity()}


def result_table(name: str, summary: dict[str, Any]) -> str:
    accuracy = {b: summary["groups"][b]["accuracy"] for b in ("direct", "natural", "gt", "random", "blank")}
    gt, random_, blank = accuracy["gt"], accuracy["random"], accuracy["blank"]
    dependence = observation_dependence(gt, random_, blank)
    pct = lambda v: "-" if v is None else f"{100 * v:.1f}"  # noqa: E731
    return ("| ckpt | direct | natural | gt | random | blank | dependence |\n|---|---|---|---|---|---|---|\n"
            "| " + " | ".join([name, *(pct(v) for v in accuracy.values()), pct(dependence)]) + " |")


def ensure_manifest(output: Path, manifest: dict[str, Any], start_over: bool = True) -> None:
    manifest = json.loads(json.dumps(manifest))
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) == manifest:
        return
    if not start_over:
        raise ValueError(f"{output} holds results from different settings; rerun the rollout to replace them")
    if path.exists() or (output / "records.jsonl").exists():
        print(f"Settings differ from the previous run in {output}; deleting its results and starting over.",
              flush=True)
        for name in ("records.jsonl", "summary.json", "ROLLOUT_DONE"):
            (output / name).unlink(missing_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")




def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"Torn journal line {number} in {path}; repair before resuming") from exc
    return records




def make_infer(args: argparse.Namespace) -> Callable:
    if args.service_identity is None:
        raise ValueError('Formal rollout requires --service-identity from this controlled launcher; arbitrary endpoints are forbidden')
    identity = json.loads(args.service_identity.read_text())
    contract = service_contract(args)
    if identity.get('contract') != contract:
        raise ValueError('Service checkpoint/template/dtype/runtime/source/endpoint contract mismatch')
    runtime = identity.get('runtime', {})
    if (runtime.get('engine_class') != contract['engine_class'] or runtime.get('enable_multi_turn') is not False or
            runtime.get('multi_turn_scheduler') is not None or runtime.get('model_dir') != contract['checkpoint']['path'] or
            not isinstance(runtime.get('engine_model_dir'), str) or
            str(Path(runtime['engine_model_dir']).resolve()) != contract['checkpoint']['path'] or
            runtime.get('model_name') != swift_model_name(runtime['engine_model_dir'])):
        raise ValueError('Service runtime engine/checkpoint/single-turn identity mismatch')
    for name in ('torch_dtype', 'model_type', 'template_type', 'template_backend', 'response_prefix'):
        if runtime.get(name) != contract[name]:
            raise ValueError(f'Service runtime {name} identity mismatch')
    if not runtime.get('template_meta') or not runtime.get('template_class', '').startswith('swift.template.'):
        raise ValueError('Missing runtime template/prefix snapshot')
    engine_args = runtime.get('engine_args', {})
    if (not isinstance(engine_args.get('model'), str) or str(Path(engine_args['model']).resolve()) != runtime['model_dir'] or
            engine_args.get('dtype') not in ('bfloat16', 'torch.bfloat16')):
        raise ValueError('Service actual vLLM model/dtype identity mismatch')
    if any(engine_args.get(key) != contract[key] for key in
           ('tensor_parallel_size', 'data_parallel_size', 'max_model_len', 'gpu_memory_utilization')):
        raise ValueError('Service actual vLLM runtime configuration mismatch')
    identity_sha = digest(identity)
    root = contract['endpoint']
    headers = {SERVICE_HEADER: identity['instance'], IDENTITY_HEADER: identity_sha}
    def validate_process():
        if json.loads(args.service_identity.read_text()) != identity:
            raise ValueError('Service identity file changed; refuse stale/replaced endpoint')
        validate_local_listener(identity, args.expected_service_pid)
    def post(path, body, timeout):
        validate_process()
        response = requests.post(root + path, json=body, timeout=timeout, headers=headers,
                                 allow_redirects=False, proxies={'http': '', 'https': ''})
        response.raise_for_status()
        if response.status_code != 200 or response.url != root + path:
            raise ValueError('Service redirected or returned an unexpected endpoint/status')
        if response.headers.get(SERVICE_HEADER) != identity['instance'] or response.headers.get(IDENTITY_HEADER) != identity_sha:
            raise ValueError('Response is not from the bound controlled service instance')
        validate_process()
        return response.json()
    challenge = uuid.uuid4().hex
    reply = post('/external_visdep_identity/', {'challenge': challenge}, 30)
    if reply != {'challenge': challenge, 'identity': identity}:
        raise ValueError('Live service identity/challenge does not match its local runtime descriptor')
    info = post('/get_engine_type/', {}, 30)
    if info.get('engine_type') != 'GRPOVllmEngine' or info.get('enable_multi_turn') is not False or info.get('use_gym_env'):
        raise ValueError('A single-turn Swift server without a tool scheduler/gym is required')
    def infer(request: dict[str, Any]) -> tuple[str, str | None]:
        outputs = post('/infer/', {'infer_requests': [request], 'request_config': sampling_for(args.protocol)}, 86400)
        return unpack_single_turn(outputs, runtime['model_name'])
    infer.service_binding = {'contract': contract, 'runtime': runtime}
    infer.service_instance = identity
    return infer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dry_build", "rollout", "summarize", "serve"), default="dry_build")
    parser.add_argument("--protocol", choices=tuple(native.MODELS), required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--service-identity", type=Path)
    parser.add_argument("--expected-service-pid", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--crop-tokens", type=int, default=int(os.environ.get("REFRAME_CROP_MAX_TOKEN_NUM", "1960")))
    parser.add_argument("--retry-errors", action="store_true")
    args = parser.parse_args(argv)
    if args.workers <= 0 or args.crop_tokens <= 0 or (args.max_rows is not None and args.max_rows <= 0):
        parser.error("workers, crop tokens and max rows must be positive")
    if min(args.tensor_parallel_size, args.data_parallel_size, args.max_model_len) <= 0 or not 0 < args.gpu_memory_utilization < 1:
        parser.error('parallel sizes/model length must be positive and memory utilization between 0 and 1')
    if args.stage in ('rollout', 'summarize') and args.output_dir is None:
        parser.error('--output-dir is required')
    if args.stage != 'serve' and args.data is None:
        parser.error('--data is required')
    if args.stage in ('serve', 'rollout') and (not args.model or args.service_identity is None):
        parser.error('--model and --service-identity from the controlled launcher are required for serve/rollout')
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == 'serve':
        return serve_controlled(args)
    rows, total = load_rows(args.data, args.max_rows)
    if not rows:
        raise ValueError("Empty probe data")
    if args.stage == "dry_build":
        report = {"stage": "dry_build", "version": VERSION, "protocol": native.protocol_status(args.protocol), "data_rows": total,
                  "selected_rows": len(rows), "crop_tokens": args.crop_tokens,
                  "sampling_per_round": sampling_for(args.protocol), "expected_per_branch": len(rows),
                  "expected_per_model": len(rows) * len(native.BRANCHES),
                  "built": [], "unavailable": [], "complete_experiment": False}
        if args.protocol not in native.BLOCKED:
            for row in rows:
                try:
                    image = load_image(row)
                    reference_letter(row)
                    prepared = prepare_interventions(row, args.protocol, image, args.crop_tokens)
                except (ValueError, OSError) as exc:
                    report["unavailable"].append({"qid": str(row["qid"]), "status": "data_unavailable", "error": str(exc)})
                    continue
                branches = {branch: build_request(row, args.protocol, branch, prepared) for branch in native.BRANCHES}
                shared = [copy.deepcopy(branches[branch]) for branch in native.INTERVENTIONS]
                for request in shared:
                    request["images"][-1] = "PIXELS"
                if shared[0] != shared[1] or shared[1] != shared[2]:
                    raise AssertionError("Interventions differ outside returned pixels")
                report["built"].append({"qid": str(row["qid"]), "intervention": prepared["info"],
                    "branch_shapes": {branch: {"roles": [m["role"] for m in request["messages"]], "images": len(request["images"])} for branch, request in branches.items()},
                    "intervention_nonpixel_fields_equal": True})
        report["random_control"] = random_diagnostics([entry["intervention"] for entry in report["built"]])
        report["ready_for_requested_rows"] = args.protocol not in native.BLOCKED and not report["unavailable"]
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ready_for_requested_rows"] else 2
    with output_lock(args.output_dir):
        infer = make_infer(args) if args.stage == 'rollout' else None
        if infer is not None:
            service = infer.service_binding
        else:
            saved = json.loads((args.output_dir / 'manifest.json').read_text())
            service = saved.get('service')
            if not service or service['contract'] != service_contract(args):
                raise ValueError('Offline summary checkpoint/runtime/endpoint differs from the bound experiment')
        ensure_manifest(args.output_dir, manifest_for(args, rows, total, service), start_over=args.stage == "rollout")
        if infer is not None:
            with (args.output_dir / 'service_instances.jsonl').open('a') as audit:
                audit.write(json.dumps(infer.service_instance, sort_keys=True) + '\n')
                audit.flush(); os.fsync(audit.fileno())
        records_path = args.output_dir / "records.jsonl"
        records = load_records(records_path)
        summarize(rows, records, args.protocol, total)
        if args.stage == "rollout":
            latest = {key(record): record for record in records}
            pending = [(row, branch) for row, branch in plan_jobs(rows)
                       if (str(row["qid"]), branch) not in latest or
                       (args.retry_errors and latest[(str(row["qid"]), branch)]["status"] == "infra_error")]
            def execute(job):
                row, branch = job
                record = run_job(row, args.protocol, branch, infer, args.crop_tokens)
                record['service_instance_sha256'] = digest(infer.service_instance)
                return record
            with records_path.open("a") as journal, ThreadPoolExecutor(max_workers=args.workers) as pool:
                for record in pool.map(execute, pending):
                    journal.write(json.dumps(record, ensure_ascii=False) + "\n")
                    journal.flush()
                    os.fsync(journal.fileno())
                    records.append(record)
        summary = summarize(rows, records, args.protocol, total)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        if summary["complete_experiment"]:
            (args.output_dir / "ROLLOUT_DONE").write_text(digest(json.loads((args.output_dir / "manifest.json").read_text())) + "\n")
        print(result_table(args.protocol, summary))
        print(f"Full summary: {args.output_dir / 'summary.json'}", flush=True)
        return 0 if summary["complete_requested"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
