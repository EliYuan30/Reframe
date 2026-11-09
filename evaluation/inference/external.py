#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import functools
import importlib.metadata
import json
import os
import queue
import re
import socket
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "ms-swift")]

from evaluation.inference import tools as native  # noqa: E402
from evaluation.inference import protocols as P  # noqa: E402
from evaluation.inference import service as runtime  # noqa: E402

VERSION = 1
SERVICE_HEADER = "X-External-Onestep-Instance"
IDENTITY_HEADER = "X-External-Onestep-Identity"
LAUNCHER = REPO_ROOT / "evaluation" / "launchers/external.sh"
DEFAULT_IMAGE_DATA = Path("/path/to/your/benchmarks")
DEFAULT_VIDEO_DATA = Path("/path/to/your/video-benchmarks")
VISUALPROBE = ("visualprobe_easy", "visualprobe_medium", "visualprobe_hard")
RETRYABLE = {"connection", "timeout", "engine_error", "http_502", "http_503", "http_504"}
SAMPLE_FILE = "sample.eval_iter0_0.jsonl"
IDENTITY_TTL_S = 60.0


class ServiceError(RuntimeError):
    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def classify_engine_error(message: str) -> str:
    text = message.lower()
    if any(marker in text for marker in ("leaves no room for generation", "maximum model length",
                                         "maximum context length", "max_model_len", "is longer than")):
        return "context_overflow"
    return "engine_error"


def data_path(benchmark: str, image_root: Path, video_root: Path) -> Path:
    return (image_root if P.modality(benchmark) == "image" else video_root) / f"{benchmark}.parquet"


def load_benchmark(path: Path, limit: int | None) -> tuple[list[dict[str, Any]], int]:
    rows = pq.read_table(path).to_pylist()
    qids = [str(row["qid"]) for row in rows]
    if len(set(qids)) != len(qids):
        raise ValueError(f"Duplicate qids in {path}")
    return (rows[:limit] if limit else rows), len(rows)


def eval_dir(output_root: Path, benchmark: str, protocol: str, mode: str) -> Path:
    return output_root / "results" / protocol / benchmark / mode


@functools.lru_cache(maxsize=1)
def correctness_function() -> Callable[[str, str], float]:
    sys.path.insert(0, str(REPO_ROOT / "ms-swift"))
    from reframe.plugin import _correctness
    return _correctness


def trajectory_text(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(message.get("content", "")) for message in messages[2:])


def run_trajectory(row: dict[str, Any], benchmark: str, protocol: str, mode: str,
                   infer: Callable[[dict[str, Any], str], dict[str, Any]], timestamp_mode: str = "reframe",
                   crop_tokens: int = P.CROP_TOKENS, correctness: Callable[[str, str], float] | None = None,
                   max_pixels: int = P.MAX_PIXELS, max_turns: int = P.MAX_TURNS,
                   max_operations: int = P.MAX_OPERATIONS) -> dict[str, Any]:
    native.require_supported(protocol)
    tools = mode == "w_tool"
    gts = P.target_answer(row["answer"])
    media = P.media_for(row, benchmark)
    bank = P.ImageBank(media.sources)
    source_sizes = [bank.size(index) for index in range(len(bank))] if protocol == "thyme" else None
    visible_sizes = ([P.header_size(path) for path in media.visible]
                     if tools and protocol in ("deepeyes", "deepeyesv2") else None)
    request, meta = P.build_request(row, benchmark, protocol, mode, media, timestamp_mode, source_sizes)
    inputs = len(media.visible)
    interpreter = None
    if tools and protocol in native.CODE_PROTOCOLS:
        aliases = P.thyme_aliases(inputs, media.is_video) if protocol == "thyme" else None
        views = [P.visible_size(size, max_pixels) for size in visible_sizes] if protocol == "deepeyesv2" else None
        interpreter = P.MultiImageInterpreter(bank, protocol, aliases, views)
    turns: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    flags = {"emitted_operation": False, "unsupported": False, "invalid_call": False,
             "budget_exhausted": False, "truncated": False, "context_overflow": False}
    operations = 0
    final_text = ""
    for turn in range(max_turns if tools else 1):
        try:
            reply = infer(copy.deepcopy(request), benchmark)
        except ServiceError as exc:
            if exc.kind != "context_overflow" or turn == 0:
                raise
            flags["context_overflow"] = True
            break
        text, finish = reply["text"], reply.get("finish_reason")
        turns.append({"text": text, "finish_reason": finish, "usage": reply.get("usage")})
        flags["truncated"] |= finish == "length"
        if not tools:
            flags["emitted_operation"] = native.has_operation(text)
            native.append_assistant(request, protocol, text)
            final_text = text
            break
        try:
            bounded, call = native.bounded_action(text, protocol)
        except (ValueError, native.UnsupportedOperation) as exc:
            unsupported = isinstance(exc, native.UnsupportedOperation)
            flags["unsupported"] |= unsupported
            flags["invalid_call"] = True
            native.append_assistant(request, protocol, text)
            final_text = text
            if finish == "length":
                break
            if operations >= max_operations or turn == max_turns - 1:
                flags["budget_exhausted"] = True
                break
            operations += 1
            calls.append({"turn": turn, "success": False, "error": str(exc), "unsupported": unsupported})
            native.append_observation(request, protocol, native.observation(protocol, None, error=str(exc), count=0))
            continue
        native.append_assistant(request, protocol, bounded)
        final_text = bounded
        if call is None or finish == "length":
            break
        if operations >= max_operations or turn == max_turns - 1:
            flags["budget_exhausted"] = True
            break
        operations += 1
        trace: dict[str, Any] = {"turn": turn, "call": call, "success": False}
        try:
            frames, stdout, info = P.execute_operation(protocol, call, bank, interpreter, visible_sizes or [], inputs,
                                                       max_pixels)
            # As in Reframe: each returned image gets crop_tokens, and one call returns at most
            # MAX_CROP_BOXES images' worth; more outputs share that total.
            total = crop_tokens * P.MAX_CROP_BOXES
            if len(frames) > total:
                raise native.UnsupportedOperation("Too many outputs for crop token budget")
            per_image = crop_tokens if len(frames) <= P.MAX_CROP_BOXES else max(1, total // len(frames))
            returned = [native.budget_crop(frame.image, per_image) for frame in frames]
            observation = native.observation(protocol, frames[0].image if frames else None, stdout=stdout, count=len(frames))
            for frame in frames:
                bank.append(frame.image)
            request["images"].extend(native.data_url(value) for value in returned)
            trace.update(success=True, image_count=len(frames), stdout=stdout[:512], **info,
                         source_boxes=[[round(v, 2) for v in frame.box] for frame in frames],
                         native_sizes=[list(frame.image.size) for frame in frames],
                         returned_sizes=[list(value.size) for value in returned])
        except (ValueError, native.UnsupportedOperation) as exc:
            observation = native.observation(protocol, None, error=str(exc), count=0)
            trace.update(error=str(exc), unsupported=isinstance(exc, native.UnsupportedOperation))
            flags["unsupported"] |= trace["unsupported"]
            flags["invalid_call"] = True
        calls.append(trace)
        native.append_observation(request, protocol, observation)
    score = correctness or correctness_function()
    final = P.final_answer(final_text, gts)
    first = P.final_answer(turns[0]["text"], gts) if turns else final
    match = float(score(final["final_answer"], gts)) if final["answer_found"] else 0.0
    round0 = float(score(first["final_answer"], gts)) if first["answer_found"] else 0.0
    usage = [turn.get("usage") or {} for turn in turns]
    return {
        "qids": str(row["qid"]), "questions": str(row["question"]), "solutions": trajectory_text(request["messages"]),
        "gts": gts, "round0_correctness": round0, "round1_correctness": match, "match": match,
        "validity": float(final["answer_found"] and not flags["truncated"] and not flags["budget_exhausted"]
                          and not flags["context_overflow"]),
        "final_answer": final["final_answer"], "tool_calls": calls, "tools_enabled": tools,
        "external": {
            "version": VERSION, "protocol_version": P.PROTOCOL_VERSION, "model": protocol, "mode": mode,
            "benchmark": benchmark, "answer_found": final["answer_found"], "answer_source": final["answer_source"],
            "mcq_reference": final["mcq_reference"], "mcq_letter_parsed": final["mcq_letter_parsed"],
            "turns": len(turns), "operations": operations, "flags": flags, **meta,
            "prompt_tokens": [item.get("prompt_tokens") for item in usage],
            "completion_tokens": [item.get("completion_tokens") for item in usage],
            "raw_turns": turns,
        },
    }


def repair_journal(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.split(b"\n")
    tail = lines.pop()
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            raise ValueError(f"Damaged interior journal line {number} in {path}; repair manually") from exc
    if tail.strip():
        try:
            records.append(json.loads(tail))
            with path.open("ab") as handle:
                handle.write(b"\n")
        except ValueError:
            with path.open("r+b") as handle:
                handle.truncate(len(data) - len(tail))
    return records


class Journal:
    def __init__(self, path: Path, fsync_every: int = 32):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.fsync_every, self.pending = path, fsync_every, 0
        self.lock = threading.Lock()
        self.handle = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self.lock:
            self.handle.write(line)
            self.handle.flush()
            self.pending += 1
            if self.pending >= self.fsync_every:
                os.fsync(self.handle.fileno())
                self.pending = 0

    def close(self) -> None:
        with self.lock:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()


@functools.lru_cache(maxsize=1)
def source_identity() -> dict[str, Any]:
    root = runtime.SWIFT_ROOT
    tree = {str(path.relative_to(root)): runtime.file_digest(path) for path in sorted(root.rglob("*.py"))}
    if not tree or "infer_engine/grpo_vllm_engine.py" not in tree:
        raise ValueError("Checked-out Swift sources unavailable")
    files = {"client": Path(__file__), "protocols": Path(P.__file__), "native": Path(native.__file__),
             "service_helpers": Path(runtime.__file__), "launcher": LAUNCHER}
    files["benchmark_registry"] = REPO_ROOT / "evaluation/benchmarks.py"
    files.update({f"reframe/{path.name}": path for path in (REPO_ROOT / "reframe").glob("*.py")})
    return {"swift_root": str(root.resolve()), "swift_py_count": len(tree), "swift_tree_sha256": runtime.digest(tree),
            **{name: runtime.file_digest(path) for name, path in files.items()}}


def service_contract(args: argparse.Namespace, port: int) -> dict[str, Any]:
    checkpoint = runtime.model_identity(args.model)
    if not checkpoint or not checkpoint.get("configs", {}).get("config.json") or not checkpoint.get("shard_size_mtime_ns"):
        raise ValueError("Controlled service requires a local checkpoint with config.json and weight shards")
    packages = {}
    for package in ("ms-swift", "torch", "vllm", "transformers", "Pillow", "fastapi", "uvicorn"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {"schema": 1, "endpoint": runtime.endpoint_for(args.host, port), "protocol": args.protocol,
            "checkpoint": checkpoint, "sources": source_identity(),
            "python": {"executable": str(Path(sys.executable).resolve()), "version": sys.version}, "packages": packages,
            "engine_class": "swift.infer_engine.grpo_vllm_engine.GRPOVllmEngine", "model_type": "qwen2_5_vl",
            "template_type": "qwen2_5_vl", "template_backend": "swift", "response_prefix": None,
            "torch_dtype": "bfloat16", "single_turn": True, "scheduler": None, "gym": False,
            "tensor_parallel_size": 1, "data_parallel_size": 1, "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization, "limit_mm_per_prompt": args.limit_mm_per_prompt,
            "max_num_seqs": args.max_num_seqs, "samplings": P.allowed_samplings(args.protocol),
            "processor": processor_env(args)}


def processor_env(args: argparse.Namespace) -> dict[str, str]:
    values = {name: os.environ.get(name, default) for name, default in
              (("MIN_PIXELS", str(P.MIN_PIXELS)), ("MAX_PIXELS", str(P.MAX_PIXELS)))}
    expected = {"MIN_PIXELS": str(P.MIN_PIXELS), "MAX_PIXELS": str(getattr(args, "max_pixels", P.MAX_PIXELS))}
    if values != expected:
        raise ValueError(f"Processor environment {values} differs from the client pixel budget {expected}")
    return values


def validate_listener(identity: dict[str, Any], expected_pid: int | None = None) -> None:
    pid = identity["pid"]
    if type(pid) is not int or (expected_pid is not None and pid != expected_pid):
        raise ValueError("Service PID differs from the process started by this launcher")
    proc = Path(f"/proc/{pid}")
    if proc.stat().st_uid != os.getuid() or runtime.process_start_ticks(pid) != identity["process_start_ticks"]:
        raise ValueError("Stale or foreign service process identity")
    decoded = [part.decode(errors="replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
    stages = [index for index, part in enumerate(decoded[:-1]) if part == "--stage"]
    if len(decoded) < 2 or Path(decoded[1]).resolve() != Path(__file__).resolve() or not stages or decoded[stages[-1] + 1] != "serve":
        raise ValueError("Listener is not the controlled external one-step serve process")
    if str((proc / "exe").resolve()) != identity["contract"]["python"]["executable"]:
        raise ValueError("Service Python executable mismatch")
    inodes = set()
    for entry in (proc / "fd").iterdir():
        try:
            target = os.readlink(entry)
        except FileNotFoundError:
            continue
        if target.startswith("socket:["):
            inodes.add(target[8:-1])
    port = int(identity["contract"]["endpoint"].rsplit(":", 1)[1])
    expected = f"0100007F:{port:04X}"
    entries = (proc / "net/tcp").read_text().splitlines()[1:]
    if not any(parts[1] == expected and parts[3] == "0A" and parts[9] in inodes
               for parts in (line.split() for line in entries)):
        raise ValueError("Requested endpoint listener is not owned by the identified service PID")


def serve(args: argparse.Namespace) -> int:
    contract = service_contract(args, args.port)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", args.port))
    listener.listen(256)
    sys.path.insert(0, str(runtime.SWIFT_ROOT.parent))
    import swift
    if Path(swift.__file__).resolve().parent != runtime.SWIFT_ROOT.resolve():
        raise ValueError("Imported Swift is not the fingerprinted checkout")
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request
    globals()["Request"] = Request
    from fastapi.responses import JSONResponse
    import uvicorn
    from swift.arguments import RolloutArguments
    from swift.infer_engine.protocol import RequestConfig, RolloutInferRequest
    from swift.pipelines.infer.rollout import SwiftRolloutDeploy, safe_set_start_method
    safe_set_start_method()
    extra = {}
    if args.limit_mm_per_prompt:
        extra["vllm_limit_mm_per_prompt"] = args.limit_mm_per_prompt
    if args.max_num_seqs:
        extra["vllm_max_num_seqs"] = args.max_num_seqs
    swift_args = RolloutArguments(model=args.model, model_type="qwen2_5_vl", template="qwen2_5_vl",
                                 template_backend="swift", response_prefix=None, torch_dtype="bfloat16",
                                 vllm_tensor_parallel_size=1, vllm_data_parallel_size=1,
                                 vllm_max_model_len=args.max_model_len,
                                 vllm_gpu_memory_utilization=args.gpu_memory_utilization,
                                 vllm_use_async_engine=True, multi_turn_scheduler=None, max_turns=1,
                                 use_gym_env=False, **extra)
    allowed = P.allowed_samplings(args.protocol)
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(app):
        engine = SwiftRolloutDeploy.get_infer_engine(swift_args, template=swift_args.get_template())
        observed = runtime.observed_engine_identity(engine, contract, swift_args)
        if service_contract(args, args.port) != contract:
            raise ValueError("Checkpoint/source/runtime changed while the engine initialized")
        identity = {"contract": contract, "runtime": observed, "instance": uuid.uuid4().hex, "pid": os.getpid(),
                    "process_start_ticks": runtime.process_start_ticks(os.getpid()),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        validate_listener(identity, os.getpid())
        state.update(engine=engine, identity=identity, identity_sha256=runtime.digest(identity))
        args.service_identity.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.service_identity.with_name(args.service_identity.name + f".{os.getpid()}.partial")
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(identity, output, sort_keys=True, indent=2)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, args.service_identity)
        yield

    app = FastAPI(lifespan=lifespan)

    def response(payload, status=200):
        return JSONResponse(payload, status_code=status, headers={
            SERVICE_HEADER: state["identity"]["instance"], IDENTITY_HEADER: state["identity_sha256"]})

    @app.get("/health/")
    async def health():
        return response({"ready": True})

    @app.post("/external_onestep_identity/")
    async def identity_route(request: Request):
        body = await request.json()
        return response({"challenge": body.get("challenge"), "identity": state["identity"]})

    @app.post("/infer/")
    async def infer_route(request: Request):
        if (request.headers.get(SERVICE_HEADER) != state["identity"]["instance"]
                or request.headers.get(IDENTITY_HEADER) != state["identity_sha256"]):
            return response({"error": "Service identity headers required", "kind": "identity"}, 409)
        body = await request.json()
        if body.get("request_config") not in allowed or len(body.get("infer_requests", [])) != 1:
            return response({"error": "Only the fixed single-request sampling protocol is accepted", "kind": "protocol"}, 400)
        try:
            output = await runtime.infer_controlled_batch(
                state["engine"], [RolloutInferRequest(**body["infer_requests"][0])], RequestConfig(**body["request_config"]))
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:2000]
            return response({"error": message, "kind": classify_engine_error(message)}, 500)
        return response(output)

    uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")).run(sockets=[listener])
    return 0


def check_identity(identity: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    if identity.get("contract") != contract:
        raise ValueError("Service checkpoint/template/runtime/source/endpoint contract mismatch")
    observed = identity.get("runtime", {})
    if (observed.get("engine_class") != contract["engine_class"] or observed.get("enable_multi_turn") is not False
            or observed.get("multi_turn_scheduler") is not None or observed.get("model_dir") != contract["checkpoint"]["path"]
            or observed.get("model_name") != runtime.swift_model_name(observed.get("engine_model_dir", ""))):
        raise ValueError("Service runtime engine/checkpoint/single-turn identity mismatch")
    for name in ("torch_dtype", "model_type", "template_type", "template_backend", "response_prefix"):
        if observed.get(name) != contract[name]:
            raise ValueError(f"Service runtime {name} identity mismatch")
    engine_args = observed.get("engine_args", {})
    if any(engine_args.get(key) != contract[key] for key in
           ("tensor_parallel_size", "data_parallel_size", "max_model_len", "gpu_memory_utilization")):
        raise ValueError("Service actual vLLM runtime configuration mismatch")
    return observed


class Endpoint:
    def __init__(self, path: Path, expected_pid: int | None, contract: dict[str, Any], protocol: str):
        self.path, self.expected_pid, self.protocol = path, expected_pid, protocol
        self.identity = json.loads(path.read_text())
        self.runtime = check_identity(self.identity, contract)
        self.sha = runtime.digest(self.identity)
        self.root = contract["endpoint"]
        self.headers = {SERVICE_HEADER: self.identity["instance"], IDENTITY_HEADER: self.sha}
        self.lock = threading.Lock()
        self.checked = 0.0
        self.local = threading.local()
        self.validate(force=True)
        challenge = uuid.uuid4().hex
        status, reply = self.post("/external_onestep_identity/", {"challenge": challenge}, 60)
        if status != 200 or reply != {"challenge": challenge, "identity": self.identity}:
            raise ValueError("Live service identity/challenge does not match its descriptor")

    def validate(self, force: bool = False) -> None:
        with self.lock:
            if not force and time.monotonic() - self.checked < IDENTITY_TTL_S:
                return
            if json.loads(self.path.read_text()) != self.identity:
                raise ValueError("Service identity file changed; refuse stale/replaced endpoint")
            validate_listener(self.identity, self.expected_pid)
            self.checked = time.monotonic()

    def session(self) -> requests.Session:
        if not hasattr(self.local, "session"):
            self.local.session = requests.Session()
            self.local.session.trust_env = False
        return self.local.session

    def post(self, path: str, body: dict[str, Any], timeout: float) -> tuple[int, Any]:
        self.validate()
        try:
            response = self.session().post(self.root + path, json=body, timeout=timeout, headers=self.headers,
                                           allow_redirects=False)
        except requests.Timeout as exc:
            raise ServiceError("timeout", str(exc)) from exc
        except requests.ConnectionError as exc:
            raise ServiceError("connection", str(exc)) from exc
        if response.headers.get(SERVICE_HEADER) != self.identity["instance"] or response.headers.get(IDENTITY_HEADER) != self.sha:
            raise ServiceError("identity", "Response is not from the bound controlled service instance")
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": response.text[:500]}
        return response.status_code, payload

    def infer(self, request: dict[str, Any], benchmark: str, attempts: int = 3) -> dict[str, Any]:
        body = {"infer_requests": [request], "request_config": P.sampling(self.protocol, benchmark)}
        for attempt in range(attempts):
            try:
                status, payload = self.post("/infer/", body, 7200)
                if status != 200:
                    kind = payload.get("kind") if isinstance(payload, dict) else None
                    raise ServiceError(kind or f"http_{status}", str(payload)[:2000])
                text, finish = runtime.unpack_single_turn(payload, self.runtime["model_name"])
                item = payload[0].get("response", payload[0])
                usage = item.get("usage") if isinstance(item, dict) else None
                return {"text": text, "finish_reason": finish, "usage": usage}
            except ServiceError as exc:
                if exc.kind not in RETRYABLE or attempt == attempts - 1:
                    raise
                time.sleep(5 * (attempt + 1))
        raise AssertionError("unreachable")


def benchmark_identity(benchmark: str, path: Path, rows: list[dict[str, Any]], total: int) -> dict[str, Any]:
    return {"data_path": str(path.resolve()), "data_sha256": runtime.file_digest(path), "data_rows": total,
            "selected_rows": len(rows), "row_fingerprint": runtime.digest(rows), "modality": P.modality(benchmark),
            "official_scorer": P.OFFICIAL_SCORERS.get(benchmark)}


def common_manifest(args: argparse.Namespace, binding: dict[str, Any] | None) -> dict[str, Any]:
    return {"version": VERSION, "protocol_version": P.PROTOCOL_VERSION, "model": args.protocol,
            "protocol_status": native.protocol_status(args.protocol), "modes": list(P.MODES),
            "sampling": {"image": P.sampling(args.protocol, P.IMAGE_BENCHMARKS[0]),
                         "video": P.sampling(args.protocol, P.VIDEO_BENCHMARKS[0])},
            "max_turns_w_tool": args.max_turns, "max_operations": args.max_operations, "max_turns_wo_tool": 1,
            "crop_tokens": args.crop_tokens, "pixels": {"MIN_PIXELS": P.MIN_PIXELS, "MAX_PIXELS": args.max_pixels, "patch": P.PATCH},
            "budget": budget_config(args),
            "timestamp_mode": args.timestamp_mode, "max_rows": args.max_rows,
            "video_frames": "64 pre-resized candidate frames (row image) as images; tools use tool_video_frames originals",
            "model_identity": runtime.model_identity(args.model), "code": source_identity(),
            "service_binding": binding, "deviations": P.deviations()}


def budget_config(args: argparse.Namespace) -> dict[str, Any]:
    default = (args.max_pixels, args.crop_tokens, args.max_turns, args.max_operations) == (
        P.MAX_PIXELS, P.CROP_TOKENS, P.MAX_TURNS, P.MAX_OPERATIONS)
    return {"default": default, "max_pixels": args.max_pixels, "crop_tokens": args.crop_tokens,
            "max_turns": args.max_turns, "max_operations": args.max_operations, "max_model_len": args.max_model_len}


def ensure_manifest(path: Path, common: dict[str, Any], benchmarks: dict[str, Any]) -> None:
    common, benchmarks = json.loads(json.dumps(common)), json.loads(json.dumps(benchmarks))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("common") != common:
            raise ValueError(f"Resume fingerprint mismatch in {path}; use a fresh output root")
        for name, entry in benchmarks.items():
            if name in saved["benchmarks"] and saved["benchmarks"][name] != entry:
                raise ValueError(f"Benchmark {name} data changed since {path}; use a fresh output root")
        merged = {**saved["benchmarks"], **benchmarks}
    else:
        merged = benchmarks
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps({"common": common, "benchmarks": merged}, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


@contextmanager
def model_lock(output_root: Path, protocol: str):
    with runtime.output_lock(output_root / "locks" / protocol):
        yield


def diagnostics(rows: list[dict[str, Any]], records: list[dict[str, Any]], errors: list[dict[str, Any]],
                benchmark: str, protocol: str, mode: str) -> dict[str, Any]:
    qids = [str(row["qid"]) for row in rows]
    expected = set(qids)
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["qids"] in expected:
            unique.setdefault(record["qids"], record)
    latest_errors: dict[str, dict[str, Any]] = {}
    for error in errors:
        latest_errors[str(error["qid"])] = error
    unresolved = {qid: error for qid, error in latest_errors.items() if qid in expected and qid not in unique}
    done = list(unique.values())
    flags = Counter()
    sources = Counter()
    calls = Counter()
    tokens = Counter()
    for record in done:
        external = record.get("external", {})
        for name, value in external.get("flags", {}).items():
            flags[name] += bool(value)
        sources[external.get("answer_source", "unknown")] += 1
        trace = record.get("tool_calls") or []
        calls["trajectories_with_call"] += bool(trace)
        calls["calls"] += len(trace)
        calls["successful_calls"] += sum(bool(item.get("success")) for item in trace)
        calls["unsupported_calls"] += sum(bool(item.get("unsupported")) for item in trace)
        tokens["prompt"] += sum(value or 0 for value in external.get("prompt_tokens", []))
        tokens["completion"] += sum(value or 0 for value in external.get("completion_tokens", []))
        tokens["turns"] += external.get("turns", 0)
    count = len(done)
    rate = (lambda value: value / count) if count else (lambda value: None)
    mcq = [record for record in done if record.get("external", {}).get("mcq_reference")]
    return {"benchmark": benchmark, "model": protocol, "mode": mode, "rows": len(qids), "completed": count,
            "complete": count == len(qids), "pending": len(qids) - count,
            "unresolved_errors": dict(Counter(error.get("kind", "unknown") for error in unresolved.values())),
            "error_events": len(errors),
            "strict_match_accuracy": rate(sum(float(record["match"]) for record in done)),
            "no_answer_rate": rate(sum(not record.get("external", {}).get("answer_found") for record in done)),
            "truncated_rate": rate(flags["truncated"]), "budget_exhausted_rate": rate(flags["budget_exhausted"]),
            "context_overflow_rate": rate(flags["context_overflow"]),
            "unsupported_rate": rate(flags["unsupported"]), "invalid_call_rate": rate(flags["invalid_call"]),
            "wo_tool_emitted_operation_rate": rate(flags["emitted_operation"]) if mode == "wo_tool" else None,
            "call_rate": rate(calls["trajectories_with_call"]), "calls": dict(calls),
            "mcq_rows": len(mcq), "mcq_letter_unparsed": sum(record["external"].get("mcq_letter_parsed") is False for record in mcq),
            "answer_sources": dict(sources), "mean_turns": rate(tokens["turns"]),
            "tokens": {"prompt": tokens["prompt"], "completion": tokens["completion"]}}


def write_outputs(directory: Path, rows: list[dict[str, Any]], benchmark: str, protocol: str, mode: str) -> dict[str, Any]:
    records = repair_journal(directory / "logs" / SAMPLE_FILE)
    errors = repair_journal(directory / "errors.jsonl")
    report = diagnostics(rows, records, errors, benchmark, protocol, mode)
    temporary = directory / "external_diagnostics.json.partial"
    directory.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, directory / "external_diagnostics.json")
    if report["complete"]:
        expected = {str(row["qid"]) for row in rows}
        unique: dict[str, float] = {}
        for record in records:
            if record["qids"] in expected:
                unique.setdefault(record["qids"], float(record["match"]))
        metrics = {"accuracy": sum(unique.values()) / len(unique), "correct": sum(unique.values()), "total": len(unique)}
        partial = directory / "logs" / "metrics_iter0.json.tmp"
        partial.write_text(json.dumps(metrics, indent=2) + "\n")
        os.replace(partial, directory / "logs" / "metrics_iter0.json")
    return report


def merge_visualprobe(reports: dict[tuple[str, str], dict[str, Any]], directories: dict[tuple[str, str], Path]) -> dict[str, Any]:
    merged = {}
    for mode in P.MODES:
        keys = [(bench, mode) for bench in VISUALPROBE]
        if not all(key in reports and reports[key]["complete"] for key in keys):
            merged[mode] = None
            continue
        rows = sum(reports[key]["rows"] for key in keys)
        strict = sum(reports[key]["strict_match_accuracy"] * reports[key]["rows"] for key in keys) / rows
        judged = [directories[key] / "logs" / "llm_judge_metrics.json" for key in keys]
        judge = None
        if all(path.is_file() for path in judged):
            values = [json.loads(path.read_text()) for path in judged]
            judge = sum(value["llm_judge_correct"] for value in values) / sum(value["total"] for value in values)
        merged[mode] = {"rows": rows, "strict_match_accuracy": strict, "llm_judge_accuracy": judge}
    return merged


def selected(args: argparse.Namespace) -> list[str]:
    names = [name for name in args.benchmarks.split(",") if name] if args.benchmarks else list(P.BENCHMARKS)
    unknown = sorted(set(names) - set(P.BENCHMARKS))
    if unknown:
        raise ValueError(f"Unknown benchmarks: {unknown}")
    return names


def selected_modes(args: argparse.Namespace) -> list[str]:
    modes = [mode for mode in args.modes.split(",") if mode] if args.modes else list(P.MODES)
    if set(modes) - set(P.MODES):
        raise ValueError(f"Unknown modes: {modes}")
    return modes


def sample_indices(count: int, samples: int) -> list[int]:
    if count <= samples:
        return list(range(count))
    return sorted({round(index * (count - 1) / (samples - 1)) for index in range(samples)}) if samples > 1 else [0]


def estimate_text_tokens(text: str) -> int:
    return -(-len(text) // 3)


def dry_build(args: argparse.Namespace) -> int:
    protocols = list(P.MODELS) if args.protocol == "all" else [args.protocol]
    report: dict[str, Any] = {"stage": "dry_build", "version": VERSION, "protocol_version": P.PROTOCOL_VERSION,
                              "protocols": protocols, "modes": selected_modes(args), "max_model_len": args.max_model_len,
                              "budget": budget_config(args),
                              "timestamp_mode": args.timestamp_mode, "media_samples_per_benchmark": args.media_samples,
                              "benchmarks": {}, "failures": []}
    for benchmark in selected(args):
        path = data_path(benchmark, args.image_data, args.video_data)
        rows, total = load_benchmark(path, args.max_rows)
        entry = {"data_path": str(path), "data_sha256": runtime.file_digest(path), "rows": total, "built_requests": 0,
                 "build_failures": 0, "media_checked_rows": 0, "max_inputs": 0, "worst_case_context": 0,
                 "max_prompt_estimate": 0, "visible_tokens_max": 0, "timestamp_modes": Counter()}
        samples = set(sample_indices(len(rows), args.media_samples))
        for index, row in enumerate(rows):
            try:
                media = P.media_for(row, benchmark)
            except (ValueError, TypeError) as exc:
                entry["build_failures"] += 1
                report["failures"].append({"benchmark": benchmark, "qid": str(row.get("qid")), "error": str(exc)})
                continue
            entry["max_inputs"] = max(entry["max_inputs"], len(media.visible))
            sizes = None
            visible_tokens = None
            if index in samples:
                try:
                    sizes = [P.header_size(source) for source in media.sources]
                    visible = [P.header_size(item) for item in media.visible]
                    visible_tokens = sum(P.image_tokens(size, max_pixels=args.max_pixels) for size in visible)
                    P.ImageBank(media.sources)[0]
                    P.ImageBank(media.visible)[len(media.visible) - 1]
                    entry["media_checked_rows"] += 1
                    entry["visible_tokens_max"] = max(entry["visible_tokens_max"], visible_tokens)
                except (OSError, ValueError) as exc:
                    entry["build_failures"] += 1
                    report["failures"].append({"benchmark": benchmark, "qid": str(row["qid"]), "error": f"media: {exc}"})
                    continue
            for protocol in protocols:
                for mode in report["modes"]:
                    try:
                        request, meta = P.build_request(row, benchmark, protocol, mode, media, args.timestamp_mode, sizes)
                        if any(key in json.dumps(request) for key in ("\"answer\"", "\"solution\"")):
                            raise ValueError("reference field leaked into request")
                    except (ValueError, TypeError, KeyError) as exc:
                        entry["build_failures"] += 1
                        report["failures"].append({"benchmark": benchmark, "qid": str(row["qid"]), "protocol": protocol,
                                                   "mode": mode, "error": str(exc)})
                        continue
                    entry["built_requests"] += 1
                    entry["timestamp_modes"][meta["timestamp_mode"]] += 1
                    if visible_tokens is not None:
                        text = sum(estimate_text_tokens(message["content"]) for message in request["messages"])
                        prompt = text + visible_tokens
                        budget = P.MAX_TOKENS[P.modality(benchmark)]
                        worst = prompt + (args.max_turns * budget + args.max_operations * (args.crop_tokens * P.MAX_CROP_BOXES + 256)
                                          if mode == "w_tool" else budget)
                        entry["max_prompt_estimate"] = max(entry["max_prompt_estimate"], prompt)
                        entry["worst_case_context"] = max(entry["worst_case_context"], worst)
        entry["timestamp_modes"] = dict(entry["timestamp_modes"])
        entry["expected_requests"] = total * len(protocols) * len(report["modes"]) if not args.max_rows else len(rows) * len(protocols) * len(report["modes"])
        entry["fits_max_model_len"] = entry["worst_case_context"] < args.max_model_len
        report["benchmarks"][benchmark] = entry
    report["trajectories_per_model_per_mode"] = sum(entry["rows"] for entry in report["benchmarks"].values())
    report["ready"] = not report["failures"] and all(
        entry["built_requests"] == entry["expected_requests"] and entry["fits_max_model_len"] and entry["media_checked_rows"]
        for entry in report["benchmarks"].values())
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)
    return 0 if report["ready"] else 2


def load_selection(args: argparse.Namespace) -> dict[str, tuple[Path, list[dict[str, Any]], int]]:
    selection = {}
    for benchmark in selected(args):
        path = data_path(benchmark, args.image_data, args.video_data)
        rows, total = load_benchmark(path, args.max_rows)
        selection[benchmark] = (path, rows, total)
    return selection


def summarize_all(args: argparse.Namespace, selection) -> dict[str, Any]:
    reports, directories = {}, {}
    for benchmark, (_, rows, _) in selection.items():
        for mode in selected_modes(args):
            directory = eval_dir(args.output_root, benchmark, args.protocol, mode)
            directories[(benchmark, mode)] = directory
            reports[(benchmark, mode)] = write_outputs(directory, rows, benchmark, args.protocol, mode)
    summary = {"model": args.protocol,
               "complete": all(report["complete"] for report in reports.values()),
               "entries": {f"{bench}/{mode}": report for (bench, mode), report in reports.items()},
               "visualprobe_merged": merge_visualprobe(reports, directories)}
    path = args.output_root / "results" / args.protocol / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    os.replace(partial, path)
    return summary


def rollout(args: argparse.Namespace, endpoints: list[Any] | None = None) -> int:
    selection = load_selection(args)
    modes = selected_modes(args)
    with model_lock(args.output_root, args.protocol):
        if endpoints is None:
            identities = [Path(value) for value in args.service_identities.split(",") if value]
            pids = [int(value) for value in args.expected_service_pids.split(",")] if args.expected_service_pids else [None] * len(identities)
            if len(pids) != len(identities) or not identities:
                raise ValueError("Need one expected PID per service identity")
            endpoints = []
            for index, (path, pid) in enumerate(zip(identities, pids)):
                identity = json.loads(path.read_text())
                port = int(identity["contract"]["endpoint"].rsplit(":", 1)[1])
                endpoints.append(Endpoint(path, pid, service_contract(args, port), args.protocol))
            binding = {key: value for key, value in service_contract(args, 1).items() if key != "endpoint"}
            binding["runtime"] = {key: value for key, value in endpoints[0].runtime.items()}
        else:
            binding = None
        manifest = args.output_root / "manifests" / f"{args.protocol}.json"
        ensure_manifest(manifest, common_manifest(args, binding),
                        {name: benchmark_identity(name, path, rows, total) for name, (path, rows, total) in selection.items()})
        audit = args.output_root / "manifests" / f"{args.protocol}.service_instances.jsonl"
        with audit.open("a") as handle:
            for endpoint in endpoints:
                handle.write(json.dumps(getattr(endpoint, "identity", {"test": True}), sort_keys=True) + "\n")
        jobs: queue.Queue = queue.Queue()
        journals: dict[tuple[str, str], Journal] = {}
        error_journals: dict[tuple[str, str], Journal] = {}
        pending = 0
        for benchmark, (_, rows, _) in selection.items():
            done = {}
            for mode in modes:
                directory = eval_dir(args.output_root, benchmark, args.protocol, mode)
                done[mode] = {record["qids"] for record in repair_journal(directory / "logs" / SAMPLE_FILE)}
                repair_journal(directory / "errors.jsonl")
                journals[(benchmark, mode)] = Journal(directory / "logs" / SAMPLE_FILE)
                error_journals[(benchmark, mode)] = Journal(directory / "errors.jsonl", fsync_every=1)
            for row in rows:
                for mode in modes:
                    if str(row["qid"]) not in done[mode]:
                        jobs.put((benchmark, row, mode))
                        pending += 1
        print(json.dumps({"event": "rollout_start", "model": args.protocol, "pending": pending,
                          "endpoints": len(endpoints), "workers_per_endpoint": args.workers_per_endpoint}), flush=True)
        progress = Counter()
        progress_lock = threading.Lock()
        started = time.monotonic()

        def worker(endpoint):
            while True:
                try:
                    benchmark, row, mode = jobs.get_nowait()
                except queue.Empty:
                    return
                key = (benchmark, mode)
                try:
                    record = run_trajectory(row, benchmark, args.protocol, mode, endpoint.infer, args.timestamp_mode,
                                            args.crop_tokens, max_pixels=args.max_pixels, max_turns=args.max_turns,
                                            max_operations=args.max_operations)
                    journals[key].write(record)
                    outcome = "ok"
                except ServiceError as exc:
                    outcome = exc.kind
                    error_journals[key].write({"qid": str(row["qid"]), "kind": exc.kind, "error": str(exc)[:2000], "time": time.time()})
                except (OSError, ValueError) as exc:
                    outcome = "data_or_client"
                    error_journals[key].write({"qid": str(row["qid"]), "kind": "data_or_client",
                                               "error": f"{type(exc).__name__}: {exc}"[:2000], "time": time.time()})
                except Exception as exc:  # noqa: BLE001
                    outcome = "client_error"
                    error_journals[key].write({"qid": str(row["qid"]), "kind": "client_error",
                                               "error": f"{type(exc).__name__}: {exc}"[:2000], "time": time.time()})
                with progress_lock:
                    progress[outcome] += 1
                    total = sum(progress.values())
                    if total % args.progress_every == 0 or total == pending:
                        elapsed = time.monotonic() - started
                        print(json.dumps({"event": "progress", "done": total, "pending": pending, "outcomes": dict(progress),
                                          "elapsed_s": round(elapsed), "per_s": round(total / max(elapsed, 1e-9), 3)}), flush=True)

        threads = [threading.Thread(target=worker, args=(endpoints[index % len(endpoints)],), daemon=True)
                   for index in range(len(endpoints) * args.workers_per_endpoint)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for journal in list(journals.values()) + list(error_journals.values()):
            journal.close()
        summary = summarize_all(args, selection)
    print(json.dumps({"event": "rollout_done", "complete": summary["complete"], "outcomes": dict(progress)}), flush=True)
    return 0 if summary["complete"] else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("dry_build", "serve", "rollout", "summarize"), required=True)
    parser.add_argument("--protocol", choices=(*P.MODELS, "all"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--image-data", type=Path, default=DEFAULT_IMAGE_DATA)
    parser.add_argument("--video-data", type=Path, default=DEFAULT_VIDEO_DATA)
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--modes", default="")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--timestamp-mode", choices=P.TIMESTAMP_MODES, default="reframe")
    parser.add_argument("--crop-tokens", type=int, default=P.CROP_TOKENS,
                        help=f"returned-image visual-token budget per operation (default {P.CROP_TOKENS}, max {P.MAX_CROP_TOKENS})")
    parser.add_argument("--max-pixels", type=int, default=P.MAX_PIXELS,
                        help=f"input pixel cap; must equal the services' MAX_PIXELS env (default {P.MAX_PIXELS}, "
                             f"official checkpoints {P.OFFICIAL_MAX_PIXELS})")
    parser.add_argument("--max-turns", type=int, default=P.MAX_TURNS)
    parser.add_argument("--max-operations", type=int, default=P.MAX_OPERATIONS)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--media-samples", type=int, default=6)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18200)
    parser.add_argument("--service-identity", type=Path)
    parser.add_argument("--service-identities", default="")
    parser.add_argument("--expected-service-pids", default="")
    parser.add_argument("--workers-per-endpoint", type=int, default=48)
    parser.add_argument("--progress-every", type=int, default=200)
    parser.add_argument("--max-model-len", type=int, default=81920)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--limit-mm-per-prompt", default="")
    parser.add_argument("--max-num-seqs", type=int, default=0)
    args = parser.parse_args(argv)
    if args.protocol == "all" and args.stage != "dry_build":
        parser.error("--protocol all is only valid for dry_build")
    if min(args.crop_tokens, args.workers_per_endpoint, args.max_model_len, args.media_samples, args.progress_every) <= 0:
        parser.error("crop tokens, workers, model length, media samples and progress interval must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("--max-rows must be positive")
    if not 0 < args.gpu_memory_utilization < 1:
        parser.error("--gpu-memory-utilization must be in (0, 1)")
    if args.crop_tokens > P.MAX_CROP_TOKENS:
        parser.error(f"--crop-tokens must be <= {P.MAX_CROP_TOKENS}")
    if not P.MIN_PIXELS * 4 <= args.max_pixels <= P.OFFICIAL_MAX_PIXELS:
        parser.error(f"--max-pixels must be within [{P.MIN_PIXELS * 4}, {P.OFFICIAL_MAX_PIXELS}]")
    if args.crop_tokens * P.PATCH * P.PATCH > args.max_pixels:
        parser.error("--crop-tokens * 784 must not exceed --max-pixels (the processor caps returned images too)")
    if not 1 <= args.max_operations < args.max_turns <= 32:
        parser.error("need 1 <= --max-operations < --max-turns <= 32")
    if args.stage in ("rollout", "summarize") and args.output_root is None:
        parser.error("--output-root is required")
    if args.stage == "serve" and (not args.model or args.service_identity is None):
        parser.error("serve requires --model and --service-identity")
    if args.stage == "rollout" and (not args.model or not args.service_identities):
        parser.error("rollout requires --model and --service-identities from the launcher")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "dry_build":
        return dry_build(args)
    if args.stage == "serve":
        return serve(args)
    if args.stage == "rollout":
        return rollout(args)
    with model_lock(args.output_root, args.protocol):
        summary = summarize_all(args, load_selection(args))
    print(json.dumps({key: summary[key] for key in ("model", "complete", "visualprobe_merged")}, indent=2))
    return 0 if summary["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
