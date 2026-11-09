#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.benchmarks import DATASETS, IMAGE_GROUPS, jobs
from evaluation.scoring import JUDGE_BACKEND, JUDGE_BASE_URL, JUDGE_MODEL, JUDGE_THINKING_LEVEL

BACKENDS = ("reframe", "pixelreasoner", "thyme", "deepeyes", "deepeyesv2")
SCORERS = {name: name for name in ("videomme_v2", "videoocr", "vsi_bench", "perceptionbench", "vstar_video", "mlvu")}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Prepare benchmark Parquets or the V* dependence input")
    prep.add_argument("--benchmark", required=True, choices=("all", "vstar_dependence", *DATASETS))
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--adapter-args", nargs=argparse.REMAINDER, default=[])
    for name, help_text in (("benchmark", "Run the 14 benchmarks with or without tools"),
                            ("dependence", "Run the five V* observation conditions")):
        q = sub.add_parser(name, help=help_text)
        q.add_argument("--backend", choices=BACKENDS, default="reframe")
        q.add_argument("--model", type=Path)
        q.add_argument("--output-dir", type=Path, required=True)
        q.add_argument("--gpus", default="0")
        q.add_argument("--name", default="reframe", help="Output label for a Reframe/base/specialist checkpoint")
        q.add_argument("--port", type=int, default=32001)
        q.add_argument("--batch-size", type=positive_int, default=64)
        q.add_argument("--max-rows", type=positive_int, help="Evaluate only the first N questions")
        q.add_argument("--max-model-len", type=positive_int)
        q.add_argument("--dry-run", action="store_true", help="Print commands without model loading or API calls")
        if name == "benchmark":
            q.add_argument("--data-root", type=Path, required=True)
            q.add_argument("--video-data-root", type=Path)
            q.add_argument("--modality", choices=("all", "image", "video"), default="all")
            q.add_argument("--tool-mode", choices=("both", "w_tool", "wo_tool"), default="both")
            q.add_argument("--benchmarks", nargs="+", choices=(*DATASETS, "visualprobe"))
            q.add_argument("--stage", choices=("run", "infer", "score", "summarize", "dry-build"), default="run",
                           help="run: generate benchmark predictions and score; infer/score: run either stage separately")
            q.add_argument("--judge-backend", choices=("openai", "gemini"), default=JUDGE_BACKEND,
                           help="LLM judge API: an OpenAI-compatible endpoint or the native Gemini API "
                                "(default: %(default)s); the key is read from LLM_JUDGE_API_KEY")
            q.add_argument("--judge-base-url", default=JUDGE_BASE_URL,
                           help="OpenAI-compatible endpoint of the LLM judge (default: %(default)s)")
            q.add_argument("--judge-model", default=JUDGE_MODEL,
                           help="LLM judge model (default: %(default)s)")
            q.add_argument("--judge-thinking-level", default=JUDGE_THINKING_LEVEL,
                           help="thinking level of the gemini judge backend (default: %(default)s)")
            q.add_argument("--judge-workers", type=positive_int, default=8)
            q.add_argument("--videomme-v2-metadata", type=Path)
        else:
            q.add_argument("--data", type=Path, required=True)
            q.add_argument("--stage", choices=("rollout", "dry-build", "summarize"), default="rollout")
    report = sub.add_parser("report", help="Combine external-model observation-dependence reports")
    report.add_argument("--input-root", type=Path, required=True)
    return p


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "ms-swift"), env.get("PYTHONPATH", "")])
    return env


def execute(command, env, dry_run=False):
    command = [str(x) for x in command]
    if dry_run:
        print(shlex.join(command), flush=True)
    else:
        subprocess.run(command, env=env, cwd=ROOT, check=True)


def prepare(args):
    os.environ["REFRAME_EVAL_RAW_ROOT"] = str(args.data_root.resolve())
    os.environ["REFRAME_EVAL_OUTPUT_ROOT"] = str(args.output_dir.resolve())
    names = DATASETS if args.benchmark == "all" else (args.benchmark,)
    for name in names:
        sys.argv = [name, *args.adapter_args]
        if name == "vstar_dependence":
            sys.argv = [name, "--bench", str(args.data_root / "vstar.parquet"),
                        "--output", str(args.output_dir / "vstar_dependence.parquet")]
            runpy.run_module("evaluation.analysis.prepare", run_name="__main__")
        elif name.startswith("hrbench_"):
            importlib.import_module("evaluation.adapters.hrbench").prepare(name.rsplit("_", 1)[1])
        elif name.startswith("visualprobe_"):
            importlib.import_module("evaluation.adapters.visualprobe").prepare(name.rsplit("_", 1)[1])
        else:
            runpy.run_module(f"evaluation.adapters.{name}", run_name="__main__")


def selection(args):
    chosen = None
    if args.benchmarks:
        chosen = {part for name in args.benchmarks for part in IMAGE_GROUPS.get(name, (name,))}
    plan = [row for row in jobs(args.modality, args.tool_mode) if chosen is None or row[0] in chosen]
    if not plan:
        raise ValueError("No benchmarks match the requested modality and selection")
    return plan


def benchmark(args):
    env = environment()
    plan = selection(args)
    data_root = args.data_root.resolve()
    video_root = (args.video_data_root or data_root).resolve()
    output = args.output_dir.resolve()
    wants_score = args.stage in ("run", "score")
    if args.stage in ("run", "infer") and not args.model:
        raise ValueError("--model is required to generate benchmark predictions")
    if not args.dry_run:
        for name, _, modality in plan:
            path = (data_root if modality == "image" else video_root) / f"{name}.parquet"
            if not path.is_file():
                raise FileNotFoundError(path)
        if wants_score:
            if args.judge_backend == "gemini" and not os.environ.get("LLM_JUDGE_API_KEY"):
                raise ValueError("The gemini judge backend requires LLM_JUDGE_API_KEY")
            if any(name == "videomme_v2" for name, _, _ in plan) and (
                    not args.videomme_v2_metadata or not args.videomme_v2_metadata.is_file()):
                raise ValueError("Video-MME-v2 scoring requires --videomme-v2-metadata")
    if wants_score:
        env.update(LLM_JUDGE_BACKEND=args.judge_backend, LLM_JUDGE_BASE_URL=args.judge_base_url,
                   LLM_JUDGE_MODEL=args.judge_model, LLM_JUDGE_THINKING_LEVEL=args.judge_thinking_level)
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, NUM_GPUS=str(len(args.gpus.split(','))),
               PORT=str(args.port), BATCH_SIZE=str(args.batch_size),
               MAX_ROWS=str(args.max_rows) if args.max_rows else "",
               MAX_MODEL_LEN=str(args.max_model_len) if args.max_model_len else "")
    if args.backend != "reframe":
        env.update(PROTOCOL=args.backend, MODEL_PATH=str(args.model.resolve()) if args.model else "",
                   IMAGE_DATA=str(data_root), VIDEO_DATA=str(video_root), OUTPUT_ROOT=str(output),
                   BENCHMARKS=','.join(dict.fromkeys(name for name, _, _ in plan)),
                   MODES=','.join(dict.fromkeys(mode for _, mode, _ in plan)))
        if args.stage != "score":
            stage = {"infer": "run", "dry-build": "dry_build"}.get(args.stage, args.stage)
            print(f"{args.backend}: {stage}, {len(plan)} benchmark/mode jobs", flush=True)
            execute(["bash", ROOT / "evaluation/launchers/external.sh", stage], env, args.dry_run)
        if not wants_score:
            return
    summaries = {}
    for index, (name, mode, modality) in enumerate(plan):
        data = (data_root if modality == "image" else video_root) / f"{name}.parquet"
        label = args.name if args.backend == "reframe" else args.backend
        log_dir = output / "results" / label / name / mode / "logs"
        print(f"[{index+1}/{len(plan)}] {name} {mode} -> {log_dir}", flush=True)
        if args.stage == "dry-build":
            if args.dry_run:
                print(f"Build requests from {data}")
            else:
                import pyarrow.parquet as pq
                from evaluation.inference.reframe import build_request
                rows = pq.read_table(data).to_pylist()[:args.max_rows]
                for row in rows:
                    build_request(row, modality, mode == "w_tool", name)
                print(f"Built {len(rows)} requests")
            continue
        if args.backend == "reframe" and args.stage in ("run", "infer"):
            job_env = dict(env, MODEL_PATH=str(args.model.resolve()), DATA=str(data),
                           OUTPUT_DIR=str(log_dir.parent), BENCHMARK=name, REFRAME_MODE=modality,
                           TOOL_MODE=mode, PORT=str(args.port+index),
                           MAX_TOKENS=str(8192 if modality == "image" else 16384))
            execute(["bash", ROOT / "evaluation/launchers/reframe.sh"], job_env, args.dry_run)
        judge = ["--judge-workers", str(args.judge_workers)]
        if wants_score:
            score_flag = log_dir / f".scored_{args.judge_backend}_{args.judge_model}"
            if args.backend != "reframe" and not args.dry_run:
                if not (log_dir / "metrics_iter0.json").is_file():
                    raise FileNotFoundError(f"Incomplete external evaluation: {log_dir}")
                if score_flag.is_file():
                    continue
            if name != "vstar_video":
                execute([sys.executable, "-m", "evaluation.scoring.judge", "--log-dir", log_dir, *judge],
                        env, args.dry_run)
            if name in SCORERS:
                metadata = args.videomme_v2_metadata or Path('/path/to/your/Video-MME-v2/test.parquet')
                extra = ["--metadata", metadata] if name == "videomme_v2" else ["--data", data]
                if name == "vstar_video":
                    extra += judge
                execute([sys.executable, "-m", f"evaluation.scoring.{SCORERS[name]}", "--log-dir", log_dir, *extra], env, args.dry_run)
            if args.backend != "reframe" and not args.dry_run:
                metric_file = "vstar_video_metrics.json" if name == "vstar_video" else "llm_judge_metrics.json"
                if json.loads((log_dir / metric_file).read_text()).get("judge_failed", 0):
                    score_flag.unlink(missing_ok=True)
                else:
                    score_flag.touch()
        if args.stage == "summarize":
            summaries[f"{name}/{mode}"] = {p.name: json.loads(p.read_text()) for p in log_dir.glob('*metrics*.json')}
    if args.stage == "summarize":
        if not any(summaries.values()):
            raise FileNotFoundError(f"No benchmark metrics under {output}")
        print(json.dumps(summaries, indent=2))
    if args.backend != "reframe" and wants_score:
        execute(["bash", ROOT / "evaluation/launchers/external.sh", "summarize"], env, args.dry_run)


def dependence(args):
    env = environment()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, PROBE_DATA=str(args.data.resolve()),
               NUM_GPUS=str(len(args.gpus.split(','))), PORT=str(args.port),
               BATCH_SIZE=str(args.batch_size), STAGE=args.stage.replace('-', '_'),
               MAX_ROWS=str(args.max_rows) if args.max_rows else "",
               MAX_MODEL_LEN=str(args.max_model_len) if args.max_model_len else "")
    if args.stage == "rollout" and not args.model:
        raise ValueError("--model is required for rollout")
    if args.backend != "reframe":
        env.update(PROTOCOL=args.backend, MODEL_PATH=str(args.model.resolve()) if args.model else "",
                   OUTPUT_DIR=str(args.output_dir.resolve() / "dependence" / args.backend))
        execute(["bash", ROOT / "evaluation/launchers/external_dependence.sh"], env, args.dry_run)
    elif args.stage == "rollout":
        env.update(MODEL_PATH=str(args.model.resolve()), MODEL_NAME=args.name,
                   OUTPUT_DIR=str(args.output_dir.resolve() / "dependence" / args.name))
        execute(["bash", ROOT / "evaluation/launchers/dependence.sh"], env, args.dry_run)
    else:
        command = [sys.executable, "-m", "evaluation.analysis.dependence", "--stage", env["STAGE"],
                   "--modality", "image", "--data", args.data.resolve(),
                   "--output-dir", args.output_dir.resolve() / "dependence" / args.name, "--name", args.name]
        if args.max_rows:
            command += ["--dry-build-rows", args.max_rows]
        execute(command, env, args.dry_run)


def main():
    p = parser()
    args = p.parse_args()
    try:
        if args.command == "prepare":
            prepare(args)
        elif args.command == "benchmark":
            benchmark(args)
        elif args.command == "dependence":
            dependence(args)
        else:
            execute([sys.executable, "-m", "evaluation.analysis.report", args.input_root.resolve()], environment())
    except (ValueError, FileNotFoundError) as exc:
        p.error(str(exc))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode)


if __name__ == "__main__":
    main()
