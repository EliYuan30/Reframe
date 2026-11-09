#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_identity(model_path: Path) -> dict:
    resolved = model_path.resolve()
    identity = {"path": str(resolved), "files": {}}
    if resolved.is_dir():
        for name in (
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "adapter_config.json",
            "trainer_state.json",
        ):
            candidate = resolved / name
            if candidate.is_file():
                identity["files"][name] = sha256_file(candidate)
    return identity


def build_eval_manifest(
    data_path: Path,
    model_path: Path,
    benchmark: str,
    mode: str,
    with_tools: bool,
    max_tokens: int,
    temperature: float,
    top_p: float,
    plugin_path: Path,
    eval_script_path: Path,
    max_model_len: int | None = None,
    max_rows: int | None = None,
) -> dict:
    manifest = {
        "schema_version": 1,
        "max_rows": max_rows,
        "data": {"path": str(data_path.resolve()), "sha256": sha256_file(data_path)},
        "model": model_identity(model_path),
        "benchmark": benchmark,
        "mode": mode,
        "with_tools": with_tools,
        "generation": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        "visual_budget": {
            name: os.environ.get(name)
            for name in (
                "MAX_PIXELS", "MIN_PIXELS", "IMAGE_MAX_TOKEN_NUM", "IMAGE_MIN_TOKEN_NUM",
                "VIDEO_MAX_TOKEN_NUM", "VIDEO_MIN_TOKEN_NUM", "REFRAME_CROP_MAX_TOKEN_NUM",
                "REFRAME_SELECT_MAX_TOKEN_NUM", "REFRAME_MAX_TOOL_VISUAL_TOKENS",
            )
        },
        "code": {
            "plugin_sha256": sha256_file(plugin_path),
            "shared_package": {path.name: sha256_file(path)
                               for path in sorted(plugin_path.parent.glob("*.py"))},
            "eval_script_sha256": sha256_file(eval_script_path),
        },
    }
    manifest["generation"]["max_model_len"] = max_model_len
    return manifest


def validate_eval_manifest(
    manifest_path: Path, expected: dict, log_dir: Path
) -> None:
    if manifest_path.is_file():
        actual = json.loads(manifest_path.read_text())
        if actual != expected:
            raise ValueError(
                "Evaluation manifest mismatch; use a new tagname/output directory.\n"
                f"existing={json.dumps(actual, sort_keys=True)}\n"
                f"requested={json.dumps(expected, sort_keys=True)}"
            )
        return
    if any(log_dir.glob("sample.eval_iter0_*.jsonl")):
        raise ValueError(
            f"Existing evaluation logs have no manifest: {log_dir}. "
            "Use a new tagname/output directory."
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(expected, indent=2) + "\n")
    temporary_path.replace(manifest_path)


def load_completed_qids(log_dir: Path) -> set[str]:
    completed = set()
    for path in sorted(log_dir.glob("sample.eval_iter0_*.jsonl")):
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"Ignoring malformed JSON at {path}:{line_number}",
                        file=sys.stderr,
                    )
                    continue
                qid = record.get("qids")
                if not qid:
                    raise ValueError(f"Missing qids at {path}:{line_number}")
                completed.add(str(qid))
    return completed


def prepare_resume_data(
    data_path: Path, log_dir: Path, output_path: Path, max_rows: int | None = None
) -> tuple[int, int, int]:
    table = pq.read_table(data_path)
    if max_rows is not None:
        if max_rows < 1:
            raise ValueError("max_rows must be positive")
        table = table.slice(0, max_rows)
    if "qid" not in table.column_names:
        raise ValueError(f"Evaluation data has no qid column: {data_path}")

    qids = [str(qid) for qid in table.column("qid").to_pylist()]
    if len(qids) != len(set(qids)):
        raise ValueError(f"Evaluation data contains duplicate qids: {data_path}")

    qid_set = set(qids)
    completed = load_completed_qids(log_dir)
    unexpected = completed - qid_set
    if unexpected:
        sample = sorted(unexpected)[:5]
        raise ValueError(
            f"Existing logs contain {len(unexpected)} qids not present in "
            f"{data_path}: {sample}"
        )

    remaining_mask = pa.array([qid not in completed for qid in qids])
    remaining = table.filter(remaining_mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pq.write_table(remaining, temporary_path)
    temporary_path.replace(output_path)
    return len(qids), len(completed), remaining.num_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--with-tools", action="store_true")
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    expected_manifest = build_eval_manifest(
        args.data,
        args.model,
        args.benchmark,
        args.mode,
        args.with_tools,
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.plugin,
        args.eval_script,
        max_model_len=args.max_model_len,
        max_rows=args.max_rows,
    )
    validate_eval_manifest(args.manifest, expected_manifest, args.log_dir)
    total, completed, remaining = prepare_resume_data(
        args.data, args.log_dir, args.output, args.max_rows
    )
    print(
        f"Evaluation resume: completed={completed}, remaining={remaining}, "
        f"total={total}",
        file=sys.stderr,
    )
    print(remaining)


if __name__ == "__main__":
    main()
