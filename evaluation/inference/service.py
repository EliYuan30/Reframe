from __future__ import annotations
import hashlib
import json
import os
import re
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

SWIFT_ROOT = Path(__file__).resolve().parents[2] / "ms-swift" / "swift"

def digest(value: Any) -> str:
    def encode(item):
        if isinstance(item, bytes):
            return {"bytes_sha256": hashlib.sha256(item).hexdigest()}
        raise TypeError(f"Unfingerprintable value: {type(item).__name__}")
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=encode).encode()).hexdigest()



def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()



def model_identity(model: str | None) -> Any:
    if not model:
        return None
    path = Path(model)
    if not path.is_dir():
        return {"model": model, "local": False}
    config_names = ('config.json', 'tokenizer_config.json', 'generation_config.json', 'preprocessor_config.json',
                    'processor_config.json', 'tokenizer.json', 'model.safetensors.index.json', 'pytorch_model.bin.index.json',
                    'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'tokenizer.model', 'vocab.txt', 'merges.txt')
    metadata = {path / name for name in config_names if (path / name).is_file()}
    metadata |= set(path.glob('*.jinja')) | set(path.rglob('*.py'))
    configs = {str(entry.relative_to(path)): file_digest(entry) for entry in sorted(metadata) if entry.is_file()}
    shards = {entry.name: {"path": str(entry.resolve()), "size": entry.stat().st_size,
                           "mtime_ns": entry.stat().st_mtime_ns, "inode": entry.stat().st_ino}
              for entry in sorted(set(path.glob("*.safetensors")) | set(path.glob("pytorch_model*.bin")))}
    return {"path": str(path.resolve()), "configs": configs, "shard_size_mtime_ns": shards}



def swift_model_name(model: str) -> str:
    path = str(model).rstrip('/')
    match = re.search(r'/models--(?:.+?--)?(.+?)/snapshots/', path) or re.search(r'/models/(?:.+?--)?(.+?)/snapshots/', path)
    return match.group(1) if match else path.rsplit('/', 1)[-1].replace('___', '.')



def endpoint_for(host: str, port: int) -> str:
    if host not in ('127.0.0.1', 'localhost') or not 1 <= port <= 65535:
        raise ValueError('Formal rollout requires this launcher-owned loopback service; arbitrary/remote endpoints are forbidden')
    return f'http://127.0.0.1:{port}'



def process_start_ticks(pid: int) -> str:
    return Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[19]



def json_snapshot(value: Any) -> Any:
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (tuple, list)):
        return [json_snapshot(part) for part in value]
    if isinstance(value, dict):
        return {str(key): json_snapshot(part) for key, part in value.items()}
    if isinstance(value, type):
        return value.__module__ + '.' + value.__qualname__
    raise ValueError(f'Non-reproducible runtime template value: {type(value).__name__}')



def observed_engine_identity(engine: Any, contract: dict[str, Any], deployment: Any) -> dict[str, Any]:
    if deployment.multi_turn_scheduler is not None or deployment.use_gym_env or deployment.max_turns != 1:
        raise ValueError('Controlled deployment must disable scheduler/gym and use one turn')
    template = engine.template
    meta = template.template_meta
    template_meta = {field.name: json_snapshot(getattr(meta, field.name)) for field in fields(meta)} if is_dataclass(meta) else None
    runtime = {'engine_class': type(engine).__module__ + '.' + type(engine).__qualname__,
               'model_dir': str(Path(engine.model_dir).resolve()), 'engine_model_dir': str(engine.model_dir), 'model_name': engine.model_name,
               'model_type': engine.model_info.model_type, 'torch_dtype': str(engine.model_info.torch_dtype).removeprefix('torch.'),
               'template_type': meta.template_type, 'template_backend': template.template_backend,
               'response_prefix': template.response_prefix, 'resolved_response_prefix': template._get_response_prefix(),
               'template_class': type(template).__module__ + '.' + type(template).__qualname__, 'template_meta': template_meta,
               'template_settings': {key: json_snapshot(getattr(template, key)) for key in
                                     ('use_chat_template', 'enable_thinking', 'preserve_thinking', 'add_non_thinking_prefix', 'chat_template_kwargs')},
               'engine_args': {key: json_snapshot(getattr(engine.engine_args, key)) for key in
                               ('model', 'dtype', 'tensor_parallel_size', 'data_parallel_size', 'max_model_len', 'gpu_memory_utilization')},
               'enable_multi_turn': False, 'multi_turn_scheduler': None,
               'deployment': {'max_turns': deployment.max_turns, 'use_gym_env': deployment.use_gym_env}}
    runtime['engine_args']['model'] = str(Path(runtime['engine_args']['model']).resolve())
    if runtime['engine_class'] != contract['engine_class'] or runtime['enable_multi_turn'] or runtime['multi_turn_scheduler']:
        raise ValueError('Actual runtime is not the required single-turn GRPOVllmEngine')
    for key in ('model_type', 'torch_dtype', 'template_type', 'template_backend', 'response_prefix'):
        if runtime[key] != contract[key]:
            raise ValueError(f'Actual engine {key} differs from controlled contract')
    if runtime['model_dir'] != contract['checkpoint']['path'] or runtime['model_name'] != swift_model_name(runtime['engine_model_dir']):
        raise ValueError('Actual engine checkpoint differs from controlled contract')
    if str(Path(runtime['engine_args']['model']).resolve()) != runtime['model_dir'] or runtime['engine_args']['dtype'] not in ('bfloat16', 'torch.bfloat16'):
        raise ValueError('Actual vLLM model/dtype differs from controlled contract')
    for key in ('tensor_parallel_size', 'data_parallel_size', 'max_model_len', 'gpu_memory_utilization'):
        if runtime['engine_args'][key] != contract[key]:
            raise ValueError(f'Actual vLLM {key} differs from controlled contract')
    if not template_meta or not runtime['template_class'].startswith('swift.template.'):
        raise ValueError('Template/prefix implementation was not captured')
    return runtime



async def infer_controlled_batch(engine: Any, infer_requests: list, request_config: Any) -> list[dict[str, Any]]:
    from pydantic import BaseModel
    output = await engine.async_infer(infer_requests, request_config)
    if not isinstance(output, list) or len(output) != 1 or not isinstance(output[0], BaseModel):
        raise TypeError('Expected one native Pydantic RolloutOutput')
    return [item.model_dump(mode='json') for item in output]



@contextmanager
def output_lock(output: Path):
    import fcntl
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Output directory is already locked: {output}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)



def unpack_single_turn(outputs: Any, expected_model_name: str) -> tuple[str, str | None]:
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], dict):
        raise RuntimeError('Swift returned wrong output count/schema')
    item = outputs[0]
    if item.get('messages') is not None or any(item.get(name) for name in
            ('response_token_ids', 'response_loss_mask', 'rollout_infos', 'rollout_logprobs', 'prompt_logprobs')):
        raise RuntimeError('Scheduler/multi-turn trajectory output is forbidden')
    if 'response' in item:
        if 'choices' in item or not isinstance(item['response'], dict):
            raise RuntimeError('Malformed single-turn RolloutOutput.response')
        item = item['response']
    if item.get('messages') is not None:
        raise RuntimeError('Scheduler messages output is forbidden')
    if item.get('model') != expected_model_name:
        raise RuntimeError('Actual response model_name does not match the bound engine identity')
    choices = item.get('choices')
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError('Expected one single-turn Swift ChatCompletionResponse choice')
    message = choices[0].get('message', {})
    if message.get('role') != 'assistant' or not isinstance(message.get('content'), str):
        raise RuntimeError('Server completion is not a single assistant text response')
    return message['content'], choices[0].get('finish_reason')
