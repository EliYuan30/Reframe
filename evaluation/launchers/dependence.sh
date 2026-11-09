#!/usr/bin/env bash
set -euo pipefail

export MIN_PIXELS=4096
export MAX_PIXELS=5242880
export IMAGE_MIN_TOKEN_NUM=4
export IMAGE_MAX_TOKEN_NUM=5120
export VIDEO_MAX_TOKEN_NUM=128
export REFRAME_CROP_MAX_TOKEN_NUM=1960
export REFRAME_SELECT_MAX_TOKEN_NUM=400
export REFRAME_MAX_TOOL_VISUAL_TOKENS=4096
export REFRAME_MODE=image
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NCCL_NET_PLUGIN=none
export NCCL_NVLS_ENABLE=0
export SWIFT_ROLLOUT_ENCODE_WORKERS=64

mkdir -p "$OUTPUT_DIR"
cleanup() {
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  sleep 1
  kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
if curl --noproxy '*' --max-time 2 -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "Port $PORT is already in use" >&2
  exit 2
fi

setsid swift rollout \
  --model "$MODEL_PATH" \
  --model_type qwen3_5 \
  --external_plugins reframe/plugin.py \
  --multi_turn_scheduler og_mopd \
  --max_turns 3 \
  --stop_words '</tool_call>' \
  --vllm_use_async_engine true \
  --vllm_data_parallel_size "$NUM_GPUS" \
  --vllm_tensor_parallel_size 1 \
  --vllm_max_model_len "${MAX_MODEL_LEN:-18192}" \
  --vllm_gpu_memory_utilization 0.85 \
  --port "$PORT" >"$OUTPUT_DIR/rollout_server.log" 2>&1 &
SERVER_PID=$!
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=0
for ((attempt=0; attempt<1800; attempt++)); do
  kill -0 "$SERVER_PID" 2>/dev/null || break
  if curl --noproxy '*' --max-time 2 -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { tail -n 60 "$OUTPUT_DIR/rollout_server.log" >&2; exit 1; }

EXTRA_ARGS=()
[[ -z "${MAX_ROWS:-}" ]] || EXTRA_ARGS+=(--max-rows "$MAX_ROWS")
python -m evaluation.analysis.dependence \
  --stage rollout \
  --modality image \
  --data "$PROBE_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --model "$MODEL_PATH" \
  --name "$MODEL_NAME" \
  --ports "$PORT" \
  --batch-size "$BATCH_SIZE" \
  --max-tokens 8192 \
  --intervention-tools schemas \
  --branches direct natural gt random blank \
  "${EXTRA_ARGS[@]}"
