#!/usr/bin/env bash
set -euo pipefail

export MIN_PIXELS=3136
export MAX_PIXELS=4014080
export REFRAME_CROP_MAX_TOKEN_NUM=1960
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VLLM_USE_FLASHINFER_SAMPLER=0

ARGS=(
  --protocol "$PROTOCOL"
  --data "$PROBE_DATA"
  --model "$MODEL_PATH"
  --output-dir "$OUTPUT_DIR"
  --host 127.0.0.1
  --port "$PORT"
  --workers "$BATCH_SIZE"
  --service-identity "$OUTPUT_DIR/service_identity.json"
  --tensor-parallel-size 1
  --data-parallel-size 1
  --max-model-len "${MAX_MODEL_LEN:-65536}"
  --gpu-memory-utilization 0.85
)
[[ -z "${MAX_ROWS:-}" ]] || ARGS+=(--max-rows "$MAX_ROWS")
if [[ "$STAGE" != rollout ]]; then
  exec python evaluation/analysis/external_dependence.py \
    --stage "$STAGE" \
    "${ARGS[@]}"
fi

WORK_DIR=$(mktemp -d /tmp/reframe-dependence.XXXXXX)
mkdir -p "$OUTPUT_DIR/cache"
cleanup() {
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  sleep 1
  kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  rm -rf "$WORK_DIR"
}
TMPDIR="$WORK_DIR" XDG_CACHE_HOME="$OUTPUT_DIR/cache" VLLM_CACHE_ROOT="$OUTPUT_DIR/cache/vllm" \
  setsid python evaluation/analysis/external_dependence.py \
  --stage serve \
  "${ARGS[@]}" >"$OUTPUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=0
for ((attempt=0; attempt<900; attempt++)); do
  kill -0 "$SERVER_PID" 2>/dev/null || break
  if curl --noproxy '*' --max-time 2 -fsS "http://127.0.0.1:$PORT/health/" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { tail -n 60 "$OUTPUT_DIR/server.log" >&2; exit 1; }

python evaluation/analysis/external_dependence.py \
  --stage rollout \
  --expected-service-pid "$SERVER_PID" \
  "${ARGS[@]}"
