#!/usr/bin/env bash
set -euo pipefail

export MIN_PIXELS=3136
export MAX_PIXELS=4014080
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VLLM_USE_FLASHINFER_SAMPLER=0

ARGS=(
  --protocol "$PROTOCOL"
  --model "$MODEL_PATH"
  --image-data "$IMAGE_DATA"
  --video-data "$VIDEO_DATA"
  --output-root "$OUTPUT_ROOT"
  --benchmarks "$BENCHMARKS"
  --modes "$MODES"
  --timestamp-mode reframe
  --max-model-len "${MAX_MODEL_LEN:-81920}"
  --max-pixels 4014080
  --crop-tokens 1960
  --max-turns 3
  --max-operations 2
  --gpu-memory-utilization 0.90
  --max-num-seqs 0
)
[[ -z "${MAX_ROWS:-}" ]] || ARGS+=(--max-rows "$MAX_ROWS")
case "$1" in
  dry_build)
    exec python evaluation/inference/external.py \
      --stage dry_build \
      --media-samples 6 \
      --report "$OUTPUT_ROOT/dry_build/dry_build_${PROTOCOL}.json" \
      "${ARGS[@]}" ;;
  summarize)
    exec python evaluation/inference/external.py \
      --stage summarize \
      "${ARGS[@]}" ;;
  run) ;;
  *) echo 'Use evaluation/run.py benchmark to launch evaluation.' >&2; exit 2 ;;
esac

IFS=, read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
WORK_DIR=$(mktemp -d /tmp/reframe-eval.XXXXXX)
LOG_DIR="$OUTPUT_ROOT/logs/$PROTOCOL"
SERVICE_DIR="$OUTPUT_ROOT/services/$PROTOCOL"
mkdir -p "$LOG_DIR" "$SERVICE_DIR"
SERVICE_PIDS=()
IDENTITIES=()
cleanup() {
  local pid
  for pid in "${SERVICE_PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${SERVICE_PIDS[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for i in "${!GPU_IDS[@]}"; do
  mkdir -p "$WORK_DIR/$i" "$OUTPUT_ROOT/cache/$PROTOCOL/gpu$i"
  IDENTITIES+=("$SERVICE_DIR/gpu$i.json")
  CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" TMPDIR="$WORK_DIR/$i" VLLM_RPC_BASE_PATH="$WORK_DIR/$i" \
    XDG_CACHE_HOME="$OUTPUT_ROOT/cache/$PROTOCOL/gpu$i" \
    VLLM_CACHE_ROOT="$OUTPUT_ROOT/cache/$PROTOCOL/gpu$i/vllm" \
    setsid python evaluation/inference/external.py \
    --stage serve \
    --port "$((PORT + i))" \
    --service-identity "${IDENTITIES[$i]}" \
    "${ARGS[@]}" >"$LOG_DIR/serve_gpu$i.log" 2>&1 &
  SERVICE_PIDS+=("$!")
done

deadline=$((SECONDS + 2400))
for i in "${!SERVICE_PIDS[@]}"; do
  ready=0
  while (( SECONDS < deadline )); do
    kill -0 "${SERVICE_PIDS[$i]}" 2>/dev/null || break
    if [[ -f "${IDENTITIES[$i]}" ]] \
        && curl --noproxy '*' --max-time 2 -fsS "http://127.0.0.1:$((PORT + i))/health/" >/dev/null 2>&1 \
        && python -c 'import json,sys; sys.exit(json.load(open(sys.argv[1]))["pid"] != int(sys.argv[2]))' \
          "${IDENTITIES[$i]}" "${SERVICE_PIDS[$i]}"; then
      ready=1
      break
    fi
    sleep 2
  done
  [[ "$ready" == 1 ]] || { tail -n 60 "$LOG_DIR/serve_gpu$i.log" >&2; exit 1; }
done

python evaluation/inference/external.py \
  --stage rollout \
  --service-identities "$(IFS=,; echo "${IDENTITIES[*]}")" \
  --expected-service-pids "$(IFS=,; echo "${SERVICE_PIDS[*]}")" \
  --workers-per-endpoint "$BATCH_SIZE" \
  "${ARGS[@]}" >"$LOG_DIR/rollout.log" 2>&1
