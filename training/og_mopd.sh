#!/usr/bin/env bash
set -euo pipefail

if [[ $# == 0 || $1 == --help || $1 == -h ]]; then
  echo 'Usage: bash training/og_mopd.sh --model PATH --image-teacher PATH --video-teacher PATH --data-root PATH --output-dir PATH'
  echo 'Options: --media-root PATH --resume PATH'
  echo 'Multi-node: --nodes N --node-rank N --master-addr HOST --master-port PORT'
  exit 0
fi
MODEL_PATH=""
IMAGE_TEACHER=""
VIDEO_TEACHER=""
DATA_ROOT=""
OUTPUT_DIR=""
RESUME_ARGS=()
export NNODES=1
export NODE_RANK=0
while (( $# )); do
  (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; }
  case "$1" in
    --model) MODEL_PATH=$2 ;;
    --image-teacher) IMAGE_TEACHER=$2 ;;
    --video-teacher) VIDEO_TEACHER=$2 ;;
    --data-root) DATA_ROOT=$2 ;;
    --media-root) export REFRAME_MEDIA_ROOT=$2 ;;
    --output-dir) OUTPUT_DIR=$2 ;;
    --resume) RESUME_ARGS=(--resume_from_checkpoint "$2") ;;
    --nodes) export NNODES=$2 ;;
    --node-rank) export NODE_RANK=$2 ;;
    --master-addr) export MASTER_ADDR=$2 ;;
    --master-port) export MASTER_PORT=$2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
  shift 2
done
: "${MODEL_PATH:?Pass --model}" "${IMAGE_TEACHER:?Pass --image-teacher}" "${VIDEO_TEACHER:?Pass --video-teacher}"
: "${DATA_ROOT:?Pass --data-root}" "${OUTPUT_DIR:?Pass --output-dir}"
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE

export REFRAME_DATA_ROOT=$DATA_ROOT
export REFRAME_MODE=mix
export MAX_PIXELS=4014080
export MIN_PIXELS=3136
export IMAGE_MAX_TOKEN_NUM=3920
export IMAGE_MIN_TOKEN_NUM=3
export VIDEO_MAX_TOKEN_NUM=128
export VIDEO_MIN_TOKEN_NUM=128
export REFRAME_MAX_TOOL_VISUAL_TOKENS=4096
export REFRAME_SELECT_MAX_TOKEN_NUM=400
export REFRAME_CROP_MAX_TOKEN_NUM=1960
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export WANDB_PROJECT=reframe
export no_proxy="localhost,127.0.0.1,::1${no_proxy:+,$no_proxy}"
export NO_PROXY=$no_proxy

mkdir -p "$OUTPUT_DIR/server_logs_node${NODE_RANK}"
if (( ${#RESUME_ARGS[@]} == 0 )) && compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
  echo 'Output already contains checkpoints; use --resume or a new --output-dir.' >&2
  exit 2
fi
exec > >(tee -a "$OUTPUT_DIR/train_node${NODE_RANK}.log") 2>&1

TEACHER_PIDS=()
cleanup() {
  local pid
  for pid in "${TEACHER_PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${TEACHER_PIDS[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_teacher() {
  local model=$1 port=$2 gpus=$3 log=$4
  env -u NNODES -u NODE_RANK -u MASTER_ADDR -u MASTER_PORT \
    -u GROUP_RANK -u ROLE_RANK -u LOCAL_SIZE CUDA_VISIBLE_DEVICES="$gpus" \
    setsid --wait swift deploy \
    --model "$model" \
    --model_type qwen3_5 \
    --infer_backend vllm \
    --vllm_enforce_eager true \
    --max_logprobs 1 \
    --max_length 32768 \
    --vllm_max_model_len 32768 \
    --vllm_data_parallel_size 1 \
    --vllm_tensor_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.42 \
    --vllm_engine_kwargs '{"async_scheduling": false, "mm_processor_cache_gb": 0, "additional_config": {"gdn_prefill_backend": "triton"}}' \
    --host 127.0.0.1 \
    --port "$port" >"$log" 2>&1 &
  local pid=$!
  TEACHER_PIDS+=("$pid")
  for (( attempt=0; attempt<900; attempt++ )); do
    kill -0 "$pid" 2>/dev/null || { tail -n 30 "$log" >&2; return 1; }
    curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "Teacher on port $port did not become ready; see $log" >&2
  return 1
}

start_teacher "$IMAGE_TEACHER" 18101 7 "$OUTPUT_DIR/server_logs_node${NODE_RANK}/image_teacher.log"
start_teacher "$VIDEO_TEACHER" 18102 7 "$OUTPUT_DIR/server_logs_node${NODE_RANK}/video_teacher.log"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 NPROC_PER_NODE=7 swift rlhf \
  --rlhf_type grpo \
  --model "$MODEL_PATH" \
  --model_type qwen3_5 \
  --tuner_type full \
  --teacher_model_server '[{"url":"http://127.0.0.1:18101","tags":["image"]},{"url":"http://127.0.0.1:18102","tags":["video"]}]' \
  --teacher_tag_key teacher_route \
  --teacher_kl_coef 1.0 \
  --num_iterations 4 \
  --mopd_advantage_clip 5.0 \
  --mopd_branch_loss_coef 1.0 \
  --external_plugins reframe/plugin.py \
  --dataset reframe-mix \
  --split_dataset_ratio 0 \
  --dataset_shuffle true \
  --load_from_cache_file false \
  --dataset_num_proc 8 \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.40 \
  --vllm_max_num_seqs 128 \
  --vllm_enable_prefix_caching true \
  --vllm_enforce_eager false \
  --vllm_engine_kwargs '{"max_num_batched_tokens": 65536, "async_scheduling": false, "additional_config": {"gdn_prefill_backend": "triton"}}' \
  --vllm_max_model_len 32768 \
  --sleep_level 1 \
  --multi_turn_scheduler og_mopd \
  --max_turns 3 \
  --stop_words '</tool_call>' \
  --completion_length_limit_scope total \
  --torch_dtype bfloat16 \
  --padding_free false \
  --max_length 32768 \
  --max_completion_length 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_generations 1 \
  --learning_rate 1e-6 \
  --optim adamw_torch_fused \
  --weight_decay 0.1 \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --adam_epsilon 1e-8 \
  --seed 42 \
  --data_seed 42 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.1 \
  --max_grad_norm 1.0 \
  --beta 0 \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k -1 \
  --deepspeed training/configs/zero2.json \
  --ddp_timeout 1800 \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --freeze_vit true \
  --freeze_aligner true \
  --logging_steps 1 \
  --save_strategy "${SAVE_STRATEGY:-steps}" \
  --save_steps 50 \
  --save_total_limit 12 \
  --eval_strategy no \
  --eval_on_start false \
  --output_dir "$OUTPUT_DIR" \
  --add_version false \
  --report_to "${REPORT_TO:-wandb}" \
  --run_name "$(basename "$OUTPUT_DIR")" \
  --log_completions true \
  "${RESUME_ARGS[@]}"
