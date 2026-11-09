#!/usr/bin/env bash
set -euo pipefail

if [[ $# == 0 || $1 == --help || $1 == -h ]]; then
  echo 'Usage: bash training/gd_grpo.sh image|video|mix --model PATH --data-root PATH --output-dir PATH'
  echo 'Options: --media-root PATH --resume PATH'
  echo 'Multi-node: --nodes N --node-rank N --master-addr HOST --master-port PORT'
  exit 0
fi
MODE=$1
shift
if [[ "$MODE" != image && "$MODE" != video && "$MODE" != mix ]]; then
  echo "Unknown mode: $MODE" >&2
  exit 2
fi

MODEL_PATH=""
DATA_ROOT=""
OUTPUT_DIR=""
RESUME_ARGS=()
export NNODES=1
export NODE_RANK=0
while (( $# )); do
  (( $# >= 2 )) || { echo "Missing value for $1" >&2; exit 2; }
  case "$1" in
    --model) MODEL_PATH=$2 ;;
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
: "${MODEL_PATH:?Pass --model}" "${DATA_ROOT:?Pass --data-root}" "${OUTPUT_DIR:?Pass --output-dir}"
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE

export REFRAME_DATA_ROOT=$DATA_ROOT
export REFRAME_MODE=$MODE
export GD_GRPO_DROP_SUBGROUP_ADVANTAGES=${GD_GRPO_DROP_SUBGROUP_ADVANTAGES:-1}
read -r -a REWARDS <<< "${REWARD_FUNCS:-reframe_answer reframe_call reframe_observation reframe_penalty}"

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

mkdir -p "$OUTPUT_DIR"
if (( ${#RESUME_ARGS[@]} == 0 )) && compgen -G "$OUTPUT_DIR/checkpoint-*" >/dev/null; then
  echo 'Output already contains checkpoints; use --resume or a new --output-dir.' >&2
  exit 2
fi
exec > >(tee -a "$OUTPUT_DIR/train_node${NODE_RANK}.log") 2>&1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC_PER_NODE=8 exec swift rlhf \
  --rlhf_type grpo \
  --model "$MODEL_PATH" \
  --model_type qwen3_5 \
  --tuner_type full \
  --external_plugins reframe/plugin.py \
  --dataset "reframe-$MODE" \
  --reward_funcs "${REWARDS[@]}" \
  --split_dataset_ratio 0 \
  --dataset_shuffle true \
  --load_from_cache_file false \
  --dataset_num_proc 8 \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.45 \
  --vllm_max_num_seqs 128 \
  --vllm_enable_prefix_caching true \
  --vllm_enforce_eager false \
  --vllm_engine_kwargs '{"max_num_batched_tokens": 65536, "async_scheduling": false, "additional_config": {"gdn_prefill_backend": "triton"}}' \
  --vllm_max_model_len 32768 \
  --sleep_level 1 \
  --multi_turn_scheduler gd_grpo \
  --max_turns 3 \
  --stop_words '</tool_call>' '<|box_end|>' '<|box_start|>' \
  --completion_length_limit_scope per_round \
  --torch_dtype bfloat16 \
  --padding_free false \
  --log_rollout_offpolicy_metrics true \
  --log_entropy true \
  --rollout_importance_sampling_mode sequence_truncate \
  --rollout_importance_sampling_threshold 2.0 \
  --max_length 32768 \
  --max_completion_length 2048 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_generations 8 \
  --num_iterations 1 \
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
  --loss_type dr_grpo \
  --beta 0.03 \
  --ref_model "$MODEL_PATH" \
  --offload_ref_model true \
  --mask_positive_invalid_final_advantages "${MASK_POSITIVE_INVALID_FINAL_ADVANTAGES:-true}" \
  --overlong_filter false \
  --sync_ref_model false \
  --temperature 1.0 \
  --top_p 0.95 \
  --top_k -1 \
  --deepspeed training/configs/zero2.json \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
  --freeze_vit true \
  --freeze_aligner true \
  --logging_steps 1 \
  --save_strategy "${SAVE_STRATEGY:-steps}" \
  --save_steps 50 \
  --save_total_limit 15 \
  --eval_strategy no \
  --eval_on_start false \
  --output_dir "$OUTPUT_DIR" \
  --add_version false \
  --report_to "${REPORT_TO:-wandb}" \
  --run_name "$(basename "$OUTPUT_DIR")" \
  --log_completions true \
  "${RESUME_ARGS[@]}"
