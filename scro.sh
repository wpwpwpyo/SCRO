#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$project_dir"
export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"

export HF_HOME=${HF_HOME:-"$project_dir/../cache/.cache/huggingface"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$HF_HOME/datasets"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$HF_HOME/transformers"}

# Usage: bash scro_debug.sh MODEL DATASET N RUN_MODE GPU_DEVICES \
#   MICRO_BATCH USE_CACHE Z_OP_BOUND LAST_LAYER_FIT_ERROR
#
# Method choices are fixed: SCRO Stage2, spectral-scaled Z, constrained
# residual row space, subject-aware requests, fast context templates, the
# complete numerical-rank spectrum, a cosine scheduler with T_max=steps, and
# no sample clamp. Layers, steps, LR, weight decay, KL factor, and rewrite-loss
# alpha are read exclusively from hparams/SCRO/<model>.json.
if (( $# > 9 )); then
  echo "scro_debug.sh accepts at most 9 positional arguments." >&2
  exit 2
fi

model_name=${1:-Llama-3.1-8B}
ds_name=${2:-zsre}
dataset_size=${3:-500}
run_mode=${4:-edit_eval}
gpu_devices=${5:-0}
joint_z_micro_batch_size=${6:-2}
use_cache=${7:-1}
joint_spectral_z_op_bound=${8:-25}
last_layer_fit_error=${9:-off}

nnodes=${NNODES:-1}
node_rank=${NODE_RANK:-0}
master_addr=${MASTER_ADDR:-}
master_port=${MASTER_PORT:-29500}
export DIST_TIMEOUT_SECONDS=${DIST_TIMEOUT_SECONDS:-3600}

case "$run_mode" in
  save_zs|edit_eval|save_model) eval_only=0 ;;
  eval|glue_eval) eval_only=1 ;;
  *) echo "Unsupported run mode: $run_mode (expected save_zs/save_model/edit_eval/eval/glue_eval)" >&2; exit 2 ;;
esac

case "$model_name" in
  Llama-3.1-8B) hparams_fname=LLAMA3-8B.json ;;
  Llama-2-13b-hf) hparams_fname=LLAMA2-13B.json ;;
  Llama-2-7b-hf) hparams_fname=LLAMA2-7B.json ;;
  falcon-7b) hparams_fname=Falcon-7B.json ;;
  deepseek-llm-7b-base) hparams_fname=Deepseek-7B.json ;;
  Qwen2.5-7B) hparams_fname=Qwen-7B.json ;;
  gemma-7b-it) hparams_fname=GEMMA-7B.json ;;
  phi-1_5) hparams_fname=Phi-1_5.json ;;
  *) echo "Unsupported model_name: $model_name" >&2; exit 2 ;;
esac
hparams_path="$project_dir/hparams/SCRO/$hparams_fname"
if [[ ! -f "$hparams_path" ]]; then
  echo "Missing SCRO config for $model_name: $hparams_path" >&2
  exit 2
fi

last_layer_fit_error=${last_layer_fit_error,,}
case "$last_layer_fit_error" in
  off|on) ;;
  *) echo "LAST_LAYER_FIT_ERROR must be off or on" >&2; exit 2 ;;
esac
if [[ "$last_layer_fit_error" == "on" ]]; then
  case "$run_mode" in
    save_model|edit_eval) ;;
    *) echo "LAST_LAYER_FIT_ERROR=on requires save_model or edit_eval" >&2; exit 2 ;;
  esac
fi
if [[ "$use_cache" != "0" && "$use_cache" != "1" ]]; then
  echo "USE_CACHE must be 0 or 1" >&2
  exit 2
fi
if [[ -z "$joint_spectral_z_op_bound" ]]; then
  echo "Z_OP_BOUND is required" >&2
  exit 2
fi
if [[ ! "$joint_z_micro_batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "MICRO_BATCH must be a positive integer" >&2
  exit 2
fi
if [[ ! "$nnodes" =~ ^[1-9][0-9]*$ ]]; then
  echo "NNODES must be a positive integer" >&2
  exit 2
fi
if [[ ! "$node_rank" =~ ^[0-9]+$ ]] || (( node_rank >= nnodes )); then
  echo "NODE_RANK must be an integer in [0, NNODES)" >&2
  exit 2
fi
if [[ ! "$master_port" =~ ^[0-9]+$ ]] || (( master_port < 1 || master_port > 65535 )); then
  echo "MASTER_PORT must be an integer in [1, 65535]" >&2
  exit 2
fi
if [[ ! "$DIST_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "DIST_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi
if (( nnodes > 1 && ! eval_only )) && [[ -z "$master_addr" ]]; then
  echo "MASTER_ADDR must be set when NNODES > 1" >&2
  exit 2
fi
if [[ -z "$gpu_devices" ]]; then
  echo "GPU_DEVICES must contain at least one physical GPU id" >&2
  exit 2
fi

IFS=',' read -r -a gpu_array <<< "$gpu_devices"
for gpu_id in "${gpu_array[@]}"; do
  if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "GPU_DEVICES must be a comma-separated list of integers" >&2
    exit 2
  fi
done
local_world_size=${#gpu_array[@]}
distributed_world_size=$((nnodes * local_world_size))
export CUDA_VISIBLE_DEVICES="$gpu_devices"
device=0
launcher=(python -u)
distributed_args=()
if (( distributed_world_size > 1 )); then
  distributed_args=(--distributed)
  if (( nnodes == 1 )); then
    launcher=(torchrun --standalone --nproc-per-node="$local_world_size")
  else
    launcher=(
      torchrun
      --nnodes="$nnodes"
      --node-rank="$node_rank"
      --nproc-per-node="$local_world_size"
      --rdzv-backend=static
      --master-addr="$master_addr"
      --master-port="$master_port"
    )
  fi
fi
hparams_tag=$(sha256sum "$hparams_path" | cut -c1-12)
args_name="scro_hparams${hparams_tag}_zop_h${joint_spectral_z_op_bound}"
args_name+="_micro_batch${joint_z_micro_batch_size}"
if (( distributed_world_size > 1 )); then
  args_name+="_dist${distributed_world_size}"
fi

results_dir="results/${model_name}/${ds_name}/SCRO/${args_name}_ds${dataset_size}"
logs_dir="logs/${model_name}/${ds_name}/SCRO/${args_name}_ds${dataset_size}"
if (( eval_only && node_rank != 0 )); then
  echo "$run_mode runs only on node 0; skipping node $node_rank."
  exit 0
fi
if (( eval_only )) && [[ ! -d "$results_dir/edited_model/ds${dataset_size}" ]]; then
  echo "Saved model not found: $results_dir/edited_model/ds${dataset_size}" >&2
  echo "Run save_model or edit_eval with the same configuration first." >&2
  exit 2
fi
mkdir -p "$results_dir" "$logs_dir"

command=(
  "${launcher[@]}" -m experiments.edit
  --alg_name SCRO
  --model_name "$model_name"
  --hparams_fname "$hparams_fname"
  --ds_name "$ds_name"
  --dataset_size_limit "$dataset_size"
  --results_dir "$results_dir"
  --device "$device"
  --joint_spectral_z_op_bound "$joint_spectral_z_op_bound"
  --joint_z_micro_batch_size "$joint_z_micro_batch_size"
  "${distributed_args[@]}"
)
if [[ "$last_layer_fit_error" == "on" ]]; then
  command+=(--last_layer_fit_error)
fi
if [[ "$use_cache" == "1" ]]; then
  command+=(--use_cache)
fi

case "$run_mode" in
  save_zs) command+=(--only_save_zs) ;;
  edit_eval) command+=(--eval_in_memory) ;;
  save_model) ;;
  eval)
    command=(
      python -u -m experiments.eval
      --alg_name SCRO
      --model_name "$model_name"
      --ds_name "$ds_name"
      --results_dir "$results_dir"
      --dataset_size_limit "$dataset_size"
      --total_machine 1
      --this_machine 0
      --device "$device"
      --assigned_prefix_len 0
      --store_case_metrics
    )
    ;;
  glue_eval)
    command=(
      python -u -m experiments.glue_eval
      --alg_name SCRO
      --results_dir "$results_dir"
      --dataset_size_limit "$dataset_size"
      --device "$device"
    )
    ;;
esac

log_file="$logs_dir/${run_mode}.log"
if (( nnodes > 1 )); then
  log_file="$logs_dir/${run_mode}_node${node_rank}.log"
fi

printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu_devices"
printf 'SCRO topology: nnodes=%s node_rank=%s local_world_size=%s world_size=%s\n' \
  "$nnodes" "$node_rank" "$local_world_size" "$distributed_world_size"
if (( nnodes > 1 )); then
  printf 'Multi-node mode requires shared data, cache, and result paths across nodes.\n'
fi
if (( eval_only )); then
  printf 'Standalone evaluation: one process on node 0, logical device=%s.\n' "$device"
fi
printf 'SCRO command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}" 2>&1 | tee "$log_file"
