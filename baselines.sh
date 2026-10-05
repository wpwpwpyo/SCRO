#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$project_dir"

export HF_HOME=${HF_HOME:-"$project_dir/../cache/.cache/huggingface"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$HF_HOME/datasets"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$HF_HOME/transformers"}

if (( $# < 8 || $# > 10 )); then
    echo "Usage: bash general.sh ALG MODEL DATASET SIZE RUN_MODE TOTAL_MACHINE THIS_MACHINE DEVICE [LAST_LAYER_FIT_ERROR [EDIT_RESOURCE_PROFILE]]" >&2
    exit 2
fi

alg_name=$1 #EAMET MEMIT PMET ROME FT MEND ALPHAEDIT
model_name=$2 # Llama-3.1-8B falcon-7b deepseek-llm-7b-base Qwen2.5-7B gemma-7b-it phi-1_5 Llama-2-13b-hf 
ds_name=$3 #counterfact zsre wikirecent
dataset_size_limit=$4
run_mode=$5 # save_zs save_model eval glue_eval edit_eval
total_machine=$6
this_machine=$7
device=$8
# Position 9 enables the minimal final-layer DeltaK/R dump.
last_layer_fit_error=${9:-off}
edit_resource_profile=${10:-off}

edit_resource_profile=${edit_resource_profile,,}
case "$edit_resource_profile" in
    on|off) ;;
    *) echo "EDIT_RESOURCE_PROFILE must be on or off" >&2; exit 2 ;;
esac

edit_resource_profile_args=()
if [[ "$edit_resource_profile" == "on" ]]; then
    case "$run_mode" in
        save_zs|save_model|edit_eval|edit_eval_debug) ;;
        *)
            echo "EDIT_RESOURCE_PROFILE=on requires an editing run mode" >&2
            exit 2
            ;;
    esac
    if [[ "$total_machine" != "1" || "$this_machine" != "0" ]]; then
        echo "EDIT_RESOURCE_PROFILE=on requires TOTAL_MACHINE=1 and THIS_MACHINE=0" >&2
        exit 2
    fi
    edit_resource_profile_args=(--profile_edit_resources)
fi

last_layer_fit_error=${last_layer_fit_error,,}
case "$last_layer_fit_error" in
    on|off) ;;
    *) echo "LAST_LAYER_FIT_ERROR must be on or off" >&2; exit 2 ;;
esac
last_layer_fit_args=()
if [[ "$last_layer_fit_error" == "on" ]]; then
    case "$run_mode" in
        save_model|edit_eval|edit_eval_debug) ;;
        *)
            echo "LAST_LAYER_FIT_ERROR=on requires save_model, edit_eval, or edit_eval_debug" >&2
            exit 2
            ;;
    esac
    if [[ "$edit_resource_profile" == "on" ]]; then
        echo "LAST_LAYER_FIT_ERROR=on cannot be combined with EDIT_RESOURCE_PROFILE=on" >&2
        exit 2
    fi
    last_layer_fit_args=(--last_layer_fit_error)
fi

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

# bash general.sh MEMIT Llama-3.1-8B zsre 500 edit_eval 1 0 1
# bash general.sh EAMET Llama-3.1-8B zsre 100 edit_eval 1 0 0

args_name=${SAIR_RUN_TAG:-baseline}

results_dir="results/${model_name}/${ds_name}/${alg_name}/${args_name}"
logs_dir="logs/${model_name}/${ds_name}/${alg_name}/${args_name}/ds_${dataset_size_limit}"
mkdir -p "$logs_dir"

echo "$run_mode"

if [[ $run_mode == "save_zs" ]]; then
    nohup python3 -m experiments.edit \
        --alg_name $alg_name \
        --model_name $model_name \
        --hparams_fname $hparams_fname \
        --ds_name $ds_name \
        --results_dir $results_dir \
        --dataset_size_limit $dataset_size_limit \
        --total_machine $total_machine \
        --this_machine $this_machine \
        --device $device \
        --use_cache \
        --only_save_zs \
        "${edit_resource_profile_args[@]}" \
        > "$logs_dir/save_zs_${total_machine}_${this_machine}.log" 2>&1
elif [[ $run_mode == "save_model" ]]; then
    total_machine=1
    this_machine=0
    nohup python3 -m experiments.edit \
        --alg_name $alg_name \
        --model_name $model_name \
        --hparams_fname $hparams_fname \
        --ds_name $ds_name \
        --results_dir $results_dir \
        --dataset_size_limit $dataset_size_limit \
        --total_machine $total_machine \
        --this_machine $this_machine \
        --device $device \
        --use_cache \
        "${last_layer_fit_args[@]}" \
        "${edit_resource_profile_args[@]}" \
        > "$logs_dir/save_model.log" 2>&1
elif [[ $run_mode == "eval" ]]; then
    nohup python3 -m experiments.eval \
        --alg_name $alg_name \
        --model_name $model_name \
        --ds_name $ds_name \
        --results_dir $results_dir \
        --dataset_size_limit $dataset_size_limit \
        --total_machine $total_machine \
        --this_machine $this_machine \
        --device $device \
        --assigned_prefix_len 0 \
        --store_case_metrics \
        > "$logs_dir/eval_${total_machine}_${this_machine}.log" 2>&1
    
    # python3 -m experiments.eval \
    #     --alg_name $alg_name \
    #     --model_name $model_name \
    #     --ds_name $ds_name \
    #     --results_dir $results_dir \
    #     --dataset_size_limit $dataset_size_limit \
    #     --total_machine $total_machine \
    #     --this_machine $this_machine \
    #     --device $device \
    #     --assigned_prefix_len 0 \
    #     --store_case_metrics \
    #     --original
elif [[ $run_mode == "eval_original" ]]; then
    # nohup python3 -m experiments.eval \
    #     --alg_name $alg_name \
    #     --model_name $model_name \
    #     --ds_name $ds_name \
    #     --results_dir $results_dir \
    #     --dataset_size_limit $dataset_size_limit \
    #     --total_machine $total_machine \
    #     --this_machine $this_machine \
    #     --device $device \
    #     --assigned_prefix_len 0 \
    #     --store_case_metrics \
    #     --original \
    #     > "$logs_dir/eval_original_${total_machine}_${this_machine}.log" 2>&1

    
    python3 -m experiments.eval \
        --alg_name $alg_name \
        --model_name $model_name \
        --ds_name $ds_name \
        --results_dir $results_dir \
        --dataset_size_limit $dataset_size_limit \
        --total_machine $total_machine \
        --this_machine $this_machine \
        --device $device \
        --assigned_prefix_len 0 \
        --store_case_metrics \
        --original

elif [[ $run_mode == "glue_eval" ]]; then
    total_machine=1
    this_machine=0
    # nohup python3 -m experiments.glue_eval \
    #     --alg_name $alg_name \
    #     --results_dir $results_dir \
    #     --dataset_size_limit $dataset_size_limit \
    #     --device $device \
    #     > "$logs_dir/eval_glue.log" 2>&1
    python3 -m experiments.glue_eval \
        --alg_name $alg_name \
        --results_dir $results_dir \
        --dataset_size_limit $dataset_size_limit \
        --device $device
elif [[ $run_mode == "glue_eval_original" ]]; then
    total_machine=1
    this_machine=0
    python3 -m experiments.glue_eval \
        --alg_name $alg_name \
        --results_dir $results_dir \
        --dataset_size_limit $dataset_size_limit \
        --device $device \
        --original \
        --model_name $model_name
elif [[ $run_mode == "edit_eval" ]]; then
    total_machine=1
    this_machine=0
    nohup python3 -m experiments.edit \
        --alg_name $alg_name \
        --model_name $model_name \
        --hparams_fname $hparams_fname \
        --ds_name $ds_name \
        --results_dir $results_dir \
        --dataset_size_limit $dataset_size_limit \
        --total_machine $total_machine \
        --this_machine $this_machine \
        --device $device \
        --use_cache \
        "${last_layer_fit_args[@]}" \
        "${edit_resource_profile_args[@]}" \
        > "$logs_dir/save_zs_and_model.log" 2>&1

    nohup python3 -m experiments.eval \
        --alg_name $alg_name \
        --model_name $model_name \
        --ds_name $ds_name \
        --results_dir $results_dir \
        --dataset_size_limit $dataset_size_limit \
        --total_machine $total_machine \
        --this_machine $this_machine \
        --device $device \
        --assigned_prefix_len 0 \
        --store_case_metrics \
        > "$logs_dir/eval.log" 2>&1
elif [[ $run_mode == "edit_eval_debug" ]]; then
    total_machine=1
    this_machine=0
    python3 -m experiments.edit \
        --alg_name $alg_name \
        --model_name $model_name \
        --hparams_fname $hparams_fname \
        --ds_name $ds_name \
        --results_dir $results_dir \
        --dataset_size_limit $dataset_size_limit \
        --total_machine $total_machine \
        --this_machine $this_machine \
        --device $device \
        --use_cache \
        "${last_layer_fit_args[@]}" \
        "${edit_resource_profile_args[@]}"

    python3 -m experiments.eval \
        --alg_name $alg_name \
        --model_name $model_name \
        --ds_name $ds_name \
        --results_dir $results_dir \
        --dataset_size_limit $dataset_size_limit \
        --total_machine $total_machine \
        --this_machine $this_machine \
        --device $device \
        --assigned_prefix_len 0 \
        --store_case_metrics
else
    echo "Unsupported run mode: $run_mode" >&2
    exit 2
fi
