"""Standalone model-editing entry point for SCRO and bundled baselines."""

import argparse
import gc
import json
import os
import random
from datetime import timedelta
from pathlib import Path
from time import perf_counter, time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from alphaedit import AlphaEditHyperParams, apply_AlphaEdit_to_model
from alphaedit_aligned import (
    AlphaEditAlignedHyperParams,
    apply_AlphaEditAligned_to_model,
)
from baselines.ft import FTHyperParams, apply_ft_to_model
from dsets import (
    HetionetDataset,
    MENDQADataset,
    MultiCounterFactDataset,
    WikirecentDataset,
)
from eamet import EAMETHyperParams, apply_eamet_to_model
from emmet import EMMETHyperParams, apply_emmet_to_model
from memit import MEMITHyperParams, apply_memit_to_model
from memit_merge import MEMIT_MergeHyperParams, apply_memit_merge_to_model
from scro import SCROHyperParams, apply_scro_to_model
from pmet import PMETHyperParams, apply_pmet_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util.globals import DATA_DIR, HPARAMS_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ALG_DICT = {
    "EAMET": (EAMETHyperParams, apply_eamet_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "PMET": (PMETHyperParams, apply_pmet_to_model),
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "ALPHAEDIT": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
    "ALPHAEDIT_ALIGNED": (
        AlphaEditAlignedHyperParams,
        apply_AlphaEditAligned_to_model,
    ),
    "EMMET": (EMMETHyperParams, apply_emmet_to_model),
    "MEMIT_MERGE": (MEMIT_MergeHyperParams, apply_memit_merge_to_model),
    "SCRO": (SCROHyperParams, apply_scro_to_model),
}
ALG_CHOICES = sorted([*ALG_DICT, "MEND"])
LAYER_KS_ALGS = {
    "MEMIT",
    "EAMET",
    "PMET",
    "ALPHAEDIT",
    "ALPHAEDIT_ALIGNED",
    "EMMET",
    "MEMIT_MERGE",
    "SCRO",
}
LAST_LAYER_FIT_ALGS = {
    "MEMIT",
    "EAMET",
    "PMET",
    "ALPHAEDIT",
    "EMMET",
    "MEMIT_MERGE",
    "SCRO",
}
DS_DICT = {
    "counterfact": MultiCounterFactDataset,
    "zsre": MENDQADataset,
    "wikirecent": WikirecentDataset,
    "hetionet": HetionetDataset,
}


def configure_scro_hparams(hparams: SCROHyperParams, args) -> None:
    for name in (
        "joint_spectral_z_op_bound",
        "joint_z_micro_batch_size",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(hparams, name, value)

    hparams.configure_rewrite_loss_mix()
    hparams.configure_z_constraints()
    if hparams.v_num_grad_steps <= 0:
        raise ValueError("v_num_grad_steps must be positive")
    if not np.isfinite(hparams.v_lr) or hparams.v_lr <= 0:
        raise ValueError("v_lr must be finite and positive")
    if not np.isfinite(hparams.v_weight_decay) or hparams.v_weight_decay < 0:
        raise ValueError("v_weight_decay must be finite and non-negative")
    if not np.isfinite(hparams.kl_factor) or hparams.kl_factor < 0:
        raise ValueError("kl_factor must be finite and non-negative")
    if hparams.joint_z_micro_batch_size <= 0:
        raise ValueError("joint_z_micro_batch_size must be positive")
    if len(hparams.mom2_update_weight) != len(hparams.layers):
        raise ValueError("mom2_update_weight must have one value per edit layer")
    if any(
        not np.isfinite(weight) or weight < 0
        for weight in hparams.mom2_update_weight
    ):
        raise ValueError(
            "mom2_update_weight values must be finite and non-negative"
        )


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _begin_edit_resource_profile(device: int) -> Dict:
    """Prepare single-GPU counters immediately before ``apply_algo``."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    baseline_allocated_bytes = torch.cuda.memory_allocated(device)
    baseline_reserved_bytes = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    return {
        "device": int(device),
        "started_at": perf_counter(),
        "baseline_allocated_bytes": int(baseline_allocated_bytes),
        "baseline_reserved_bytes": int(baseline_reserved_bytes),
    }


def _finish_edit_resource_profile(state: Dict) -> Tuple[float, Dict]:
    """Synchronize and collect algorithm-only time and CUDA-memory peaks."""
    device = int(state["device"])
    torch.cuda.synchronize(device)
    execution_time = perf_counter() - float(state["started_at"])

    baseline_allocated_bytes = int(state["baseline_allocated_bytes"])
    baseline_reserved_bytes = int(state["baseline_reserved_bytes"])
    peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    end_allocated_bytes = int(torch.cuda.memory_allocated(device))
    end_reserved_bytes = int(torch.cuda.memory_reserved(device))
    bytes_per_gib = 1024 ** 3
    resource_profile = {
        "measurement_scope": "apply_algo_only",
        "timing_clock": "perf_counter_with_cuda_synchronize",
        "cache_preparation": "gc_collect_and_cuda_empty_cache",
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(device),
        "execution_time_seconds": float(execution_time),
        "baseline_allocated_bytes": baseline_allocated_bytes,
        "baseline_reserved_bytes": baseline_reserved_bytes,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "incremental_peak_allocated_bytes": max(
            0, peak_allocated_bytes - baseline_allocated_bytes
        ),
        "incremental_peak_reserved_bytes": max(
            0, peak_reserved_bytes - baseline_reserved_bytes
        ),
        "end_allocated_bytes": end_allocated_bytes,
        "end_reserved_bytes": end_reserved_bytes,
        "baseline_allocated_gib": float(
            baseline_allocated_bytes / bytes_per_gib
        ),
        "peak_allocated_gib": float(peak_allocated_bytes / bytes_per_gib),
        "incremental_peak_allocated_gib": float(
            max(0, peak_allocated_bytes - baseline_allocated_bytes)
            / bytes_per_gib
        ),
        "peak_reserved_gib": float(peak_reserved_bytes / bytes_per_gib),
        "incremental_peak_reserved_gib": float(
            max(0, peak_reserved_bytes - baseline_reserved_bytes)
            / bytes_per_gib
        ),
    }
    return execution_time, resource_profile


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
) -> None:
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens == 0:
        return

    input_embeddings = model.get_input_embeddings().weight.data
    output_embeddings = model.get_output_embeddings().weight.data
    input_average = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    output_average = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
    input_embeddings[-num_new_tokens:] = input_average
    output_embeddings[-num_new_tokens:] = output_average


def load_base_model(
    model_name: str, device: int
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    print("Instantiating model")
    model_path = f"../models/{model_name}"
    resolved_model_path = PROJECT_ROOT.parent / "models" / model_name
    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"Local model directory does not exist: {resolved_model_path}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir="your_cache_dir"
    ).to(f"cuda:{device}")
    tok = AutoTokenizer.from_pretrained(model_path, cache_dir="your_cache_dir")

    print("Adding special tokens.")
    if "mistral" in str(model.config._name_or_path).lower():
        tok.pad_token = tok.eos_token
    else:
        if tok.pad_token is None:
            smart_tokenizer_and_embedding_resize({"pad_token": "[PAD]"}, tok, model)

        special_tokens = {}
        if model.config.eos_token_id is not None:
            special_tokens["eos_token"] = tok.convert_ids_to_tokens(
                model.config.eos_token_id
            )
        elif tok.eos_token is None:
            special_tokens["eos_token"] = "</s>"

        if model.config.bos_token_id is not None:
            special_tokens["bos_token"] = tok.convert_ids_to_tokens(
                model.config.bos_token_id
            )
        elif tok.bos_token is None:
            special_tokens["bos_token"] = "<s>"

        if model.config.pad_token_id not in [-1, None]:
            special_tokens["unk_token"] = tok.convert_ids_to_tokens(
                model.config.pad_token_id
            )
        elif tok.pad_token_id is not None:
            special_tokens["unk_token"] = tok.convert_ids_to_tokens(tok.pad_token_id)
        elif tok.unk_token is None:
            special_tokens["unk_token"] = "[UNK]"

        tok.add_special_tokens(special_tokens)
        model.resize_token_embeddings(len(tok))
        tok.add_bos_token = "gemma" in str(model.config._name_or_path).lower()

    tok.padding_side = "right"
    print(f"padding side:{tok.padding_side}")
    return model, tok


def split_records_by_machine(
    records: List[Dict], total_machine: int, this_machine: int
) -> List[Dict]:
    if total_machine <= 0:
        raise ValueError("total_machine must be positive")
    if this_machine < 0 or this_machine >= total_machine:
        raise ValueError(
            f"this_machine must be in [0, {total_machine - 1}], got {this_machine}"
        )
    return records[this_machine::total_machine]


def get_algorithm(alg_name: str):
    if alg_name == "MEND":
        from baselines.mend import MENDHyperParams, MendRewriteExecutor

        return MENDHyperParams, MendRewriteExecutor().apply_to_model
    return ALG_DICT[alg_name]


def main(
    alg_name: str,
    model_name: str,
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    results_dir: str,
    total_machine: int = 1,
    this_machine: int = 0,
    device: int = 0,
    use_cache: bool = False,
    only_save_zs: bool = False,
    eval_in_memory: bool = False,
    joint_spectral_z_op_bound: float = None,
    joint_z_micro_batch_size: int = None,
    distributed: bool = False,
    profile_edit_resources: bool = False,
    last_layer_fit_error: bool = False,
) -> None:
    os.chdir(PROJECT_ROOT)
    alg_name = alg_name.upper()
    if alg_name not in ALG_CHOICES:
        raise ValueError(f"Unsupported algorithm: {alg_name}")
    ds_key = "hetionet" if "hetionet" in ds_name else ds_name
    if ds_key not in DS_DICT:
        raise ValueError(f"Unsupported dataset: {ds_name}")
    if dataset_size_limit <= 0:
        raise ValueError("dataset_size_limit must be positive")
    if profile_edit_resources and distributed:
        raise ValueError(
            "Edit resource profiling currently supports single-GPU runs only"
        )
    if profile_edit_resources and (total_machine != 1 or this_machine != 0):
        raise ValueError(
            "Edit resource profiling requires total_machine=1 and this_machine=0"
        )
    if profile_edit_resources and not torch.cuda.is_available():
        raise RuntimeError("Edit resource profiling requires CUDA")
    if last_layer_fit_error and alg_name not in LAST_LAYER_FIT_ALGS:
        raise ValueError(
            "last_layer_fit_error is supported only for "
            f"{sorted(LAST_LAYER_FIT_ALGS)}"
        )
    if last_layer_fit_error and only_save_zs:
        raise ValueError(
            "last_layer_fit_error requires Stage2 and cannot be used with only_save_zs"
        )
    if last_layer_fit_error and profile_edit_resources:
        raise ValueError(
            "last_layer_fit_error cannot be enabled during edit resource profiling"
        )
    split_records_by_machine([], total_machine, this_machine)
    rank = 0
    world_size = 1
    dist_timeout_seconds = None
    if distributed:
        if alg_name != "SCRO":
            raise ValueError("Distributed editing is currently supported only for SCRO")
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed SCRO requires CUDA")
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("Launch distributed SCRO with torchrun")
        device = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(device)
        dist_timeout_seconds = int(os.environ.get("DIST_TIMEOUT_SECONDS", "3600"))
        if dist_timeout_seconds <= 0:
            raise ValueError("DIST_TIMEOUT_SECONDS must be positive")
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=dist_timeout_seconds),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    elif torch.cuda.is_available():
        torch.cuda.set_device(device)
    set_seed(42)
    print(
        "Execution topology",
        {
            "distributed": distributed,
            "rank": rank,
            "world_size": world_size,
            "device": device,
            "dist_timeout_seconds": dist_timeout_seconds,
        },
    )

    run_dir = Path(results_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        print(f"Editing results will be stored at {run_dir}")

    hparams_path = HPARAMS_DIR / alg_name / hparams_fname
    if not hparams_path.exists():
        raise FileNotFoundError(f"Hyperparameter file does not exist: {hparams_path}")
    params_class, apply_algo = get_algorithm(alg_name)
    hparams = params_class.from_json(hparams_path)
    hparams.device = device
    hparams.only_save_zs = only_save_zs
    if alg_name == "SCRO":
        configure_scro_hparams(hparams, argparse.Namespace(**locals()))
    params_path = run_dir / "params.json"

    if rank == 0:
        with params_path.open("w") as handle:
            json.dump(hparams.to_dict(), handle, indent=2)
        print(f"Executing {alg_name} with parameters {hparams}")

    model, tok = load_base_model(model_name, device)
    dataset = DS_DICT[ds_key](
        DATA_DIR,
        tok=tok,
        size=dataset_size_limit,
        trigger=ds_name,
        model_name_config=model.config._name_or_path,
        randomize_editing_sequence=False,
        shuffle_seed=0,
        relation_count=ds_name.split("_")[-1] if "_" in ds_name else None,
    )
    records = [dataset[index] for index in range(len(dataset))]
    if not records:
        raise RuntimeError(f"{ds_name} returned no records")
    assigned_records = records
    if only_save_zs and not distributed:
        assigned_records = split_records_by_machine(
            records, total_machine, this_machine
        )
    if not assigned_records:
        print("No records assigned to this machine. Skipping edit.")
        return
    if alg_name == "SCRO" and len(assigned_records) <= 1:
        raise ValueError("SCRO joint optimization requires at least two requests")

    cache_template = None
    layer_ks_cache_template = None
    if use_cache:
        start_case_id = assigned_records[0]["case_id"]
        end_case_id = assigned_records[-1]["case_id"]
        cache_template = run_dir / "kv_cache/zs/layer_{}_case_{}.npz"
        layer_ks_cache_template = (
            run_dir
            / "kv_cache/layer_ks"
            / f"layer_{{}}_case_id_{start_case_id}_{end_case_id}.npz"
        )

    print(f"Will load z cache from {cache_template}")
    print(f"Will load layer_ks cache from {layer_ks_cache_template}")
    requests = [
        {"case_id": record["case_id"], **record["requested_rewrite"]}
        for record in assigned_records
    ]
    apply_kwargs = {
        "copy": False,
        "return_orig_weights": False,
        "cache_template": cache_template,
    }
    if alg_name in LAYER_KS_ALGS:
        apply_kwargs["layer_ks_cache_template"] = layer_ks_cache_template
    if last_layer_fit_error:
        apply_kwargs["last_layer_fit_error"] = True
        apply_kwargs["last_layer_fit_output_dir"] = str(
            run_dir
            / "interference_cache"
            / f"last_layer_E_ds{dataset_size_limit}"
        )
    if eval_in_memory and ds_key in {"counterfact", "zsre", "hetionet"}:
        # The EasyEdit-style locality metric needs the original model's
        # teacher-forced top-1 outputs.  Capture them before apply_algo mutates
        # the model; the edited-model evaluator will reuse this cache.
        from experiments.eval import ensure_locality_reference_cache

        ensure_locality_reference_cache(
            model=model,
            tok=tok,
            records=records,
            model_name=model_name,
            ds_name=ds_name,
            total_machine=world_size if distributed else total_machine,
            this_machine=rank if distributed else this_machine,
        )
        if distributed:
            dist.barrier()
    resource_profile = None
    if profile_edit_resources:
        profile_state = _begin_edit_resource_profile(device)
    else:
        start = time()
    model, _ = apply_algo(model, tok, requests, hparams, **apply_kwargs)
    if profile_edit_resources:
        execution_time, resource_profile = _finish_edit_resource_profile(
            profile_state
        )
    else:
        execution_time = time() - start
    if rank == 0:
        print("Execution took", execution_time)
        if resource_profile is not None:
            print("Edit algorithm resource profile", resource_profile)
            with (run_dir / "edit-resource-profile.json").open("w") as handle:
                json.dump(resource_profile, handle, indent=2)
        with (run_dir / "edit-metadata.json").open("w") as handle:
            json.dump(
                {
                    "alg_name": alg_name,
                    "model_name": model_name,
                    "ds_name": ds_name,
                    "num_edits": len(requests),
                    "execution_time": execution_time,
                    "profile_edit_resources": bool(profile_edit_resources),
                    "last_layer_fit_error": bool(last_layer_fit_error),
                    "resource_profile": resource_profile,
                    "distributed_world_size": world_size,
                },
                handle,
                indent=2,
            )

    if distributed:
        dist.barrier()
    if only_save_zs:
        return

    if rank == 0:
        edited_model_dir = run_dir / "edited_model" / f"ds{dataset_size_limit}"
        print("save the model to", edited_model_dir)
        edited_model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(edited_model_dir)
        tok.save_pretrained(edited_model_dir)
    if distributed:
        dist.barrier()

    if eval_in_memory:
        from experiments.eval import evaluate_loaded_model

        print(
            "Evaluating serialized edited model in memory. "
            f"Rank {rank}/{world_size}."
        )
        set_seed(42)
        evaluate_loaded_model(
            model=model,
            tok=tok,
            alg_name=alg_name,
            ds_name=ds_name,
            dataset_size_limit=dataset_size_limit,
            results_dir=str(run_dir),
            model_name=model_name,
            assigned_prefix_len=0,
            randomize_editing_sequence=False,
            shuffle_seed=0,
            total_machine=world_size if distributed else total_machine,
            this_machine=rank if distributed else this_machine,
            execution_time=execution_time,
        )
        return

    if distributed and rank != 0:
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alg_name", required=True, choices=ALG_CHOICES)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--hparams_fname", required=True)
    parser.add_argument("--ds_name", required=True)
    parser.add_argument("--dataset_size_limit", type=int, required=True)
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--total_machine", type=int, default=1)
    parser.add_argument("--this_machine", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--only_save_zs", action="store_true")
    parser.add_argument("--eval_in_memory", action="store_true")
    parser.add_argument(
        "--joint_spectral_z_op_bound", type=float,
        help="Hard bound beta: ||Z||_2^2 <= beta.",
    )
    parser.add_argument("--joint_z_micro_batch_size", type=int)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--profile_edit_resources",
        action="store_true",
        help=(
            "Measure apply_algo-only wall time and CUDA allocated/reserved "
            "memory peaks. Supported for single-GPU editing runs only."
        ),
    )
    parser.add_argument(
        "--last_layer_fit_error",
        action="store_true",
        help=(
            "Save final-layer DeltaK and R as deltak.npz and r.npz under "
            "the run directory."
        ),
    )
    args = parser.parse_args()
    try:
        main(**vars(args))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
