import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from experiments.last_layer_tensor_capture import (
    capture_last_layer_linear_tensors,
    capture_stage1_next_hidden,
    capture_stage2_next_hidden,
    initialise_last_layer_tensor_capture,
)

from .compute_ks import compute_ks
from .compute_z import compute_z, find_fact_lookup_idx, get_module_input_output_at_words
from .memit_hparams import MEMIT_MergeHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def get_cluster_hash(case_ids: List[int]) -> str:
    case_id_str = ",".join(str(x) for x in sorted(map(int, case_ids)))
    return hashlib.sha256(case_id_str.encode("utf-8")).hexdigest()


def apply_memit_merge_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMIT_MergeHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    layer_ks_cache_template: Optional[str] = None,
    keep_original_weight=False,
    last_layer_fit_error: bool = False,
    last_layer_fit_output_dir: Optional[str] = None,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)
    tensor_capture = initialise_last_layer_tensor_capture(
        last_layer_fit_error, last_layer_fit_output_dir, "MEMIT_MERGE"
    )
    deltas = execute_memit(
        model,
        tok,
        requests,
        hparams,
        cache_template=cache_template,
        layer_ks_cache_template=layer_ks_cache_template,
        last_layer_fit_error=last_layer_fit_error,
        last_layer_fit_output_dir=last_layer_fit_output_dir,
        last_layer_tensor_capture=tensor_capture,
        **kwargs,
    )


    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(model.device), val_mat.to(model.device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float().to(w.device)

    capture_stage2_next_hidden(
        tensor_capture,
        model=model,
        tok=tok,
        requests=requests,
        final_norm_module=hparams.ln_f_module,
        fact_token=hparams.fact_token,
        find_fact_lookup_idx=find_fact_lookup_idx,
    )

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMIT_MergeHyperParams,
    cache_template: Optional[str] = None,
    layer_ks_cache_template: Optional[str] = None,
    last_layer_fit_error: bool = False,
    last_layer_fit_output_dir: Optional[str] = None,
    last_layer_tensor_capture: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        target_new = request["target_new"]["str"] if isinstance(request["target_new"], dict) else request["target_new"]
        if target_new[0] != " ":
            # Space required for correct tokenization
            target_new = " " + target_new
        requests[i]["target_new"] = target_new

        if '{}' not in request['prompt']:
            # try:
            assert request['subject'] in request['prompt'] or \
                print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
            # except
            # if request['subject'] not in request['prompt']:
            #     print()
            requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}', 1)
    all_requests = deepcopy(requests)

    # Print first 10 requests for debugging purposes
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []
    z_by_subject = {}
    # group requests by same subject
    group_requests = defaultdict(list)
    for request in requests:
        group_requests[request['subject']].append(request)
    
    # requests = sorted(requests, key=lambda x: x['subject'])
    BALANCE_DEBUGGING = True # Found that True works better in practice
    if "balance" in kwargs:
        BALANCE_DEBUGGING = kwargs["balance"] # Found that True works better in practice
    print('BALANCE_DEBUGGING', BALANCE_DEBUGGING)

    for r_subject, request_group in group_requests.items():
        case_ids = [request["case_id"] for request in request_group]
        cluster_hash = get_cluster_hash(case_ids)
        cache_fname = (
            Path(str(cache_template).format(z_layer, cluster_hash))
            if cache_template is not None
            else None
        )

        data_loaded = False
        cur_z = None
        if cache_fname is not None and cache_fname.exists():
            try:
                data = np.load(cache_fname)
                cur_z = torch.from_numpy(data["v_star"]).to(model.device)
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request_group,
                hparams,
                z_layer,
                context_templates,
            )

            if cache_fname is not None:
                try:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": cur_z.detach().cpu().numpy(),
                            "case_ids": np.array(sorted(case_ids), dtype=np.int64),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
                except Exception as e:
                    print(f"Error saving cache file due to {e}.")

        z_by_subject[r_subject] = cur_z
        if BALANCE_DEBUGGING:
            z_list += [cur_z]
        else:
            z_list += [cur_z] * len(request_group)
    zs = torch.stack(z_list, dim=1)
    if getattr(hparams, "only_save_zs", False):
        print("Finished caching zs. Exiting because hparams.only_save_zs is True.")
        sys.exit(0)

    full_zs_for_capture = torch.stack(
        [z_by_subject[request["subject"]] for request in all_requests], dim=1
    )
    capture_stage1_next_hidden(
        last_layer_tensor_capture,
        model=model,
        tok=tok,
        requests=all_requests,
        injections=[
            (hparams.layer_module_tmp.format(z_layer), full_zs_for_capture.detach())
        ],
        final_norm_module=hparams.ln_f_module,
        original_rewrite_module=hparams.rewrite_module_tmp.format(
            hparams.layers[-1]
        ),
        fact_token=hparams.fact_token,
        find_fact_lookup_idx=find_fact_lookup_idx,
    )
    
    if BALANCE_DEBUGGING:
        # Use only the first request from each group for debugging
        # This results in only one k value without repetition
        # But when same-subject keys are different, only the first edit's key value is used
        requests = [request_group[0] for request_group in group_requests.values()]
    layer_ks_cluster_hash = get_cluster_hash([request["case_id"] for request in requests])
    


    # Insert
    # Here we can use the same zs to loop multiple times, or reduce the number of loops. For now, use the simplest approach with multiple repeated k and m
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks_cache_fname = (
            Path(str(layer_ks_cache_template).format(layer, layer_ks_cluster_hash))
            if layer_ks_cache_template is not None
            else None
        )
        layer_ks_loaded = False
        if layer_ks_cache_fname is not None and layer_ks_cache_fname.exists():
            try:
                data = np.load(layer_ks_cache_fname)
                layer_ks = torch.from_numpy(data["layer_ks"]).to(model.device)
                layer_ks_loaded = True
            except Exception as e:
                print(f"Error reading layer_ks cache file due to {e}. Recomputing...")

        if not layer_ks_loaded:
            layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
            if layer_ks_cache_fname is not None:
                try:
                    layer_ks_cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        layer_ks_cache_fname,
                        **{
                            "layer_ks": layer_ks.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached layer_ks at {layer_ks_cache_fname}")
                except Exception as e:
                    print(f"Error saving layer_ks cache file due to {e}.")
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
            track='out'
        ).T
        targets = zs - cur_zs.to(zs.device) # Difference between theoretical ideal output and current layer output
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        # What is this step doing? Generally MEMIT target count should equal requests count, i.e., key/value pair count
        repeat_factor = (layer_ks.size(1) // targets.size(1))
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        )

        # Compute update in double precision
        layer_ks, targets = (
            layer_ks.double().to(cov.device),
            targets.double().to(cov.device), # RK
        )
        # (C0 + KKT)x = K <-> x=(C0+KKT)^-1 K <-> x.T = KT (C0+KKT)^-1.T = KT(C0.T+KKT)^-1
        # C0 = k0k0T, s.t. C0.T = C0
        effective_covariance = hparams.mom2_update_weight * cov.double()
        adj_k = torch.linalg.solve(
            effective_covariance + layer_ks @ layer_ks.T, # KK^T * P 
            layer_ks,
        )
        # The norm of (C0+KKT)^-1 is very small, only 0.5765, despite both C0 and KKT having large norms
        # torch.linalg.norm(torch.linalg.solve(hparams.mom2_update_weight
        # * cov.double() + layer_ks @ layer_ks.T, 
        # torch.eye(layer_ks.shape[0], dtype=torch.double).to(f"cuda:{hparams.device}")))
        
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        if last_layer_fit_error and layer == hparams.layers[-1]:
            full_layer_ks = compute_ks(
                model, tok, all_requests, hparams, layer, context_templates
            ).T
            full_current_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in all_requests],
                words=[request["subject"] for request in all_requests],
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
                track="out",
            ).T
            full_zs = torch.stack(
                [z_by_subject[request["subject"]] for request in all_requests],
                dim=1,
            )
            full_resid = (full_zs - full_current_zs.to(full_zs.device)) / (
                len(hparams.layers) - i
            )
            if full_layer_ks.size(1) != len(all_requests):
                raise RuntimeError(
                    "MEMIT-MERGE final-layer K is not one column per original request"
                )
            output_dir = Path(last_layer_fit_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            case_ids = np.asarray(
                [request["case_id"] for request in all_requests]
            )
            full_prompt_layer_ks = compute_ks(
                model, tok, all_requests, hparams, layer, [["{}"]]
            ).T
            if full_prompt_layer_ks.size(1) != len(all_requests):
                raise RuntimeError(
                    "MEMIT-MERGE final-layer rewrite-prompt K is not one column "
                    "per original request"
                )
            capture_last_layer_linear_tensors(
                last_layer_tensor_capture,
                model=model,
                tok=tok,
                requests=all_requests,
                layer=layer,
                solver_keys=full_layer_ks,
                effective_covariance=effective_covariance,
                covariance_definition="mom2_update_weight * mom2_covariance",
                rewrite_prompt_keys=full_prompt_layer_ks,
                delta_weight=upd_matrix,
                residual=full_resid,
                rewrite_module=hparams.rewrite_module_tmp.format(layer),
                fact_token=hparams.fact_token,
                find_fact_lookup_idx=find_fact_lookup_idx,
            )
            deltak = (
                upd_matrix.float() @ full_layer_ks.to(upd_matrix.device).float()
            ).T.detach().cpu().numpy()
            delta_k_prompt = (
                upd_matrix.float()
                @ full_prompt_layer_ks.to(upd_matrix.device).float()
            ).T.detach().cpu().numpy()
            r = full_resid.T.float().detach().cpu().numpy()
            np.savez_compressed(
                output_dir / "deltak.npz",
                deltak=deltak,
                case_ids=case_ids,
                layer=np.int64(layer),
            )
            np.savez_compressed(
                output_dir / "delta_k_prompt.npz",
                delta_k_prompt=delta_k_prompt,
                case_ids=case_ids,
                layer=np.int64(layer),
            )
            np.savez_compressed(
                output_dir / "r.npz",
                r=r,
                case_ids=case_ids,
                layer=np.int64(layer),
            )
            r64 = r.astype(np.float64, copy=False)
            r_norm = np.linalg.norm(r64, axis=1)
            if np.any(r_norm == 0):
                raise RuntimeError("MEMIT-MERGE final-layer R contains zero-norm samples")
            mean_relative_l2_error = float(
                np.mean(
                    np.linalg.norm(
                        deltak.astype(np.float64, copy=False) - r64, axis=1
                    )
                    / r_norm
                )
            )
            rewrite_prompt_mean_relative_l2_error = float(
                np.mean(
                    np.linalg.norm(
                        delta_k_prompt.astype(np.float64, copy=False) - r64,
                        axis=1,
                    )
                    / r_norm
                )
            )
            fit_summary = {
                "mean_per_case_relative_l2_error": mean_relative_l2_error,
                "rewrite_prompt_mean_per_case_relative_l2_error": (
                    rewrite_prompt_mean_relative_l2_error
                ),
                "num_cases": int(r.shape[0]),
                "layer": int(layer),
                "formula": "mean_i ||Delta k_i - r_i||_2 / ||r_i||_2",
                "rewrite_prompt_formula": (
                    "mean_i ||Delta k_i^(rewrite prompt) - r_i||_2 / ||r_i||_2"
                ),
            }
            print("Last-layer fit error", fit_summary)
            with (output_dir / "mean_relative_l2_error.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(fit_summary, handle, indent=2)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to(weights_copy[weight_name].device)
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # import ipdb; ipdb.set_trace()

    # hetionet 1    0.0570
    # hetionet 2    0.4570
    # hetionet 3    0.5891
    # hetionet 4    0.6638
    # hetionet 5    0.6998

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(model.device)) if inv else COV_CACHE[key].to(model.device)
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
