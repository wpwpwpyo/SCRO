import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import copy
import random
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast, generate_standard
from util.globals import *

from experiments.last_layer_tensor_capture import (
    capture_last_layer_linear_tensors,
    capture_stage1_next_hidden,
    capture_stage2_next_hidden,
    initialise_last_layer_tensor_capture,
)

from .compute_ks import compute_ks
from .compute_z import compute_z, find_fact_lookup_idx, get_module_input_output_at_words
from .hparams import MEMITHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i:i+n]


def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    layer_ks_cache_template: Optional[str] = None,
    motivation_exp: bool=False,
    cache_motivation_fname: Optional[str] = None,
    duplicate_subjects: Optional[Dict[str, List[int]]] = None,
    last_layer_fit_error: bool = False,
    last_layer_fit_output_dir: Optional[str] = None,
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
        last_layer_fit_error, last_layer_fit_output_dir, "MEMIT"
    )
    deltas = execute_memit(model, tok, requests, hparams, 
                           cache_template=cache_template, 
                           layer_ks_cache_template=layer_ks_cache_template,
                           motivation_exp=motivation_exp,
                           cache_motivation_fname=cache_motivation_fname,
                           duplicate_subjects=duplicate_subjects,
                           last_layer_fit_error=last_layer_fit_error,
                           last_layer_fit_output_dir=last_layer_fit_output_dir,
                           last_layer_tensor_capture=tensor_capture,
                           )

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(model.device), val_mat.to(model.device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

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

    return model, weights_copy


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
    layer_ks_cache_template: Optional[str] = None,
    motivation_exp: bool=False,
    cache_motivation_fname: Optional[str] = None,
    duplicate_subjects: Optional[Dict[str, List[int]]] = None,
    last_layer_fit_error: bool = False,
    last_layer_fit_output_dir: Optional[str] = None,
    last_layer_tensor_capture: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    deltas = {}

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    # k is the name of the param, and v is the corresponding value
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer

    z_layer = hparams.layers[-1]
    z_list = []
    mask_token_id_list_all = [] 
    context_templates = get_context_templates(model, tok)

    # assert False, "Finish printing max values for each row in w"
    for re_id, request in enumerate(requests):
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(str(cache_template).format(z_layer, str(request["case_id"])))
            if cache_template is not None
            else None
        )

        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(model.device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            opt_zs = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(opt_zs)

            if cache_fname is not None:
                try:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": opt_zs.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
                except Exception as e:
                    print(f"Error loading cache file due to {e}.")

    zs = torch.stack(z_list, dim=1)
    if getattr(hparams, "only_save_zs", False):
        print("Finished caching zs. Exiting because hparams.only_save_zs is True.")
        sys.exit(0)

    capture_stage1_next_hidden(
        last_layer_tensor_capture,
        model=model,
        tok=tok,
        requests=requests,
        injections=[(hparams.layer_module_tmp.format(z_layer), zs.detach())],
        final_norm_module=hparams.ln_f_module,
        original_rewrite_module=hparams.rewrite_module_tmp.format(
            hparams.layers[-1]
        ),
        fact_token=hparams.fact_token,
        find_fact_lookup_idx=find_fact_lookup_idx,
    )

    # Insert
    edit_layers = hparams.layers

    # edit_layers = hparams.layers
    for i, layer in enumerate(edit_layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks_cache_fname = (
            Path(str(layer_ks_cache_template).format(layer))
            if layer_ks_cache_template is not None
            else None
        )
        layer_ks_loaded = False
        if (
            layer_ks_cache_fname is not None
            and layer_ks_cache_fname.exists()
        ):
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
        print(f"size of layer_ks:{layer_ks.size()}")

        cur_zs_list=[]
        for requests_slice in chunks(requests, 1000):
            temps = [request['prompt'] for request in requests_slice]
            temp_words = [request["subject"] for request in requests_slice]
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=temps,
                words=temp_words,
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T #0 for the module input before layer_module, 1 for the output after layer_module
            cur_zs_list.append(cur_zs)
        cur_zs=torch.cat(cur_zs_list,dim=1)
        targets = zs - cur_zs # targets to be distributed across layers

        # after transpose, layer_ks.size(1) and targets.size(1) means the number
        # of entries
        repeat_factor = (layer_ks.size(1) // targets.size(1))
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        # Load covariance matrix
        force_recompute = False
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
        # if torch.cuda.device_count() == 1:

        layer_ks, targets = (
            layer_ks.double(),
            targets.double(),
        )
        # else:
        #     layer_ks, targets = (
        #         layer_ks.double().to("cuda:1"),
        #         targets.double().to("cuda:1"),
        #     )

        # adj_k = torch.linalg.solve(
        #     hparams.mom2_update_weight * cov.double() + (layer_ks @ layer_ks.T),
        #     layer_ks
        # )
        effective_covariance = hparams.mom2_update_weight[i] * cov.double()
        cov_mat = effective_covariance + (layer_ks @ layer_ks.T)
        start_time = time.time()
        # if torch.cuda.device_count() == 1:
        adj_k = torch.inverse(cov_mat.to("cpu")).to(model.device) \
            @ layer_ks
        # else:
        #     adj_k = torch.inverse(cov_mat.to("cuda:1")).to("cuda:1") \
        #         @ layer_ks
        print(f"computing inverse takes:{time.time()-start_time}")

        resid = targets / (len(edit_layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        if last_layer_fit_error and layer == edit_layers[-1]:
            if layer_ks.size(1) != len(requests):
                raise RuntimeError("MEMIT final-layer K is not one column per request")
            output_dir = Path(last_layer_fit_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            case_ids = np.asarray([request["case_id"] for request in requests])
            prompt_layer_ks = compute_ks(
                model, tok, requests, hparams, layer, [["{}"]]
            ).T
            if prompt_layer_ks.size(1) != len(requests):
                raise RuntimeError(
                    "MEMIT final-layer rewrite-prompt K is not one column per request"
                )
            capture_last_layer_linear_tensors(
                last_layer_tensor_capture,
                model=model,
                tok=tok,
                requests=requests,
                layer=layer,
                solver_keys=layer_ks,
                effective_covariance=effective_covariance,
                covariance_definition=(
                    "mom2_update_weight[layer_index] * mom2_covariance"
                ),
                rewrite_prompt_keys=prompt_layer_ks,
                delta_weight=upd_matrix,
                residual=resid,
                rewrite_module=hparams.rewrite_module_tmp.format(layer),
                fact_token=hparams.fact_token,
                find_fact_lookup_idx=find_fact_lookup_idx,
            )
            deltak = (upd_matrix.float() @ layer_ks.float()).T.detach().cpu().numpy()
            delta_k_prompt = (
                upd_matrix.float() @ prompt_layer_ks.to(upd_matrix.device).float()
            ).T.detach().cpu().numpy()
            r = resid.T.float().detach().cpu().numpy()
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
                raise RuntimeError("MEMIT final-layer R contains zero-norm samples")
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
        # Check for locality failure
        scaled_cov = hparams.mom2_update_weight[i] * cov.double()
        failure_norm = upd_matrix @ scaled_cov
        locality_norm = upd_matrix @ (layer_ks @ layer_ks.T)
        target_key = targets @ layer_ks.T

        print(f"failure norm for {layer}th layer:{torch.norm(failure_norm)}")
        print(f"locality norm for {layer}th layer:{torch.norm(locality_norm)}")
        print(f"target@key norm for {layer}th layer:{torch.norm(target_key)}")
        print(f"layer_ks@layer_ks.T norm:{torch.norm((layer_ks @ layer_ks.T)/layer_ks.size(1))}")
        print(f"upd_matrix norm:{torch.norm(upd_matrix)}")

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            # if torch.cuda.device_count() == 1:
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            # else:
            #     weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to("cuda:0")
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        cov_mat.cpu()
        for x in [layer_ks, cur_zs, targets, scaled_cov, failure_norm, locality_norm, target_key]:
            x.cpu()
            del x
        torch.cuda.empty_cache()


    # import ipdb; ipdb.set_trace()

    # hetionet 1    0.0576
    # hetionet 2    0.3402
    # hetionet 3    0.4723
    # hetionet 4    0.5442
    # hetionet 5    0.5843
    # hetionet 6    

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

    # if torch.cuda.device_count() == 1:  
    return (
        torch.inverse(COV_CACHE[key].to(model.device)) if inv else COV_CACHE[key].to(model.device)
    )
    # else:
    #     try:
    #         return (
    #             torch.inverse(COV_CACHE[key].to("cuda:1")) if inv else COV_CACHE[key].to("cuda:1")
    #         )
    #     except:
    #         return (
    #             torch.inverse(COV_CACHE[key].to("cuda:0")) if inv else COV_CACHE[key].to("cuda:0")
    #         )



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

    print(f"Generating using generate_standard")
    temperature=0.5
    top_k=100
    if CONTEXT_TEMPLATES_CACHE==None:
        if "deepseek" in str(model.config._name_or_path).lower():
            initial_tplt = ["The", "Therefore", "Because", "I", "You"]
        else:
            initial_tplt = ["The", "Therefore", "Because", "I", "You", \
                            "However", "Also", "Nevertheless", "He", "It", \
                        "Can", "Because"]
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_standard(
                model,
                initial_tplt,
                tok,
                max_new_tokens=length,
                do_sample=True,
                temperature=temperature, #0.5
                top_k=top_k #100
            )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
            ]
        print(f"temperature:{temperature}")
        print(f"top_k:{top_k}")
        print(f"len(initial_tplt):{len(initial_tplt)}")

    return CONTEXT_TEMPLATES_CACHE

# def get_context_templates(model, tok):
#     global CONTEXT_TEMPLATES_CACHE

#     print(f"Generating using generate_fast")
#     if CONTEXT_TEMPLATES_CACHE is None:
#         CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
#             [
#                 f.replace("{", " ").replace("}", " ") + ". {}"
#                 for f in generate_fast(
#                     model,
#                     tok,
#                     ["The", "Therefore", "Because", "I", "You"],
#                     n_gen_per_prompt=n_gen // 5,
#                     max_out_len=length,
#                 )
#             ]
#             for length, n_gen in [(10, 5)]  # Be careful about changing this.
#         ]
#         print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

#     return CONTEXT_TEMPLATES_CACHE
