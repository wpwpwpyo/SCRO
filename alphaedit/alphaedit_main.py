import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from .alphaedit_hparams import AlphaEditHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

P_loaded = False
cache_c_new = False

def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    layer_ks_cache_template: Optional[str] = None,
    keep_original_weight=False,
    last_layer_fit_error: bool = False,
    last_layer_fit_output_dir: Optional[str] = None,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor]]:
  #-> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    global P, P_loaded, cache_c, cache_c_new

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    # Calculate the null-space projection matrix P
    # Please ensure that you have downloaded "null_space_project.pt" to the easyedit folder beforehand, or get the P by following calculation
    P_loc =  Path(P_DIR)
    model_name = model.config._name_or_path.replace("/", "_")
    layer_name = hparams.layers
    ds_name = hparams.mom2_dataset
    precision = hparams.mom2_dtype
    to_collect = ["mom2"]
    size_suffix = ""
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    p_filename = P_loc / file_extension

    if not os.path.exists(p_filename):
        print(f"The null-space projection matrix P does not exist and now calculate.")
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        del W_out
        for i, layer in enumerate(hparams.layers):
            P[i,:,:] = get_project(model, tok, layer, hparams)
        if p_filename is not None:
            p_filename.parent.mkdir(exist_ok=True, parents=True)
            torch.save(P, p_filename)
        P_loaded = True

    elif P_loaded == False:
        P = torch.load(p_filename)
        P_loaded = True

    # Maintain the global variable cache_c to avoid redundant computations.
    # If this is the first calculation (i.e., cache_c_new == false), then initialize cache_c first
    if not cache_c_new:
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        del W_out
        cache_c_new = True
    
    tensor_capture = initialise_last_layer_tensor_capture(
        last_layer_fit_error, last_layer_fit_output_dir, "ALPHAEDIT"
    )
    deltas = execute_AlphaEdit(
        model,
        tok,
        requests,
        hparams,
        cache_template=cache_template,
        layer_ks_cache_template=layer_ks_cache_template,
        last_layer_fit_error=last_layer_fit_error,
        last_layer_fit_output_dir=last_layer_fit_output_dir,
        last_layer_tensor_capture=tensor_capture,
    )

    with torch.no_grad():
        for w_name, upd_m in deltas.items():
            upd_matrix = upd_m.to(model.device)
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


def execute_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
    layer_ks_cache_template: Optional[str] = None,
    last_layer_fit_error: bool = False,
    last_layer_fit_output_dir: Optional[str] = None,
    last_layer_tensor_capture: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
        requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')
        print(
            f"Executing AlphaEdit algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
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

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, str(request["case_id"])
                )
            )
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
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
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
    for i, layer in enumerate(hparams.layers):
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
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        # if torch.cuda.device_count() == 1:
        upd_matrix = torch.linalg.solve(
            P[i,:,:].to(model.device) @ (layer_ks.to(model.device) @ layer_ks.T.to(model.device) + cache_c[i,:,:].to(model.device)) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device=model.device),
            P[i,:,:].to(model.device) @ layer_ks.to(model.device) @ resid.T.to(model.device)
        )
        # else:
        #     upd_matrix = torch.linalg.solve(
        #         P[i,:,:].to("cuda:1") @ (layer_ks.to("cuda:1") @ layer_ks.T.to("cuda:1") + cache_c[i,:,:].to("cuda:1")) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda:1"),
        #         P[i,:,:].to("cuda:1") @ layer_ks.to("cuda:1") @ resid.T.to("cuda:1")
        #     )

        # Adjust update matrix shape
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        if last_layer_fit_error and layer == hparams.layers[-1]:
            if layer_ks.size(1) != len(requests):
                raise RuntimeError("AlphaEdit final-layer K is not one column per request")
            output_dir = Path(last_layer_fit_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            case_ids = np.asarray([request["case_id"] for request in requests])
            prompt_layer_ks = compute_ks(
                model, tok, requests, hparams, layer, [["{}"]]
            ).T
            if prompt_layer_ks.size(1) != len(requests):
                raise RuntimeError(
                    "AlphaEdit final-layer rewrite-prompt K is not one column per request"
                )
            diagnostic_covariance = get_cov(
                model,
                tok,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples,
                hparams.mom2_dtype,
            ).double()
            effective_covariance = (
                float(hparams.mom2_update_weight) * diagnostic_covariance
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
                    "mom2_update_weight * mom2_covariance (diagnostic C0; "
                    "AlphaEdit's projected solve is unchanged)"
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
                raise RuntimeError("AlphaEdit final-layer R contains zero-norm samples")
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
            # if torch.cuda.device_count() == 1:
            weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
            # else:
            #     weights[weight_name][...] = weights[weight_name] + upd_matrix.float().to("cuda:0")
            deltas[weight_name] = (
                upd_matrix.detach().cpu()
            )
        
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    
    # for i, layer in enumerate(hparams.layers):
    #     layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
    #     cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T


    # import ipdb; ipdb.set_trace()

    # hetionet 1    0.5635
    # hetionet 2    0.6587
    # hetionet 3    0.7399
    # hetionet 4    0.7891
    # hetionet 5    0.8200

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
    hparams=None,
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
    #     return (
    #         torch.inverse(COV_CACHE[key].to("cuda:1")) if inv else COV_CACHE[key].to("cuda:1")
    #     )


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
            "Update matrix computed by AlphaEdit does not match original weight shape. "
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

def get_project(model, tok, layer, hparams):
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
        hparams=hparams
    ).cpu()
    U, S, _ = torch.linalg.svd(cov, full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T
