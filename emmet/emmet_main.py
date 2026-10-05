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
from util.generate import generate_fast, generate_standard
from util.globals import *
from datetime import datetime
import time

from experiments.last_layer_tensor_capture import (
    capture_last_layer_linear_tensors,
    capture_stage1_next_hidden,
    capture_stage2_next_hidden,
    initialise_last_layer_tensor_capture,
)

from .compute_ks import compute_ks
from .compute_z import compute_z, find_fact_lookup_idx, get_module_input_output_at_words
from .emmet_hparams import EMMETHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_emmet_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: EMMETHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    layer_ks_cache_template: Optional[str] = None,
    motivation_exp: bool = False,
    cache_motivation_fname: Optional[str] = None,
    last_layer_fit_error: bool = False,
    last_layer_fit_output_dir: Optional[str] = None,
    **kwargs,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    distances = {}
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    tensor_capture = initialise_last_layer_tensor_capture(
        last_layer_fit_error, last_layer_fit_output_dir, "EMMET"
    )
    deltas = execute_emmet(
        model,
        tok,
        requests,
        hparams,
        cache_template=cache_template,
        layer_ks_cache_template=layer_ks_cache_template,
        motivation_exp=motivation_exp,
        cache_motivation_fname=cache_motivation_fname,
        last_layer_fit_error=last_layer_fit_error,
        last_layer_fit_output_dir=last_layer_fit_output_dir,
        last_layer_tensor_capture=tensor_capture,
    )

    with torch.no_grad():
        for w_name, (key_mat, val_mat, preservation_distance, new_edit_distance, old_edit_distance, inside_norms) in deltas.items():
            key_mat, val_mat = key_mat.to(model.device), val_mat.to(model.device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            original_weights_norm = torch.norm(w[...]).detach().cpu().item()

            w[...] += upd_matrix.float()

            #saving all distances
            layer = w_name.split('.')[2]
            temp_dict = {
                'preservation_distance': preservation_distance,
                'new_edit_distance': new_edit_distance,
                'old_edit_distance': old_edit_distance,
                'delta_norm': torch.norm(upd_matrix).detach().cpu().item(),
                'new_weights_norm': torch.norm(w[...]).detach().cpu().item(),
                'original_weights_norm': original_weights_norm,
                'inside_norms': inside_norms
            }
            distances[layer] = temp_dict

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

    #return all the objective loss terms, plus the absolute norm of the new weights

    return model, weights_copy


def execute_emmet(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: EMMETHyperParams,
    cache_template: Optional[str] = None,
    layer_ks_cache_template: Optional[str] = None,
    motivation_exp: bool = False,
    cache_motivation_fname: Optional[str] = None,
    last_layer_fit_error: bool = False,
    last_layer_fit_output_dir: Optional[str] = None,
    last_layer_tensor_capture: Optional[Dict[str, Any]] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Save old weights for future restoration. 
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for r_id, request in enumerate(requests):
        print(r_id)
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
        # NOTE - the use of z_layer to calculate cur_zs instead of layer.
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

        #not sure why this is done because layer_ks.size(1) should be same as targets.size(1)
        repeat_factor = (layer_ks.size(1) // targets.size(1))
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # force_recompute = layer != hparams.layers[0]
        cov, preserved_keys = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
            force_recompute=hparams.calculate_objective_value,
        )

        # Compute update in double precision
        #if 'llama' not in model.config._name_or_path.lower():
        layer_ks, targets, cov = (
            layer_ks.double(),
            targets.double(),
            cov.double()
        )

        #add optimization hyper-parameters
        if hparams.mom2_update_weight != 1:
            cov *= hparams.mom2_update_weight

        if hparams.update_norm_lambda != 0:
            cov += hparams.update_norm_lambda * torch.eye(cov.shape[0], dtype=cov.dtype, device = cov.device)


        #####CALCULATING UNIFIED EDITING UPDATES
        pseudo_inverse = False
        C_inv_norm = None
        D_norm = None
        D_inv_norm = None
            
        #calculate C_inv
        C_inv = torch.inverse(cov)
        D = layer_ks.T @ C_inv @ layer_ks

        if D.shape[0] > 1:
            D = D + hparams.emmet_lambda * torch.eye(D.shape[0], dtype=D.dtype, device = D.device)#to counter ill-conditioned D
            
        try:
            D_inv = torch.inverse(D)
        except:
            pseudo_inverse = True
            D_inv = torch.linalg.pinv(D)

        C_inv_norm = torch.norm(C_inv).clone().detach().cpu().item()
        D_norm = torch.norm(D).clone().detach().cpu().item()
        D_inv_norm = torch.norm(D_inv).clone().detach().cpu().item()
        
        adj_k = (D_inv @ layer_ks.T  @ C_inv).T #Only to write it in memit form
        ######FINISHING CALCULATING UNIFIED EDITING UPDATES


        ###Layer distribution code
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T

        ##calculate_norms
        inside_norms = {
            'zs_norm' : torch.mean(torch.norm(zs, dim = 0)).detach().cpu().item(),
            'cur_zs_norm' : torch.mean(torch.norm(cur_zs, dim = 0)).detach().cpu().item(),
            'layer_ks_norm' : torch.mean(torch.norm(layer_ks, dim = 0)).detach().cpu().item(),
            'adj_norm' : torch.mean(torch.norm(adj_k , dim = 0)).detach().cpu().item(),
            'residual_norm' : torch.mean(torch.norm(resid , dim = 0)).detach().cpu().item(),
            'inside_update_norm' : torch.norm(upd_matrix).detach().cpu().item(),
            'pseudo_inverse' : pseudo_inverse,
            'C_inv_norm' : C_inv_norm,
            'D_inv_norm' : D_inv_norm,
            'D_norm' : D_norm,
            'cov' : torch.norm(cov).detach().cpu().item(),
        }

        # Adjust update matrix shape
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        if last_layer_fit_error and layer == hparams.layers[-1]:
            if layer_ks.size(1) != len(requests):
                raise RuntimeError("EMMET final-layer K is not one column per request")
            output_dir = Path(last_layer_fit_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            case_ids = np.asarray([request["case_id"] for request in requests])
            prompt_layer_ks = compute_ks(
                model, tok, requests, hparams, layer, [["{}"]]
            ).T
            if prompt_layer_ks.size(1) != len(requests):
                raise RuntimeError(
                    "EMMET final-layer rewrite-prompt K is not one column per request"
                )
            capture_last_layer_linear_tensors(
                last_layer_tensor_capture,
                model=model,
                tok=tok,
                requests=requests,
                layer=layer,
                solver_keys=layer_ks,
                effective_covariance=cov,
                covariance_definition=(
                    "mom2_update_weight * mom2_covariance + "
                    "update_norm_lambda * I"
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
                raise RuntimeError("EMMET final-layer R contains zero-norm samples")
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
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()

            #calculate distances
            if hparams.calculate_objective_value and preserved_keys is not None:
                preservation_distance, new_edit_distance, old_edit_distance = calculate_distances(weights_copy[weight_name], weights[weight_name][...], layer_ks, zs, preserved_keys)
            else:
                if hparams.calculate_objective_value:
                    print("Skipping objective distance calculation because preserved_keys are unavailable.")
                preservation_distance, new_edit_distance, old_edit_distance = None, None, None
            
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
                preservation_distance, 
                new_edit_distance, 
                old_edit_distance,
                inside_norms
            )

        # Clear GPU memory
        cov = cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x = x.cpu()
            del x
        torch.cuda.empty_cache()

        for x in [C_inv, D, D_inv]:
            x = x.cpu()
            del x
        torch.cuda.empty_cache()


    # import ipdb; ipdb.set_trace()

    # hetionet 1    0.0087
    # hetionet 2    0.1906
    # hetionet 3    0.2754
    # hetionet 4    0.3424
    # hetionet 5    0.3794

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def calculate_distances(original_weights, new_weights, edit_keys, edit_values, preserved_keys):
    preserved_keys = preserved_keys.to(new_weights.device)
    if original_weights.shape[0] != preserved_keys.shape[1]:
        original_weights = original_weights.T
        new_weights = new_weights.T

    W_old_k_old = preserved_keys.double() @ original_weights.double()
    W_hat_k_old = preserved_keys.double() @ new_weights.double()

    W_old_k_edits = original_weights.T.double() @ edit_keys.double()
    W_hat_k_edits = new_weights.T.double() @ edit_keys.double()
    v_edits = edit_values.double()

    preservation_distance = torch.mean(torch.norm(W_hat_k_old - W_old_k_old, dim = 1)).detach().cpu().item()
    new_edit_distance = torch.mean(torch.norm( W_hat_k_edits - v_edits, dim = 0)).detach().cpu().item()
    old_edit_distance = torch.mean(torch.norm( W_old_k_edits - v_edits, dim = 0)).detach().cpu().item()

    preserved_keys = preserved_keys.to("cpu")

    return preservation_distance, new_edit_distance, old_edit_distance

def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    feature_key = (model_name, layer_name, 'preserved_keys')

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE:
        layer_stats_result = layer_stats(
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
        if isinstance(layer_stats_result, tuple):
            stat, preserved_keys = layer_stats_result
        else:
            stat, preserved_keys = layer_stats_result, None

        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
        COV_CACHE[feature_key] = preserved_keys

    return COV_CACHE[key].to(model.device), COV_CACHE[feature_key]



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
    temperature = 0.5
    top_k = 100
    if CONTEXT_TEMPLATES_CACHE is None:
        if "deepseek" in str(model.config._name_or_path).lower():
            initial_tplt = ["The", "Therefore", "Because", "I", "You"]
        else:
            initial_tplt = [
                "The",
                "Therefore",
                "Because",
                "I",
                "You",
                "However",
                "Also",
                "Nevertheless",
                "He",
                "It",
                "Can",
                "Because",
            ]
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_standard(
                    model,
                    initial_tplt,
                    tok,
                    max_new_tokens=length,
                    do_sample=True,
                    temperature=temperature,
                    top_k=top_k,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"temperature:{temperature}")
        print(f"top_k:{top_k}")
        print(f"len(initial_tplt):{len(initial_tplt)}")

    return CONTEXT_TEMPLATES_CACHE
