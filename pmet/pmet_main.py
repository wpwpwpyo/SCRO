import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
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

from .compute_ks import compute_ks, compute_ks_parallel
from .compute_zs import (
    compute_zs,
    compute_z,
    find_fact_lookup_idx,
    get_module_input_output_at_words,
)
from .pmet_hparams import PMETHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}
KZ_CACHE= {}

def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i:i+n]

def apply_pmet_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PMETHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    layer_ks_cache_template: Optional[str] = None,
    motivation_exp: bool=False,
    cache_motivation_fname: Optional[str] = None,
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
        last_layer_fit_error, last_layer_fit_output_dir, "PMET"
    )
    deltas = execute_pmet(model, tok, requests, hparams, 
                          cache_template=cache_template,
                          layer_ks_cache_template=layer_ks_cache_template,
                          motivation_exp=motivation_exp,
                          cache_motivation_fname=cache_motivation_fname,
                          last_layer_fit_error=last_layer_fit_error,
                          last_layer_fit_output_dir=last_layer_fit_output_dir,
                          last_layer_tensor_capture=tensor_capture)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items(): #w_name, adj_k, resid
            upd_matrix = upd_matrix.to(model.device)
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

    print(f"\nNew weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_pmet(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: PMETHyperParams,
    cache_template: Optional[str] = None,
    layer_ks_cache_template: Optional[str] = None,
    motivation_exp: bool=False,
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
            f"MEMIT_ATTN request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter( # transformer.h.{}.attn.out_proj
            model, f"{rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
        for rewrite_module_tmp in hparams.rewrite_module_tmps
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    rewrite_module_names = hparams.rewrite_module_tmps

    def get_layer_ks_cache_fname(layer: int, rewrite_module_name: str) -> Optional[Path]:
        if layer_ks_cache_template is None:
            return None

        block_name = "attn" if "attn" in rewrite_module_name else "mlp"
        cache_fname = Path(str(layer_ks_cache_template).format(layer))
        return cache_fname.with_name(f"{cache_fname.stem}_{block_name}{cache_fname.suffix}")

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = dict()

    for rewrite_module_name in rewrite_module_names:
        z_list[rewrite_module_name] = []
    # get zs
    for request in requests:
        # Retrieve k/v pair if already stored in cache
        for rewrite_module_name in rewrite_module_names:
            block_name = "attn" if "attn" in rewrite_module_name else "mlp"

            cache_fname = (
                Path(str(cache_template).format(z_layer, str(request["case_id"]))).with_name(
                    f"layer_{z_layer}_case_{request['case_id']}_{block_name}.npz"
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
                    z_list[rewrite_module_name].append(torch.from_numpy(data["v_star"]).to(model.device))
                    data_loaded = True
                except Exception as e:
                    print(f"Error reading cache file due to {e}. Recomputing...")

            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                if len(rewrite_module_names) == 2:
                    cur_z_attn, cur_z_mlp = compute_zs( 
                            model,
                            tok,
                            request,
                            hparams,
                            z_layer,
                            context_templates,
                    )
                    z_list[rewrite_module_names[0]].append(cur_z_attn if "attn" in rewrite_module_names[0] else cur_z_mlp)
                    z_list[rewrite_module_names[1]].append(cur_z_attn if "attn" in rewrite_module_names[1] else cur_z_mlp)
                    for rewrite_module_name in rewrite_module_names:
                        block_name = "attn" if "attn" in rewrite_module_name else "mlp"
                        cache_fname = (
                            Path(str(cache_template).format(z_layer, str(request["case_id"]))).with_name(
                                f"layer_{z_layer}_case_{request['case_id']}_{block_name}.npz"
                            )
                            if cache_template is not None
                            else None
                        )
                        if cache_fname is not None:
                            cache_fname.parent.mkdir(exist_ok=True, parents=True)
                            if block_name == "attn":
                                np.savez(
                                    cache_fname,
                                    **{
                                        "v_star": cur_z_attn.detach().cpu().numpy(),
                                    },
                                )
                            else:
                                np.savez(
                                    cache_fname,
                                    **{
                                        "v_star": cur_z_mlp.detach().cpu().numpy(),
                                    },
                                )
                            print(f"Cached k/v pair at {cache_fname}")
                else:
                    cur_z_attn, cur_z_mlp = compute_zs( 
                    model,
                    tok,
                    request,
                    hparams,
                    z_layer,
                    context_templates,
                )
                    if "attn" == block_name:
                        cur_z = cur_z_attn
                    else:
                        cur_z = cur_z_mlp
                    z_list[rewrite_module_name].append(cur_z)
                    cache_fname = (
                        Path(str(cache_template).format(z_layer, str(request["case_id"]))).with_name(
                            f"layer_{z_layer}_case_{request['case_id']}_{block_name}.npz"
                        )
                        if cache_template is not None
                        else None
                    )
                    if cache_fname is not None:
                        cache_fname.parent.mkdir(exist_ok=True, parents=True)
                        np.savez(
                            cache_fname,
                            **{
                                "v_star": cur_z.detach().cpu().numpy(),
                            },
                        )
                        print(f"Cached k/v pair at {cache_fname}")
                break

    for k, v in z_list.items():
        z_list[k] = torch.stack(v, dim=1)
    if getattr(hparams, "only_save_zs", False):
        print("Finished caching zs. Exiting because hparams.only_save_zs is True.")
        sys.exit(0)

    capture_stage1_next_hidden(
        last_layer_tensor_capture,
        model=model,
        tok=tok,
        requests=requests,
        injections=[
            (module_name.format(z_layer), values.detach())
            for module_name, values in z_list.items()
        ],
        final_norm_module=hparams.ln_f_module,
        original_rewrite_module=hparams.rewrite_module_tmp.format(
            hparams.layers[-1]
        ),
        fact_token=hparams.fact_token,
        find_fact_lookup_idx=find_fact_lookup_idx,
    )

    # Insert
    edit_layers = hparams.layers

    for i, layer in enumerate(edit_layers):
        print(f"\n\nLAYER {layer}\n") 
        layers_ks = None
        # force_recompute = layer != hparams.layers[0]
        for rewrite_module_name in rewrite_module_names:
            # Get current model activations
            layer_ks_cache_fname = get_layer_ks_cache_fname(layer, rewrite_module_name)
            layer_ks_loaded = False
            if (
                layer_ks_cache_fname is not None
                and layer_ks_cache_fname.exists()
            ):
                try:
                    data = np.load(layer_ks_cache_fname)
                    if "layer_ks" in data:
                        if layers_ks is None:
                            layers_ks = {}
                        layers_ks[rewrite_module_name] = torch.from_numpy(data["layer_ks"]).to(model.device)
                        layer_ks_loaded = True
                except Exception as e:
                    print(f"Error reading layer_ks cache file due to {e}. Recomputing...")

            if layer_ks_loaded:
                pass
            elif 'gpt-j' in model.config._name_or_path and len(rewrite_module_names) == 2:
                computed_layers_ks = compute_ks_parallel(model, tok, requests, hparams, layer, context_templates)  #K eqn 19
                if layers_ks is None:
                    layers_ks = {}
                layers_ks.update(computed_layers_ks)
                if layer_ks_cache_fname is not None:
                    try:
                        for name, value in computed_layers_ks.items():
                            name_cache_fname = get_layer_ks_cache_fname(layer, name)
                            if name_cache_fname is None:
                                continue

                            name_cache_fname.parent.mkdir(exist_ok=True, parents=True)
                            np.savez(
                                name_cache_fname,
                                **{"layer_ks": value.detach().cpu().numpy()},
                            )
                            print(f"Cached layer_ks at {name_cache_fname}")
                    except Exception as e:
                        print(f"Error saving layer_ks cache file due to {e}.")
            else:
                computed_layer_ks = compute_ks(model, tok, requests, hparams, rewrite_module_name, layer, context_templates)
                if layers_ks is None:
                    layers_ks = {}
                layers_ks.update(computed_layer_ks)
                if layer_ks_cache_fname is not None:
                    try:
                        layer_ks_cache_fname.parent.mkdir(exist_ok=True, parents=True)
                        np.savez(
                            layer_ks_cache_fname,
                            **{"layer_ks": layers_ks[rewrite_module_name].detach().cpu().numpy()},
                        )
                        print(f"Cached layer_ks at {layer_ks_cache_fname}")
                    except Exception as e:
                        print(f"Error saving layer_ks cache file due to {e}.")

            print(f"Writing {layers_ks[rewrite_module_name].size(0)} key/value pair(s) into layers")

            cur_zs_list=[]
            for requests_slice in chunks(requests, 1000):
                cur_zs = get_module_input_output_at_words( # hidden states eqn 2
                    model,
                    tok,
                    z_layer,
                    context_templates=[request["prompt"] for request in requests_slice],
                    words=[request["subject"] for request in requests_slice],
                    module_template=rewrite_module_name,
                    fact_token_strategy=hparams.fact_token,
                )[1].T
                cur_zs_list.append(cur_zs)

            cur_zs=torch.cat(cur_zs_list,dim=1)
            targets = z_list[rewrite_module_name]  - cur_zs #z_i - h_i^L

            fit_layer_ks = layers_ks[rewrite_module_name].T
            repeat_factor = (fit_layer_ks.size(1) // targets.size(1))
            weight_name = f"{rewrite_module_name.format(layer)}.weight"

            # if torch.cuda.device_count() == 1:  
            layer_ks, targets = (
                fit_layer_ks.double(),
                targets.double()
            )
            # else:
            #     layer_ks, targets = (
            #         layers_ks[rewrite_module_name].T.double().to("cuda:1"),
            #         targets.double().to("cuda:1")
            #     )
            # Load covariance matrix
            force_recompute = False
            # force_recompute = layer != hparams.layers[0]
            cov = get_cov(
                model,
                tok,
                rewrite_module_name.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples
                if not force_recompute
                else hparams.mom2_n_samples // 10,
                hparams.mom2_dtype,
                force_recompute=force_recompute,
            )

            targets = targets.repeat_interleave(repeat_factor, dim=1) #r
            effective_covariance = hparams.mom2_update_weight[i] * cov.double()
            cov_mat = effective_covariance + (layer_ks @ layer_ks.T)
            resid = targets / np.sqrt((len(hparams.layers) - i))
            # if torch.cuda.device_count() == 1:  
            upd_matrix = resid @ layer_ks.T @ torch.inverse(cov_mat.to("cpu")).to(model.device)
            # else:
            #     upd_matrix =  (targets / np.sqrt((len(hparams.layers) - i ))) @ layer_ks.T @ torch.inverse(cov_mat.to("cpu")).to("cuda:1")
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            if (
                last_layer_fit_error
                and layer == hparams.layers[-1]
                and rewrite_module_name == hparams.rewrite_module_tmps[-1]
            ):
                if layer_ks.size(1) != len(requests):
                    raise RuntimeError("PMET final-layer K is not one column per request")
                output_dir = Path(last_layer_fit_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                case_ids = np.asarray([request["case_id"] for request in requests])
                prompt_layer_ks = compute_ks(
                    model,
                    tok,
                    requests,
                    hparams,
                    rewrite_module_name,
                    layer,
                    [["{}"]],
                )[rewrite_module_name].T
                if prompt_layer_ks.size(1) != len(requests):
                    raise RuntimeError(
                        "PMET final-layer rewrite-prompt K is not one column per request"
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
                    rewrite_module=rewrite_module_name.format(layer),
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
                    raise RuntimeError("PMET final-layer R contains zero-norm samples")
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

            print(weight_name, ":\norig norm", torch.linalg.norm(weights[weight_name]))
            print("upd norm", torch.linalg.norm(upd_matrix))

            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                # if torch.cuda.device_count() == 1:  
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to(model.device)
                # else:
                #     weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to("cuda:0")
                deltas[weight_name] = upd_matrix

            # Clear GPU memory

            for x in [layer_ks, cur_zs, targets]:
                x.cpu()
                del x
            torch.cuda.empty_cache()



    # import ipdb; ipdb.set_trace()

    # hetionet 1    0.0682
    # hetionet 2    0.1726
    # hetionet 3    0.2199
    # hetionet 4    0.2829
    # hetionet 5    0.3119

    # Restore state of original model
    with torch.no_grad():
        for k, _ in weights.items():
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


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
        stat = layer_stats( # download
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
