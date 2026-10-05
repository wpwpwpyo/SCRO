"""SCRO Stage1 optimization and covariance-whitened Stage2 write."""

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import STATS_DIR

from experiments.last_layer_tensor_capture import (
    capture_last_layer_linear_tensors,
    capture_stage1_next_hidden,
    capture_stage2_next_hidden,
    initialise_last_layer_tensor_capture,
)

from .compute_ks import compute_ks
from .compute_z import (
    compute_zs,
    find_fact_lookup_idx,
    get_module_input_output_at_words,
)
from .distributed_utils import (
    all_gather_columns,
    broadcast_from_main_,
    distributed_barrier,
    distributed_enabled,
    is_main_process,
    request_partition,
)
from .scro_hparams import SCROHyperParams


CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _distributed_context_templates(model, tok):
    if not distributed_enabled():
        return get_context_templates(model, tok)
    payload = [
        get_context_templates(model, tok) if is_main_process() else None
    ]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _compute_layer_ks(
    model,
    tok,
    requests,
    hparams,
    layer,
    context_templates,
):
    if not distributed_enabled():
        return compute_ks(
            model, tok, requests, hparams, layer, context_templates
        ).T

    local_start, local_end = request_partition(len(requests))
    local_requests = requests[local_start:local_end]
    if local_requests:
        local_ks = compute_ks(
            model, tok, local_requests, hparams, layer, context_templates
        ).T
    else:
        weight = nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        local_ks = weight.new_empty((weight.shape[1], 0))
    layer_ks = all_gather_columns(local_ks)
    if layer_ks.size(1) != len(requests):
        raise RuntimeError(
            f"Expected {len(requests)} gathered keys, got {layer_ks.size(1)}"
        )
    return layer_ks


def _distributed_residual_spectrum(
    model,
    tok,
    hparams,
    z_layer,
    z_layer_ks,
):
    if not distributed_enabled():
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(z_layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
        )
        basis, values = _compute_scro_residual_spectrum(
            cov,
            z_layer_ks,
            hparams.mom2_update_weight[-1],
        )
        return basis, values

    basis = values = None
    if is_main_process():
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(z_layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
        )
        basis, values = _compute_scro_residual_spectrum(
            cov,
            z_layer_ks,
            hparams.mom2_update_weight[-1],
        )

    rank_tensor = torch.tensor(
        [0 if values is None else values.numel()],
        device=model.device,
        dtype=torch.long,
    )
    dist.broadcast(rank_tensor, src=0)
    retained_rank = int(rank_tensor.item())
    if not is_main_process():
        basis = torch.empty(
            (z_layer_ks.size(1), retained_rank),
            device=model.device,
            dtype=torch.float64,
        )
        values = torch.empty(
            retained_rank, device=model.device, dtype=torch.float64
        )
    broadcast_from_main_(basis)
    broadcast_from_main_(values)
    return basis, values


def _cholesky_with_jitter(
    matrix: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:
    def chol_is_valid(chol: torch.Tensor, info: torch.Tensor) -> bool:
        diagonal = torch.diagonal(chol)
        return (
            int(info.max().item()) == 0
            and bool(torch.isfinite(diagonal).all().item())
            and bool((diagonal > 0).all().item())
        )

    if not bool(torch.isfinite(matrix).all().item()):
        raise RuntimeError("SCRO covariance contains non-finite values")

    chol, info = torch.linalg.cholesky_ex(
        matrix, upper=False, check_errors=False
    )
    if chol_is_valid(chol, info):
        return chol, 0.0, 0.0

    diagonal_scale = float(torch.diagonal(matrix).mean().item())
    if not math.isfinite(diagonal_scale) or diagonal_scale <= 0.0:
        raise RuntimeError(
            "SCRO Cholesky fallback requires a finite positive mean diagonal"
        )

    relative_jitter = 2e-3
    absolute_jitter = relative_jitter * diagonal_scale
    regularized_matrix = matrix.clone()
    torch.diagonal(regularized_matrix).add_(absolute_jitter)
    chol, info = torch.linalg.cholesky_ex(
        regularized_matrix, upper=False, check_errors=False
    )
    if chol_is_valid(chol, info):
        return chol, relative_jitter, absolute_jitter

    diagonal = torch.diagonal(chol)
    raise RuntimeError(
        "SCRO Cholesky factorization remained invalid after ridge: "
        f"info={int(info.max().item())}, "
        f"finite_diag={bool(torch.isfinite(diagonal).all().item())}, "
        f"relative_jitter={relative_jitter}, "
        f"absolute_jitter={absolute_jitter}"
    )


def _stable_svd_rank_mask(
    singular_values: torch.Tensor,
    matrix_shape: torch.Size,
    source_dtype: torch.dtype,
) -> torch.Tensor:
    """Return a numerical-rank mask at the precision of the input keys."""
    source_eps = torch.finfo(source_dtype).eps
    svd_rtol = max(matrix_shape) * source_eps
    svd_tol = singular_values[0] * svd_rtol
    return singular_values > svd_tol


def _compute_scro_adj_k(
    cov: torch.Tensor,
    layer_ks: torch.Tensor,
    mom2_weight: float,
) -> torch.Tensor:
    """Compute the exact SCRO right inverse without materializing C^-1."""
    mom2_weight = float(mom2_weight)
    if not math.isfinite(mom2_weight) or mom2_weight <= 0:
        raise ValueError("mom2_weight must be finite and positive for SCRO")

    source_dtype = layer_ks.dtype
    layer_ks = layer_ks.double()
    if not bool(torch.isfinite(layer_ks).all().item()):
        raise RuntimeError("SCRO key matrix contains non-finite values")

    scaled_cov = cov.to(device=layer_ks.device, dtype=torch.float64) * mom2_weight
    scaled_cov = 0.5 * (scaled_cov + scaled_cov.T)
    if not bool(torch.isfinite(scaled_cov).all().item()):
        raise RuntimeError("SCRO covariance contains non-finite values")
    chol, _, _ = _cholesky_with_jitter(scaled_cov)

    whitened_ks = torch.linalg.solve_triangular(
        chol, layer_ks, upper=False
    )
    u, singular_values, vh = torch.linalg.svd(
        whitened_ks, full_matrices=False
    )
    if singular_values.numel() == 0 or singular_values[0].item() <= 0:
        raise RuntimeError("SCRO whitened key matrix has rank zero")

    retained = _stable_svd_rank_mask(
        singular_values, whitened_ks.shape, source_dtype
    )
    rank = int(retained.sum().item())
    if rank <= 0:
        raise RuntimeError("SCRO stable solver discarded every singular direction")

    pinv_x_transpose = (
        u[:, retained] / singular_values[retained].unsqueeze(0)
    ) @ vh[retained, :]
    return torch.linalg.solve_triangular(
        chol.T, pinv_x_transpose, upper=True
    )


def _compute_scro_residual_spectrum(
    cov: torch.Tensor,
    layer_ks: torch.Tensor,
    mom2_weight: float,
) -> Tuple:
    """Return every singular direction above the numerical-rank tolerance."""
    source_dtype = layer_ks.dtype
    layer_ks = layer_ks.double()
    scaled_cov = cov.to(device=layer_ks.device, dtype=torch.float64)
    scaled_cov = 0.5 * (scaled_cov + scaled_cov.T) * float(mom2_weight)
    chol, _, _ = _cholesky_with_jitter(scaled_cov)
    whitened_ks = torch.linalg.solve_triangular(
        chol, layer_ks, upper=False
    )
    if not bool(torch.isfinite(whitened_ks).all().item()):
        raise RuntimeError(
            "SCRO whitening produced non-finite values after Cholesky solve"
        )
    u, singular_values, vh = torch.linalg.svd(
        whitened_ks, full_matrices=False
    )
    retained = _stable_svd_rank_mask(
        singular_values, whitened_ks.shape, source_dtype
    )
    rank = int(retained.sum().item())
    if rank <= 0:
        raise RuntimeError("SCRO residual spectrum has rank zero")
    basis = vh[retained].T.contiguous()
    retained_values = singular_values[retained].contiguous()
    return basis, retained_values


def _repeat_residual_for_keys(
    residual: torch.Tensor, layer_ks: torch.Tensor
) -> torch.Tensor:
    if layer_ks.size(1) % residual.size(1) != 0:
        raise RuntimeError("Cannot align residual columns with layer keys")
    repeat_factor = layer_ks.size(1) // residual.size(1)
    return residual.repeat_interleave(repeat_factor, dim=1).double()


def _load_cached_layer_ks(
    cache_fname: Optional[Path], device: torch.device, cache_key: str
) -> Optional[torch.Tensor]:
    if cache_fname is None or not cache_fname.exists():
        return None
    try:
        with np.load(cache_fname) as data:
            return torch.from_numpy(data[cache_key]).to(device)
    except Exception as error:
        print(f"Error reading layer_ks cache file due to {error}. Recomputing...")
        return None


def _save_cached_layer_ks(
    cache_fname: Optional[Path], layer_ks: torch.Tensor, cache_key: str
) -> None:
    if cache_fname is None:
        return
    try:
        cache_fname.parent.mkdir(exist_ok=True, parents=True)
        np.savez(cache_fname, **{cache_key: layer_ks.detach().cpu().numpy()})
        print(f"Cached layer_ks at {cache_fname}")
    except Exception as error:
        print(f"Error saving layer_ks cache file due to {error}.")


def _get_layer_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: SCROHyperParams,
    layer: int,
    context_templates: List[List[str]],
    layer_ks_cache_template: Optional[str],
) -> torch.Tensor:
    cache_fname = (
        Path(str(layer_ks_cache_template).format(layer))
        if layer_ks_cache_template is not None
        else None
    )
    cached = _load_cached_layer_ks(cache_fname, model.device, "layer_ks")
    if cached is not None:
        return cached
    layer_ks = _compute_layer_ks(
        model, tok, requests, hparams, layer, context_templates
    )
    if is_main_process():
        _save_cached_layer_ks(cache_fname, layer_ks, "layer_ks")
    distributed_barrier()
    return layer_ks


def _compute_current_zs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: SCROHyperParams,
    z_layer: int,
) -> torch.Tensor:
    if distributed_enabled():
        local_start, local_end = request_partition(len(requests))
        local_requests = requests[local_start:local_end]
    else:
        local_requests = requests
    current_zs = []
    for request_slice in chunks(local_requests, 1000):
        outputs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in request_slice],
            words=[request["subject"] for request in request_slice],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        current_zs.append(outputs)
    if current_zs:
        local_zs = torch.cat(current_zs, dim=1)
    else:
        local_zs = next(model.parameters()).new_empty(
            (model.config.hidden_size, 0)
        )
    gathered = all_gather_columns(local_zs)
    if gathered.size(1) != len(requests):
        raise RuntimeError(
            f"Expected {len(requests)} gathered z columns, got {gathered.size(1)}"
        )
    return gathered


def _joint_cache_signature(hparams: SCROHyperParams) -> str:
    fields = {
        "method": "SCRO-spectral-scaled-hard-op-full-rank-v2",
        "layers": ",".join(map(str, hparams.layers)),
        "steps": hparams.v_num_grad_steps,
        "lr": hparams.v_lr,
        "v_weight_decay": hparams.v_weight_decay,
        "kl_factor": hparams.kl_factor,
        "spectral_z_op_bound": hparams.joint_spectral_z_op_bound,
        "micro_batch_size": hparams.joint_z_micro_batch_size,
        "rewrite_bare_loss_alpha": hparams.joint_rewrite_bare_loss_alpha,
    }
    return "|".join(f"{name}={value}" for name, value in fields.items())


def apply_scro_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: SCROHyperParams,
    copy: bool = False,
    return_orig_weights: bool = False,
    layer_ks_cache_template: Optional[str] = None,
    last_layer_fit_error: bool = False,
    last_layer_fit_output_dir: Optional[str] = None,
    **_unused,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if copy:
        model = deepcopy(model)
    tensor_capture = initialise_last_layer_tensor_capture(
        last_layer_fit_error and is_main_process(),
        last_layer_fit_output_dir,
        "SCRO",
    )
    deltas = execute_scro(
        model,
        tok,
        requests,
        hparams,
        layer_ks_cache_template=layer_ks_cache_template,
        last_layer_fit_error=last_layer_fit_error,
        last_layer_fit_output_dir=last_layer_fit_output_dir,
        last_layer_tensor_capture=tensor_capture,
    )

    weights_copy = {}
    with torch.no_grad():
        for weight_name, (key_matrix, value_matrix) in deltas.items():
            key_matrix = key_matrix.to(model.device)
            value_matrix = value_matrix.to(model.device)
            update = upd_matrix_match_shape(
                key_matrix @ value_matrix.T,
                nethook.get_parameter(model, weight_name).shape,
            )
            weight = nethook.get_parameter(model, weight_name)
            if return_orig_weights:
                weights_copy[weight_name] = weight.detach().clone()
            weight.add_(update.float())

    inserted_weight_names = list(deltas.keys())
    if distributed_enabled() and not hparams.only_save_zs:
        inserted_weight_names = [
            f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            for layer in hparams.layers
        ]
        with torch.no_grad():
            for weight_name in inserted_weight_names:
                weight = nethook.get_parameter(model, weight_name)
                dist.broadcast(weight.data, src=0)
        distributed_barrier()
        print(
            f"Rank {dist.get_rank()} synchronized edited weights "
            f"{inserted_weight_names}"
        )

    if last_layer_fit_error:
        capture_stage2_next_hidden(
            tensor_capture,
            model=model,
            tok=tok,
            requests=requests,
            final_norm_module=hparams.ln_f_module,
            fact_token=hparams.fact_token,
            find_fact_lookup_idx=find_fact_lookup_idx,
        )
        distributed_barrier()

    print(f"New weights successfully inserted into {inserted_weight_names}")
    return model, weights_copy


def execute_scro(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: SCROHyperParams,
    layer_ks_cache_template: Optional[str] = None,
    last_layer_fit_error: bool = False,
    last_layer_fit_output_dir: Optional[str] = None,
    last_layer_tensor_capture: Optional[Dict[str, Any]] = None,
    **_unused,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    if len(requests) <= 1:
        raise ValueError("SCRO requires joint optimization of at least two requests")
    if not hparams.layers:
        raise ValueError("At least one edit layer is required")
    if len(hparams.mom2_update_weight) != len(hparams.layers):
        raise ValueError("mom2_update_weight must have one value per edit layer")
    hparams.configure_z_constraints()

    requests = deepcopy(requests)
    for request in requests:
        if not request["target_new"]["str"].startswith(" "):
            request["target_new"]["str"] = " " + request["target_new"]["str"]

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    weights_copy = {name: weight.detach().clone() for name, weight in weights.items()}
    context_templates = _distributed_context_templates(model, tok)
    z_layer = hparams.layers[-1]

    z_layer_ks_cache = (
        Path(
            str(layer_ks_cache_template)
            .replace("layer_{}", "z_layer_ks")
            .format(z_layer)
        )
        if layer_ks_cache_template is not None
        else None
    )
    z_layer_ks = _load_cached_layer_ks(
        z_layer_ks_cache, model.device, "z_layer_ks"
    )
    if z_layer_ks is None:
        z_layer_ks = _compute_layer_ks(
            model, tok, requests, hparams, z_layer, context_templates
        )
        if is_main_process():
            _save_cached_layer_ks(
                z_layer_ks_cache, z_layer_ks, "z_layer_ks"
            )
        distributed_barrier()

    V_r, spectral_values = _distributed_residual_spectrum(
        model, tok, hparams, z_layer, z_layer_ks
    )

    joint_cache_template = (
        Path(str(layer_ks_cache_template).replace("layer_ks/", "zs_joint/"))
        if layer_ks_cache_template is not None
        else None
    )
    joint_cache = (
        Path(str(joint_cache_template).format(z_layer))
        if joint_cache_template is not None
        else None
    )
    signature = _joint_cache_signature(hparams)
    zs = delta_matrix = zs_init = None
    if joint_cache is not None and joint_cache.exists():
        try:
            with np.load(joint_cache) as data:
                cached_signature = str(data["joint_cache_signature"].item())
                if cached_signature != signature:
                    raise ValueError("joint cache signature mismatch")
                zs = torch.from_numpy(data["zs"]).to(model.device)
                delta_matrix = torch.from_numpy(data["delta_matrix"]).to(model.device)
                zs_init = torch.from_numpy(data["zs_init"]).to(model.device)
            if is_main_process():
                print(f"Loaded SCRO joint cache from {joint_cache}")
        except Exception as error:
            if is_main_process():
                print(f"Error reading joint cache due to {error}. Recomputing...")
            zs = delta_matrix = zs_init = None

    if zs is None:
        zs, delta_matrix, zs_init = compute_zs(
            model,
            tok,
            requests,
            hparams,
            z_layer,
            context_templates,
            V_r=V_r,
            spectral_values=spectral_values,
        )
        if joint_cache is not None and is_main_process():
            joint_cache.parent.mkdir(exist_ok=True, parents=True)
            np.savez(
                joint_cache,
                zs=zs.detach().cpu().numpy(),
                delta_matrix=delta_matrix.detach().cpu().numpy(),
                zs_init=zs_init.detach().cpu().numpy(),
                joint_cache_signature=np.asarray(signature),
            )
            print(f"Cached SCRO joint residual at {joint_cache}")
        distributed_barrier()

    if hparams.only_save_zs:
        if is_main_process():
            print("Finished caching joint zs. Exiting before Stage2.")
        distributed_barrier()
        return {}

    if last_layer_fit_error:
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
        distributed_barrier()

    deltas = {}
    for layer_index, layer in enumerate(hparams.layers):
        layer_ks = _get_layer_ks(
            model,
            tok,
            requests,
            hparams,
            layer,
            context_templates,
            layer_ks_cache_template,
        )
        current_zs = _compute_current_zs(
            model, tok, requests, hparams, z_layer
        )
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        if is_main_process():
            targets = _repeat_residual_for_keys(zs - current_zs, layer_ks)
            cov = get_cov(
                model,
                tok,
                hparams.rewrite_module_tmp.format(layer),
                hparams.mom2_dataset,
                hparams.mom2_n_samples,
                hparams.mom2_dtype,
            )
            targets = targets.double()
            cov = cov.double()
            resid = targets / (len(hparams.layers) - layer_index)
            adj_k = _compute_scro_adj_k(
                cov,
                layer_ks,
                hparams.mom2_update_weight[layer_index],
            )
            update = resid @ adj_k.T
            delta_key_factor = adj_k
            delta_value_factor = resid
            layer_ks = layer_ks.double()

            update = upd_matrix_match_shape(
                update, weights[weight_name].shape
            )
            if last_layer_fit_error and layer == hparams.layers[-1]:
                if layer_ks.size(1) != len(requests):
                    raise RuntimeError("SCRO final-layer K is not one column per request")
                output_dir = Path(last_layer_fit_output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                case_ids = np.asarray([request["case_id"] for request in requests])
                prompt_layer_ks = compute_ks(
                    model, tok, requests, hparams, layer, [["{}"]]
                ).T
                if prompt_layer_ks.size(1) != len(requests):
                    raise RuntimeError(
                        "SCRO final-layer rewrite-prompt K is not one column per request"
                    )
                capture_last_layer_linear_tensors(
                    last_layer_tensor_capture,
                    model=model,
                    tok=tok,
                    requests=requests,
                    layer=layer,
                    solver_keys=layer_ks,
                    effective_covariance=(
                        hparams.mom2_update_weight[layer_index] * cov
                    ),
                    covariance_definition=(
                        "mom2_update_weight[layer_index] * mom2_covariance"
                    ),
                    rewrite_prompt_keys=prompt_layer_ks,
                    delta_weight=update,
                    residual=resid,
                    rewrite_module=hparams.rewrite_module_tmp.format(layer),
                    fact_token=hparams.fact_token,
                    find_fact_lookup_idx=find_fact_lookup_idx,
                )
                deltak = (update.float() @ layer_ks.float()).T.detach().cpu().numpy()
                delta_k_prompt = (
                    update.float() @ prompt_layer_ks.to(update.device).float()
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
                    raise RuntimeError("SCRO final-layer R contains zero-norm samples")
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
            deltas[weight_name] = (
                delta_key_factor.detach().cpu(),
                delta_value_factor.detach().cpu(),
            )
        else:
            update = torch.empty(
                weights[weight_name].shape,
                device=model.device,
                dtype=torch.float64,
            )
        broadcast_from_main_(update)

        with torch.no_grad():
            weights[weight_name].copy_(weights_copy[weight_name] + update.float())
        torch.cuda.empty_cache()

    with torch.no_grad():
        for weight_name, weight in weights.items():
            weight.copy_(weights_copy[weight_name])

    if is_main_process():
        print(f"Deltas successfully computed for {list(weights.keys())}")
    distributed_barrier()
    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: int,
    mom2_dtype: str,
) -> torch.Tensor:
    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)
    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=False,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
    return COV_CACHE[key].to(model.device)


def upd_matrix_match_shape(
    matrix: torch.Tensor, shape: torch.Size
) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError(
        "SCRO update matrix does not match the edited weight shape"
    )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE
    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                generated.replace("{", " ").replace("}", " ") + ". {}"
                for generated in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]
        ]
        print("Generating context templates using generate_fast")
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")
    return CONTEXT_TEMPLATES_CACHE
