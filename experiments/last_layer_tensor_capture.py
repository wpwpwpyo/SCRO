"""Raw last-layer tensor capture shared by model-editing algorithms.

The helpers in this module run only when ``last_layer_fit_error`` is enabled.
They deliberately save raw tensors instead of deriving paper metrics so that
all methods are compared later with exactly the same offline implementation.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from util import nethook


DEFAULT_CHUNK_CASES = 64
KEY_RANK_REFERENCE_DTYPE = torch.float32
KEY_SPECTRUM_RELATIVE_JITTER = 2e-3


def _metric_matrix(value: Any, name: str) -> torch.Tensor:
    """Return a finite sample-major matrix on CPU in float64."""
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be a 2-D matrix; got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains non-finite values")
    return tensor


def compute_epsilon_fit(
    delta_k: Any,
    residual: Any,
) -> Tuple[float, np.ndarray]:
    """Compute mean_i ||Delta k_i-r_i||_2 / ||r_i||_2.

    Both inputs are sample-major matrices with shape ``[N, d_out]``.  The
    function deliberately normalizes each request before averaging, so large
    residuals cannot dominate the reported value.
    """
    delta_k_matrix = _metric_matrix(delta_k, "delta_k")
    residual_matrix = _metric_matrix(residual, "residual")
    if delta_k_matrix.shape != residual_matrix.shape:
        raise ValueError(
            "delta_k and residual must have identical shapes; "
            f"got {tuple(delta_k_matrix.shape)} and "
            f"{tuple(residual_matrix.shape)}"
        )
    if delta_k_matrix.shape[0] == 0:
        raise ValueError("epsilon_fit requires at least one sample")

    denominators = torch.linalg.vector_norm(residual_matrix, dim=1)
    if bool((denominators <= 0).any().item()):
        bad = torch.nonzero(denominators <= 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"epsilon_fit has zero-norm residuals at samples {bad}")
    per_case = (
        torch.linalg.vector_norm(delta_k_matrix - residual_matrix, dim=1)
        / denominators
    )
    return float(per_case.mean().item()), per_case.numpy()


def compute_epsilon_sp(
    delta_h_off: Any,
    w0_h_off: Any,
    case_offsets: Any,
) -> Tuple[float, np.ndarray]:
    """Compute mean_i ||Delta H_i^o||_F / ||W0 H_i^o||_F.

    ``delta_h_off`` and ``w0_h_off`` contain token rows from several samples.
    ``case_offsets`` has length ``N+1`` and identifies the row interval for
    each sample.  Each sample is normalized independently before averaging.
    """
    delta_matrix = _metric_matrix(delta_h_off, "delta_h_off")
    baseline_matrix = _metric_matrix(w0_h_off, "w0_h_off")
    if delta_matrix.shape != baseline_matrix.shape:
        raise ValueError(
            "delta_h_off and w0_h_off must have identical shapes; "
            f"got {tuple(delta_matrix.shape)} and "
            f"{tuple(baseline_matrix.shape)}"
        )
    offsets = torch.as_tensor(case_offsets, dtype=torch.int64).flatten().cpu()
    if offsets.numel() < 2:
        raise ValueError("case_offsets must contain at least [0, M]")
    if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != delta_matrix.shape[0]:
        raise ValueError(
            "case_offsets must start at 0 and end at the number of token rows; "
            f"got endpoints ({int(offsets[0].item())}, "
            f"{int(offsets[-1].item())}) for {delta_matrix.shape[0]} rows"
        )
    if bool((offsets[1:] < offsets[:-1]).any().item()):
        raise ValueError("case_offsets must be nondecreasing")

    per_case_values = []
    for case_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        start_index = int(start.item())
        end_index = int(end.item())
        if end_index <= start_index:
            raise ValueError(
                f"epsilon_sp sample {case_index} contains no off-subject tokens"
            )
        denominator = torch.linalg.vector_norm(
            baseline_matrix[start_index:end_index]
        )
        if float(denominator.item()) <= 0.0:
            raise ValueError(
                f"epsilon_sp sample {case_index} has zero ||W0 H_i^o||_F"
            )
        numerator = torch.linalg.vector_norm(delta_matrix[start_index:end_index])
        per_case_values.append(numerator / denominator)

    per_case = torch.stack(per_case_values)
    return float(per_case.mean().item()), per_case.numpy()


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    raise TypeError(f"Could not find a tensor in {type(value)!r}")


def _case_ids(requests: Sequence[Dict[str, Any]]) -> np.ndarray:
    values = [request["case_id"] for request in requests]
    try:
        return np.asarray(values, dtype=np.int64)
    except (TypeError, ValueError):
        return np.asarray([str(value) for value in values])


def _tokenize(model, tok, requests: Sequence[Dict[str, Any]]):
    prompts = [request["prompt"].format(request["subject"]) for request in requests]
    encoded = tok(prompts, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    if hasattr(encoded, "to"):
        return encoded.to(device)
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


def _target_new_text(request: Dict[str, Any]) -> str:
    target = request.get("target_new")
    if isinstance(target, dict):
        target = target.get("str")
    if not isinstance(target, str) or not target:
        raise ValueError(
            "Each request must provide a non-empty target_new string for "
            "Stage1 rewrite-prompt evaluation"
        )
    return target


def _append_target_text(prompt: str, target: str) -> str:
    separator = "" if target.startswith(" ") else " "
    return prompt + separator + target


def _teacher_forcing_batch(
    model,
    tok,
    requests: Sequence[Dict[str, Any]],
    fact_token: str,
    find_fact_lookup_idx: Callable[..., int],
):
    """Build SCRO-style full-target teacher-forcing inputs for bare prompts."""
    rendered_prompts = [
        request["prompt"].format(request["subject"]) for request in requests
    ]
    targets = [_target_new_text(request) for request in requests]
    full_texts = [
        _append_target_text(prompt, target)
        for prompt, target in zip(rendered_prompts, targets)
    ]
    prompt_id_rows = tok(
        rendered_prompts, add_special_tokens=True
    )["input_ids"]
    full_id_rows = tok(full_texts, add_special_tokens=True)["input_ids"]
    encoded = tok(
        full_texts,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
    )
    device = next(model.parameters()).device
    if hasattr(encoded, "to"):
        encoded = encoded.to(device)
    else:
        encoded = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in encoded.items()
        }

    model_name = str(model.config._name_or_path).lower()
    lookup_positions = []
    prediction_positions = []
    target_token_ids = []
    for row, (request, prompt_ids, full_ids) in enumerate(
        zip(requests, prompt_id_rows, full_id_rows)
    ):
        prompt_len = len(prompt_ids)
        target_len = len(full_ids) - prompt_len
        if prompt_len <= 0 or target_len <= 0:
            raise RuntimeError(
                "Stage1 rewrite-prompt target is empty after tokenization: "
                f"case_id={request.get('case_id')}, prompt_len={prompt_len}, "
                f"full_len={len(full_ids)}"
            )

        raw_lookup = int(
            find_fact_lookup_idx(
                request["prompt"],
                request["subject"],
                tok,
                fact_token,
                verbose=False,
                model_name=model_name,
            )
        )
        if raw_lookup < 0:
            raw_lookup += prompt_len
        if raw_lookup < 0 or raw_lookup >= prompt_len:
            raise RuntimeError(
                "subject_last is outside the unpadded rewrite prompt during "
                "Stage1 evaluation: "
                f"case_id={request.get('case_id')}, index={raw_lookup}, "
                f"tokens={prompt_len}"
            )

        valid = torch.nonzero(
            encoded["attention_mask"][row].to(dtype=torch.bool),
            as_tuple=False,
        ).flatten()
        if int(valid.numel()) != len(full_ids):
            raise RuntimeError(
                "Teacher-forcing attention mask does not match tokenized input: "
                f"case_id={request.get('case_id')}, valid={int(valid.numel())}, "
                f"tokens={len(full_ids)}"
            )
        lookup_positions.append(int(valid[raw_lookup].item()))
        row_prediction_positions = valid[
            prompt_len - 1 : prompt_len - 1 + target_len
        ]
        if int(row_prediction_positions.numel()) != target_len:
            raise RuntimeError(
                "Could not align every Stage1 target token with a prediction "
                f"position for case_id={request.get('case_id')}"
            )
        prediction_positions.append(row_prediction_positions)
        target_token_ids.append(
            encoded["input_ids"][row, valid[prompt_len:]].detach()
        )

    return encoded, lookup_positions, prediction_positions, target_token_ids


def _positions(
    model,
    tok,
    requests: Sequence[Dict[str, Any]],
    encoded,
    fact_token: str,
    find_fact_lookup_idx: Callable[..., int],
) -> Tuple[list[int], list[int], list[torch.Tensor], list[int]]:
    """Return padded subject/prediction positions and valid-token positions."""
    model_name = str(model.config._name_or_path).lower()
    lookup_positions = []
    prediction_positions = []
    valid_positions = []
    unpadded_lookup_positions = []
    for row, request in enumerate(requests):
        raw_lookup = int(
            find_fact_lookup_idx(
                request["prompt"],
                request["subject"],
                tok,
                fact_token,
                verbose=False,
                model_name=model_name,
            )
        )
        valid = torch.nonzero(
            encoded["attention_mask"][row].to(dtype=torch.bool), as_tuple=False
        ).flatten()
        if valid.numel() == 0:
            raise RuntimeError("A rewrite prompt contains no valid tokens")
        if raw_lookup < 0:
            raw_lookup += int(valid.numel())
        if raw_lookup < 0 or raw_lookup >= int(valid.numel()):
            raise RuntimeError(
                "subject_last is outside the unpadded rewrite prompt: "
                f"case_id={request.get('case_id')}, index={raw_lookup}, "
                f"tokens={int(valid.numel())}"
            )
        lookup_positions.append(int(valid[raw_lookup].item()))
        prediction_positions.append(int(valid[-1].item()))
        valid_positions.append(valid)
        unpadded_lookup_positions.append(raw_lookup)
    return (
        lookup_positions,
        prediction_positions,
        valid_positions,
        unpadded_lookup_positions,
    )


def _orient_delta(
    delta_weight: torch.Tensor, input_size: int, output_size: int
) -> Tuple[torch.Tensor, str]:
    if tuple(delta_weight.shape) == (output_size, input_size):
        return delta_weight, "weight"
    if tuple(delta_weight.T.shape) == (output_size, input_size):
        return delta_weight.T.contiguous(), "weight_transposed"
    raise ValueError(
        "Cannot orient final-layer Delta: "
        f"shape={tuple(delta_weight.shape)}, expected output={output_size}, "
        f"input={input_size}"
    )


def _valid_cholesky(chol: torch.Tensor, info: torch.Tensor) -> bool:
    diagonal = torch.diagonal(chol)
    return (
        int(info.max().item()) == 0
        and bool(torch.isfinite(chol).all().item())
        and bool((diagonal > 0).all().item())
    )


def _cholesky_for_key_spectrum(
    covariance: torch.Tensor,
) -> Tuple[torch.Tensor, float, float]:
    """Factor C0, applying the same fixed ridge policy used by SCRO if needed."""
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            "Last-layer key spectrum requires a square covariance matrix; "
            f"got {tuple(covariance.shape)}"
        )
    covariance = 0.5 * (covariance + covariance.T)
    if not bool(torch.isfinite(covariance).all().item()):
        raise RuntimeError("Last-layer C0 contains non-finite values")

    chol, info = torch.linalg.cholesky_ex(
        covariance,
        upper=False,
        check_errors=False,
    )
    if _valid_cholesky(chol, info):
        return chol, 0.0, 0.0

    diagonal_scale = float(torch.diagonal(covariance).mean().item())
    if not math.isfinite(diagonal_scale) or diagonal_scale <= 0.0:
        raise RuntimeError(
            "Last-layer C0 Cholesky fallback requires a finite positive "
            "mean diagonal"
        )
    relative_jitter = KEY_SPECTRUM_RELATIVE_JITTER
    absolute_jitter = relative_jitter * diagonal_scale
    regularized = covariance.clone()
    torch.diagonal(regularized).add_(absolute_jitter)
    chol, info = torch.linalg.cholesky_ex(
        regularized,
        upper=False,
        check_errors=False,
    )
    if _valid_cholesky(chol, info):
        return chol, relative_jitter, absolute_jitter

    raise RuntimeError(
        "Last-layer C0 Cholesky remained invalid after ridge: "
        f"info={int(info.max().item())}, "
        f"finite={bool(torch.isfinite(chol).all().item())}, "
        f"relative_jitter={relative_jitter}, "
        f"absolute_jitter={absolute_jitter}"
    )


def _checked_singular_values(matrix: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(matrix).all().item()):
        raise RuntimeError("Last-layer key spectrum input contains non-finite values")
    try:
        singular_values = torch.linalg.svdvals(matrix)
    except RuntimeError:
        if not matrix.is_cuda:
            raise
        # gesvd is slower but more robust than the default CUDA Jacobi driver.
        _, singular_values, _ = torch.linalg.svd(
            matrix,
            full_matrices=False,
            driver="gesvd",
        )
    if not bool(torch.isfinite(singular_values).all().item()):
        raise RuntimeError("Last-layer key spectrum SVD returned non-finite values")
    return singular_values


def _positive_numerical_spectrum(
    singular_values: torch.Tensor,
    matrix_shape: torch.Size,
) -> Tuple[torch.Tensor, float, float]:
    """Apply one explicit float32-scale numerical-rank rule to K and C."""
    if singular_values.ndim != 1 or singular_values.numel() == 0:
        raise ValueError("Numerical-rank calculation requires singular values")
    rtol = max(matrix_shape) * torch.finfo(KEY_RANK_REFERENCE_DTYPE).eps
    tolerance = float((singular_values[0] * rtol).item())
    retained = singular_values > tolerance
    return singular_values[retained].contiguous(), tolerance, float(rtol)


def _compute_last_layer_key_spectrum(
    effective_covariance: torch.Tensor,
    solver_keys: torch.Tensor,
) -> Dict[str, Any]:
    """Compute ranks of K and C0^-1/2 K plus C's full numerical spectrum."""
    if solver_keys.ndim != 2:
        raise ValueError(f"K must be a matrix; got {tuple(solver_keys.shape)}")
    if effective_covariance.shape[0] != solver_keys.shape[0]:
        raise ValueError(
            "C0 and K input dimensions differ: "
            f"C0={tuple(effective_covariance.shape)}, "
            f"K={tuple(solver_keys.shape)}"
        )

    device = solver_keys.device
    keys = solver_keys.detach().to(device=device, dtype=torch.float64)
    covariance = effective_covariance.detach().to(
        device=device,
        dtype=torch.float64,
    )
    chol, relative_jitter, absolute_jitter = _cholesky_for_key_spectrum(
        covariance
    )
    whitened_keys = torch.linalg.solve_triangular(
        chol,
        keys,
        upper=False,
    )
    c_singular_values = _checked_singular_values(whitened_keys)
    c_positive, c_tolerance, rank_rtol = _positive_numerical_spectrum(
        c_singular_values,
        whitened_keys.shape,
    )
    k_singular_values = _checked_singular_values(keys)
    k_positive, k_tolerance, k_rank_rtol = _positive_numerical_spectrum(
        k_singular_values,
        keys.shape,
    )
    if rank_rtol != k_rank_rtol:
        raise RuntimeError("K and whitened K unexpectedly used different rank rtol")

    c_rank = int(c_positive.numel())
    k_rank = int(k_positive.numel())
    return {
        "c0_inv_sqrt_k_nonzero_singular_values": c_positive.detach().cpu(),
        "c0_inv_sqrt_k_rank": c_rank,
        "k_rank": k_rank,
        "c0_inv_sqrt_k_svd_tolerance": c_tolerance,
        "k_svd_tolerance": k_tolerance,
        "rank_rtol": rank_rtol,
        "rank_reference_dtype": str(KEY_RANK_REFERENCE_DTYPE).replace(
            "torch.", ""
        ),
        "relative_jitter": float(relative_jitter),
        "absolute_jitter": float(absolute_jitter),
        "c0_inv_sqrt_k_sigma_max": (
            None if c_rank == 0 else float(c_positive[0].item())
        ),
        "c0_inv_sqrt_k_sigma_min": (
            None if c_rank == 0 else float(c_positive[-1].item())
        ),
        "c0_inv_sqrt_k_condition_number": (
            None
            if c_rank == 0
            else float((c_positive[0] / c_positive[-1]).item())
        ),
        "k_shape": [int(value) for value in keys.shape],
    }


def _save_last_layer_key_spectrum(
    output_dir: Path,
    effective_covariance: torch.Tensor,
    solver_keys: torch.Tensor,
    covariance_definition: str,
) -> Dict[str, Any]:
    spectrum = _compute_last_layer_key_spectrum(
        effective_covariance,
        solver_keys,
    )
    singular_values = spectrum.pop(
        "c0_inv_sqrt_k_nonzero_singular_values"
    )
    np.savez(
        output_dir / "c0_inv_sqrt_k_spectrum.npz",
        c0_inv_sqrt_k_nonzero_singular_values=singular_values.numpy(),
        c0_inv_sqrt_k_rank=np.int64(spectrum["c0_inv_sqrt_k_rank"]),
        k_rank=np.int64(spectrum["k_rank"]),
        c0_inv_sqrt_k_svd_tolerance=np.float64(
            spectrum["c0_inv_sqrt_k_svd_tolerance"]
        ),
        k_svd_tolerance=np.float64(spectrum["k_svd_tolerance"]),
        rank_rtol=np.float64(spectrum["rank_rtol"]),
        rank_reference_dtype=np.asarray(spectrum["rank_reference_dtype"]),
        relative_jitter=np.float64(spectrum["relative_jitter"]),
        absolute_jitter=np.float64(spectrum["absolute_jitter"]),
    )
    summary = {
        "definition": "C = C0^{-1/2} K",
        "implementation": (
            "C is formed as L^{-1}K for C0=LL^T; this has the same singular "
            "values as the symmetric inverse-square-root whitening."
        ),
        "covariance_definition": covariance_definition,
        "c0_regularized_for_cholesky": bool(spectrum["relative_jitter"] > 0),
        "ranks_equal": bool(
            spectrum["c0_inv_sqrt_k_rank"] == spectrum["k_rank"]
        ),
        **spectrum,
    }
    with (output_dir / "c0_inv_sqrt_k_spectrum_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print("Last-layer key spectrum", summary)
    return summary


def initialise_last_layer_tensor_capture(
    enabled: bool,
    output_dir: Optional[str],
    algorithm: str,
    *,
    chunk_cases: int = DEFAULT_CHUNK_CASES,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    if not output_dir:
        raise ValueError("last_layer_fit_output_dir is required")
    if chunk_cases <= 0:
        raise ValueError("last-layer tensor chunk size must be positive")
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    context: Dict[str, Any] = {
        "algorithm": algorithm,
        "output_dir": path,
        "chunk_cases": int(chunk_cases),
        "complete": False,
    }
    _write_manifest(context)
    return context


def capture_stage1_next_hidden(
    context: Optional[Dict[str, Any]],
    *,
    model,
    tok,
    requests: Sequence[Dict[str, Any]],
    injections: Sequence[Tuple[str, torch.Tensor]],
    final_norm_module: str,
    original_rewrite_module: str,
    fact_token: str,
    find_fact_lookup_idx: Callable[..., int],
) -> None:
    """Capture Stage1 oracle hidden states on bare rewrite prompts."""
    if context is None:
        return
    requests = list(requests)
    if not injections:
        raise ValueError("Stage1 capture requires at least one injection")
    for module_name, targets in injections:
        if targets.ndim != 2 or targets.shape[1] != len(requests):
            raise ValueError(
                f"Injection {module_name} has shape {tuple(targets.shape)}; "
                f"expected (*, {len(requests)})"
            )

    original_w0_weight = nethook.get_parameter(
        model, f"{original_rewrite_module}.weight"
    ).detach()
    original_off_dir = context["output_dir"] / "rewrite_prompt_off_tokens"
    original_off_dir.mkdir(parents=True, exist_ok=True)
    for stale in original_off_dir.glob("part_*.npz"):
        stale.unlink()
    original_part_files = []

    chunks = []
    exact_correct_chunks = []
    token_correct_fraction_chunks = []
    target_token_count_chunks = []
    chunk_cases = context["chunk_cases"]
    for part_index, start in enumerate(range(0, len(requests), chunk_cases)):
        end = min(len(requests), start + chunk_cases)
        request_slice = requests[start:end]
        encoded = _tokenize(model, tok, request_slice)
        (
            lookup_positions,
            prediction_positions,
            valid_positions,
            raw_lookup_positions,
        ) = _positions(
            model,
            tok,
            request_slice,
            encoded,
            fact_token,
            find_fact_lookup_idx,
        )

        # This forward pass contains no Stage1 injection and occurs before any
        # model-editing update is installed.  It therefore captures both H^o
        # and W0 from the completely original model.
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=original_rewrite_module,
                retain_input=True,
                detach=True,
            ) as original_trace:
                model(**encoded)
        original_inputs = _first_tensor(original_trace.input)
        original_input_size = int(original_inputs.shape[-1])
        if tuple(original_w0_weight.shape)[1:] == (original_input_size,):
            original_w0_orientation = "weight"
            original_output_size = int(original_w0_weight.shape[0])
        elif int(original_w0_weight.shape[0]) == original_input_size:
            original_w0_orientation = "weight_transposed"
            original_output_size = int(original_w0_weight.shape[1])
        else:
            raise ValueError(
                "Cannot orient original-model W0: "
                f"shape={tuple(original_w0_weight.shape)}, "
                f"input_size={original_input_size}"
            )

        original_w0_rows = []
        original_unpadded_positions = []
        original_relative_positions = []
        original_offsets = [0]
        for row, (valid, raw_lookup) in enumerate(
            zip(valid_positions, raw_lookup_positions)
        ):
            valid_list = valid.tolist()
            off_unpadded = [
                index
                for index in range(len(valid_list))
                if index != raw_lookup
            ]
            off_padded = [valid_list[index] for index in off_unpadded]
            if off_padded:
                original_values = original_inputs[row, off_padded].detach().to(
                    device=original_w0_weight.device,
                    dtype=original_w0_weight.dtype,
                )
                if original_w0_orientation == "weight":
                    original_effects = original_values @ original_w0_weight.T
                else:
                    original_effects = original_values @ original_w0_weight
                original_w0_rows.append(
                    original_effects.detach().float().cpu()
                )
                original_unpadded_positions.extend(off_unpadded)
                original_relative_positions.extend(
                    [position - raw_lookup for position in off_unpadded]
                )
            original_offsets.append(
                original_offsets[-1] + len(off_padded)
            )

        if original_w0_rows:
            w0_h_off_original_model = torch.cat(
                original_w0_rows, dim=0
            ).numpy()
        else:
            w0_h_off_original_model = np.empty(
                (0, original_output_size), dtype=np.float32
            )
        original_part_name = f"part_{part_index:05d}.npz"
        np.savez(
            original_off_dir / original_part_name,
            w0_h_off_original_model=w0_h_off_original_model,
            case_ids=_case_ids(request_slice),
            case_offsets=np.asarray(original_offsets, dtype=np.int64),
            token_positions=np.asarray(
                original_unpadded_positions, dtype=np.int64
            ),
            token_positions_relative_to_subject=np.asarray(
                original_relative_positions, dtype=np.int64
            ),
            subject_positions=np.asarray(
                raw_lookup_positions, dtype=np.int64
            ),
        )
        original_part_files.append(
            str(Path("rewrite_prompt_off_tokens") / original_part_name)
        )
        targets_by_module = {
            module_name: targets[:, start:end]
            for module_name, targets in injections
        }

        def inject(output, layer):
            values = targets_by_module.get(layer)
            if values is None:
                return output
            hidden = _first_tensor(output)
            for row, position in enumerate(lookup_positions):
                hidden[row, position].copy_(
                    values[:, row].to(device=hidden.device, dtype=hidden.dtype)
                )
            return output

        trace_layers = list(dict.fromkeys([*targets_by_module, final_norm_module]))
        with torch.no_grad():
            with nethook.TraceDict(
                module=model,
                layers=trace_layers,
                retain_output=True,
                detach=True,
                edit_output=inject,
            ) as traces:
                model(**encoded)
        final_hidden = _first_tensor(traces[final_norm_module].output)
        chunks.append(
            torch.stack(
                [
                    final_hidden[row, position]
                    for row, position in enumerate(prediction_positions)
                ]
            )
            .detach()
            .float()
            .cpu()
        )

        (
            teacher_encoded,
            teacher_lookup_positions,
            prediction_positions,
            target_token_ids,
        ) = _teacher_forcing_batch(
            model,
            tok,
            request_slice,
            fact_token,
            find_fact_lookup_idx,
        )

        def inject_teacher_forcing(output, layer):
            values = targets_by_module.get(layer)
            if values is None:
                return output
            hidden = _first_tensor(output)
            for row, position in enumerate(teacher_lookup_positions):
                hidden[row, position].copy_(
                    values[:, row].to(
                        device=hidden.device,
                        dtype=hidden.dtype,
                    )
                )
            return output

        with torch.no_grad():
            with nethook.TraceDict(
                module=model,
                layers=list(targets_by_module),
                retain_output=False,
                edit_output=inject_teacher_forcing,
            ):
                teacher_logits = model(**teacher_encoded).logits

        chunk_exact_correct = []
        chunk_token_correct_fraction = []
        chunk_target_token_counts = []
        for row, (positions, labels) in enumerate(
            zip(prediction_positions, target_token_ids)
        ):
            predictions = teacher_logits[row, positions, :].argmax(dim=-1)
            token_correct = predictions.eq(labels.to(predictions.device))
            chunk_exact_correct.append(bool(token_correct.all().item()))
            chunk_token_correct_fraction.append(
                float(token_correct.float().mean().item())
            )
            chunk_target_token_counts.append(int(labels.numel()))
        exact_correct_chunks.append(
            torch.tensor(chunk_exact_correct, dtype=torch.bool)
        )
        token_correct_fraction_chunks.append(
            torch.tensor(chunk_token_correct_fraction, dtype=torch.float64)
        )
        target_token_count_chunks.append(
            torch.tensor(chunk_target_token_counts, dtype=torch.int64)
        )

    context["requests"] = deepcopy(requests)
    context["case_ids"] = [request["case_id"] for request in requests]
    context["h_stage1"] = torch.cat(chunks, dim=0)
    context["stage1_injection_modules"] = [name for name, _ in injections]
    context["original_model_rewrite_module"] = original_rewrite_module
    context["original_model_off_token_part_files"] = original_part_files
    exact_correct = torch.cat(exact_correct_chunks, dim=0)
    token_correct_fraction = torch.cat(
        token_correct_fraction_chunks, dim=0
    )
    target_token_counts = torch.cat(target_token_count_chunks, dim=0)
    if int(exact_correct.numel()) != len(requests):
        raise RuntimeError(
            "Stage1 rewrite-prompt correctness does not align with requests"
        )
    correct_count = int(exact_correct.sum().item())
    rewrite_prompts_correct = correct_count / len(requests)
    case_ids = _case_ids(requests)
    np.savez(
        context["output_dir"] / "stage1_rewrite_prompt_eval.npz",
        exact_correct=exact_correct.numpy(),
        token_correct_fraction=token_correct_fraction.numpy(),
        target_token_counts=target_token_counts.numpy(),
        case_ids=case_ids,
        criterion=np.asarray(
            "teacher_forced_all_target_new_tokens_exact_match"
        ),
    )
    stage1_metrics = {
        "definition": "bare_rewrite_prompt_stage1_oracle_v1",
        "criterion": "teacher_forced_all_target_new_tokens_exact_match",
        "num_requests": len(requests),
        "correct_count": correct_count,
        "rewrite_prompts_correct": rewrite_prompts_correct,
        "rewrite_prompts_correct_token": float(
            token_correct_fraction.mean().item()
        ),
    }
    with (
        context["output_dir"] / "stage1_rewrite_prompt_metrics.json"
    ).open("w", encoding="utf-8") as handle:
        json.dump(stage1_metrics, handle, indent=2, ensure_ascii=False)
    context["stage1_rewrite_correct_count"] = correct_count
    context["stage1_rewrite_prompts_correct"] = rewrite_prompts_correct
    context["stage1_rewrite_prompts_correct_token"] = stage1_metrics[
        "rewrite_prompts_correct_token"
    ]
    print("Stage1 rewrite-prompt metrics", stage1_metrics)
    _write_manifest(context)


def capture_last_layer_linear_tensors(
    context: Optional[Dict[str, Any]],
    *,
    model,
    tok,
    requests: Sequence[Dict[str, Any]],
    layer: int,
    solver_keys: torch.Tensor,
    effective_covariance: torch.Tensor,
    covariance_definition: str,
    rewrite_prompt_keys: torch.Tensor,
    delta_weight: torch.Tensor,
    residual: torch.Tensor,
    rewrite_module: str,
    fact_token: str,
    find_fact_lookup_idx: Callable[..., int],
) -> None:
    """Save key spectra, K, k, Delta, H_off, Delta H_off, and W0 H_off."""
    if context is None:
        return
    requests = list(requests)
    num_cases = len(requests)
    if solver_keys.ndim != 2 or solver_keys.shape[1] != num_cases:
        raise ValueError(
            f"solver K shape {tuple(solver_keys.shape)} does not match {num_cases}"
        )
    if rewrite_prompt_keys.ndim != 2 or rewrite_prompt_keys.shape[1] != num_cases:
        raise ValueError(
            "rewrite-prompt K shape "
            f"{tuple(rewrite_prompt_keys.shape)} does not match {num_cases}"
        )
    if residual.ndim != 2 or residual.shape[1] != num_cases:
        raise ValueError(
            f"residual shape {tuple(residual.shape)} does not match {num_cases}"
        )
    if context.get("case_ids") != [request["case_id"] for request in requests]:
        raise RuntimeError("Stage1 and final-layer request ordering differ")

    delta_operator, delta_orientation = _orient_delta(
        delta_weight.detach().float(),
        int(solver_keys.shape[0]),
        int(residual.shape[0]),
    )
    delta_operator = delta_operator.detach().float()
    w0_weight = nethook.get_parameter(
        model, f"{rewrite_module}.weight"
    ).detach()
    input_size = int(solver_keys.shape[0])
    output_size = int(residual.shape[0])
    if tuple(w0_weight.shape) == (output_size, input_size):
        w0_orientation = "weight"
    elif tuple(w0_weight.shape) == (input_size, output_size):
        w0_orientation = "weight_transposed"
    else:
        raise ValueError(
            "Cannot orient original final-layer W0: "
            f"shape={tuple(w0_weight.shape)}, expected output={output_size}, "
            f"input={input_size}"
        )
    output_dir: Path = context["output_dir"]
    case_ids = _case_ids(requests)
    key_spectrum_summary = _save_last_layer_key_spectrum(
        output_dir,
        effective_covariance,
        solver_keys,
        covariance_definition,
    )
    np.savez(
        output_dir / "k_solver_mean.npz",
        k_solver_mean=solver_keys.T.detach().float().cpu().numpy(),
        case_ids=case_ids,
        layer=np.int64(layer),
    )
    np.savez(
        output_dir / "k_rewrite_prompt.npz",
        k_rewrite_prompt=rewrite_prompt_keys.T.detach().float().cpu().numpy(),
        case_ids=case_ids,
        layer=np.int64(layer),
    )
    np.savez(
        output_dir / "delta_operator.npz",
        delta=delta_operator.cpu().numpy(),
        layer=np.int64(layer),
        orientation=np.asarray(delta_orientation),
    )

    delta_k_sample_major = (
        delta_operator.to(solver_keys.device) @ solver_keys.detach().float()
    ).T.detach().cpu()
    residual_sample_major = residual.T.detach().float().cpu()
    epsilon_fit, epsilon_fit_per_case = compute_epsilon_fit(
        delta_k_sample_major,
        residual_sample_major,
    )

    off_dir = output_dir / "rewrite_prompt_off_tokens"
    off_dir.mkdir(parents=True, exist_ok=True)
    if context.get("original_model_rewrite_module") != rewrite_module:
        raise RuntimeError(
            "Original-model and dynamic final-layer captures use different "
            "rewrite modules: "
            f"{context.get('original_model_rewrite_module')} vs {rewrite_module}"
        )

    part_files = []
    total_off_tokens = 0
    epsilon_sp_per_case_parts = []
    chunk_cases = context["chunk_cases"]
    for part_index, start in enumerate(range(0, num_cases, chunk_cases)):
        end = min(num_cases, start + chunk_cases)
        request_slice = requests[start:end]
        encoded = _tokenize(model, tok, request_slice)
        lookup_positions, _, valid_positions, raw_lookup_positions = _positions(
            model,
            tok,
            request_slice,
            encoded,
            fact_token,
            find_fact_lookup_idx,
        )
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=rewrite_module,
                retain_input=True,
                detach=True,
            ) as trace:
                model(**encoded)
        rewrite_inputs = _first_tensor(trace.input)
        h_rows = []
        delta_rows = []
        w0_rows = []
        unpadded_positions = []
        relative_positions = []
        offsets = [0]
        delta_device = delta_operator.to(rewrite_inputs.device)
        for row, (valid, lookup, raw_lookup) in enumerate(
            zip(valid_positions, lookup_positions, raw_lookup_positions)
        ):
            valid_list = valid.tolist()
            off_unpadded = [index for index in range(len(valid_list)) if index != raw_lookup]
            off_padded = [valid_list[index] for index in off_unpadded]
            if off_padded:
                native_values = rewrite_inputs[row, off_padded].detach()
                values = native_values.float()
                effects = values @ delta_device.T
                w0_values = native_values.to(
                    device=w0_weight.device,
                    dtype=w0_weight.dtype,
                )
                if w0_orientation == "weight":
                    w0_effects = w0_values @ w0_weight.T
                else:
                    w0_effects = w0_values @ w0_weight
                h_rows.append(values.cpu())
                delta_rows.append(effects.detach().float().cpu())
                w0_rows.append(w0_effects.detach().float().cpu())
                unpadded_positions.extend(off_unpadded)
                relative_positions.extend(
                    [position - raw_lookup for position in off_unpadded]
                )
            offsets.append(offsets[-1] + len(off_padded))

        if h_rows:
            h_off = torch.cat(h_rows, dim=0).numpy()
            delta_h_off = torch.cat(delta_rows, dim=0).numpy()
            w0_h_off = torch.cat(w0_rows, dim=0).numpy()
        else:
            h_off = np.empty((0, solver_keys.shape[0]), dtype=np.float32)
            delta_h_off = np.empty((0, residual.shape[0]), dtype=np.float32)
            w0_h_off = np.empty((0, residual.shape[0]), dtype=np.float32)
        part_name = f"part_{part_index:05d}.npz"
        original_part_path = off_dir / part_name
        if not original_part_path.is_file():
            raise FileNotFoundError(
                "Missing original-model W0H capture for final-layer part: "
                f"{original_part_path}"
            )
        with np.load(original_part_path, allow_pickle=False) as original_part:
            w0_h_off_original_model = original_part[
                "w0_h_off_original_model"
            ]
            original_case_ids = original_part["case_ids"]
            original_case_offsets = original_part["case_offsets"]
            original_token_positions = original_part["token_positions"]
            original_relative_positions = original_part[
                "token_positions_relative_to_subject"
            ]
            original_subject_positions = original_part["subject_positions"]
        current_case_ids = _case_ids(request_slice)
        case_offsets = np.asarray(offsets, dtype=np.int64)
        current_token_positions = np.asarray(
            unpadded_positions, dtype=np.int64
        )
        current_relative_positions = np.asarray(
            relative_positions, dtype=np.int64
        )
        current_subject_positions = np.asarray(
            raw_lookup_positions, dtype=np.int64
        )
        if not np.array_equal(original_case_ids, current_case_ids):
            raise RuntimeError(
                f"Original/dynamic case ordering differs in {part_name}"
            )
        if not np.array_equal(original_case_offsets, case_offsets):
            raise RuntimeError(
                f"Original/dynamic token offsets differ in {part_name}"
            )
        if not np.array_equal(
            original_token_positions, current_token_positions
        ):
            raise RuntimeError(
                f"Original/dynamic token positions differ in {part_name}"
            )
        if not np.array_equal(
            original_relative_positions, current_relative_positions
        ) or not np.array_equal(
            original_subject_positions, current_subject_positions
        ):
            raise RuntimeError(
                f"Original/dynamic subject alignment differs in {part_name}"
            )
        if w0_h_off_original_model.shape != w0_h_off.shape:
            raise RuntimeError(
                "Original-model and dynamic W0H shapes differ in "
                f"{part_name}: {w0_h_off_original_model.shape} vs "
                f"{w0_h_off.shape}"
            )
        _, epsilon_sp_part = compute_epsilon_sp(
            delta_h_off,
            w0_h_off,
            case_offsets,
        )
        epsilon_sp_per_case_parts.append(epsilon_sp_part)
        np.savez(
            off_dir / part_name,
            h_off=h_off,
            delta_h_off=delta_h_off,
            w0_h_off=w0_h_off,
            w0_h_off_original_model=w0_h_off_original_model,
            case_ids=current_case_ids,
            case_offsets=case_offsets,
            token_positions=current_token_positions,
            token_positions_relative_to_subject=current_relative_positions,
            subject_positions=current_subject_positions,
            layer=np.int64(layer),
        )
        part_files.append(str(Path("rewrite_prompt_off_tokens") / part_name))
        total_off_tokens += int(h_off.shape[0])

    epsilon_sp_per_case = np.concatenate(epsilon_sp_per_case_parts)
    if epsilon_fit_per_case.shape != (num_cases,):
        raise RuntimeError(
            "epsilon_fit per-case values do not align with requests: "
            f"{epsilon_fit_per_case.shape} vs ({num_cases},)"
        )
    if epsilon_sp_per_case.shape != (num_cases,):
        raise RuntimeError(
            "epsilon_sp per-case values do not align with requests: "
            f"{epsilon_sp_per_case.shape} vs ({num_cases},)"
        )
    epsilon_sp = float(epsilon_sp_per_case.mean())
    np.savez_compressed(
        output_dir / "epsilon_per_case.npz",
        epsilon_fit=epsilon_fit_per_case,
        epsilon_sp=epsilon_sp_per_case,
        case_ids=case_ids,
        layer=np.int64(layer),
    )
    epsilon_summary = {
        "aggregation": "mean_of_per_case_relative_norms",
        "num_cases": num_cases,
        "epsilon_fit": epsilon_fit,
        "epsilon_fit_definition": (
            "mean_i ||Delta k_i-r_i||_2 / ||r_i||_2 using solver K"
        ),
        "epsilon_sp": epsilon_sp,
        "epsilon_sp_definition": (
            "mean_i ||Delta H_i^o||_F / ||W0 H_i^o||_F on bare "
            "rewrite prompts"
        ),
    }
    with (output_dir / "epsilon_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(epsilon_summary, handle, indent=2, ensure_ascii=False)
    print("Last-layer epsilon metrics", epsilon_summary)

    context.update(
        {
            "layer": int(layer),
            "delta_orientation": delta_orientation,
            "w0_orientation": w0_orientation,
            "num_cases": num_cases,
            "num_off_tokens": total_off_tokens,
            "off_token_part_files": part_files,
            "input_size": int(solver_keys.shape[0]),
            "output_size": int(residual.shape[0]),
            "c0_inv_sqrt_k_rank": key_spectrum_summary[
                "c0_inv_sqrt_k_rank"
            ],
            "k_rank": key_spectrum_summary["k_rank"],
            "key_ranks_equal": key_spectrum_summary["ranks_equal"],
            "key_covariance_definition": covariance_definition,
            "epsilon_fit": epsilon_fit,
            "epsilon_sp": epsilon_sp,
        }
    )
    _write_manifest(context)


def capture_stage2_next_hidden(
    context: Optional[Dict[str, Any]],
    *,
    model,
    tok,
    requests: Sequence[Dict[str, Any]],
    final_norm_module: str,
    fact_token: str,
    find_fact_lookup_idx: Callable[..., int],
) -> None:
    """Capture fully edited Stage2 hidden states and finalize the manifest."""
    if context is None:
        return
    requests = list(requests)
    case_ids = [request["case_id"] for request in requests]
    if context.get("case_ids") != case_ids:
        raise RuntimeError("Stage1 and Stage2 request ordering differ")
    if "h_stage1" not in context:
        raise RuntimeError("Stage1 hidden states were not captured")

    chunks = []
    chunk_cases = context["chunk_cases"]
    for start in range(0, len(requests), chunk_cases):
        end = min(len(requests), start + chunk_cases)
        request_slice = requests[start:end]
        encoded = _tokenize(model, tok, request_slice)
        _, prediction_positions, _, _ = _positions(
            model,
            tok,
            request_slice,
            encoded,
            fact_token,
            find_fact_lookup_idx,
        )
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=final_norm_module,
                retain_output=True,
                detach=True,
            ) as trace:
                model(**encoded)
        final_hidden = _first_tensor(trace.output)
        chunks.append(
            torch.stack(
                [
                    final_hidden[row, position]
                    for row, position in enumerate(prediction_positions)
                ]
            )
            .detach()
            .float()
            .cpu()
        )

    h_stage1 = context["h_stage1"]
    h_stage2 = torch.cat(chunks, dim=0)
    if h_stage1.shape != h_stage2.shape:
        raise RuntimeError(
            f"Stage1/Stage2 hidden shapes differ: {h_stage1.shape} vs {h_stage2.shape}"
        )
    np.savez(
        context["output_dir"] / "next_hidden_stage1_stage2.npz",
        h_stage1=h_stage1.numpy(),
        h_stage2=h_stage2.numpy(),
        case_ids=_case_ids(requests),
        layer=np.int64(context["layer"]),
        position_definition=np.asarray(
            "final_norm_output_at_last_valid_bare_rewrite_prompt_token"
        ),
    )
    context["complete"] = True
    context["next_hidden_size"] = int(h_stage1.shape[1])
    _write_manifest(context)


def _write_manifest(context: Dict[str, Any]) -> None:
    output_dir: Path = context["output_dir"]
    manifest = {
        "version": 6,
        "algorithm": context["algorithm"],
        "complete": bool(context.get("complete", False)),
        "layer": context.get("layer"),
        "num_cases": context.get("num_cases"),
        "num_off_tokens": context.get("num_off_tokens"),
        "input_size": context.get("input_size"),
        "output_size": context.get("output_size"),
        "c0_inv_sqrt_k_rank": context.get("c0_inv_sqrt_k_rank"),
        "k_rank": context.get("k_rank"),
        "key_ranks_equal": context.get("key_ranks_equal"),
        "key_covariance_definition": context.get(
            "key_covariance_definition"
        ),
        "next_hidden_size": context.get("next_hidden_size"),
        "chunk_cases": int(context["chunk_cases"]),
        "delta_orientation": context.get("delta_orientation"),
        "w0_orientation": context.get("w0_orientation"),
        "original_model_rewrite_module": context.get(
            "original_model_rewrite_module"
        ),
        "stage1_injection_modules": context.get("stage1_injection_modules"),
        "stage1_rewrite_correct_count": context.get(
            "stage1_rewrite_correct_count"
        ),
        "stage1_rewrite_prompts_correct": context.get(
            "stage1_rewrite_prompts_correct"
        ),
        "stage1_rewrite_prompts_correct_token": context.get(
            "stage1_rewrite_prompts_correct_token"
        ),
        "epsilon_fit": context.get("epsilon_fit"),
        "epsilon_sp": context.get("epsilon_sp"),
        "files": {
            "solver_keys": "k_solver_mean.npz",
            "key_spectrum": "c0_inv_sqrt_k_spectrum.npz",
            "key_spectrum_summary": (
                "c0_inv_sqrt_k_spectrum_summary.json"
            ),
            "rewrite_prompt_keys": "k_rewrite_prompt.npz",
            "delta": "delta_operator.npz",
            "off_token_parts": context.get("off_token_part_files", []),
            "epsilon_per_case": "epsilon_per_case.npz",
            "epsilon_summary": "epsilon_summary.json",
            "stage1_rewrite_prompt_eval": "stage1_rewrite_prompt_eval.npz",
            "stage1_rewrite_prompt_metrics": (
                "stage1_rewrite_prompt_metrics.json"
            ),
            "stage1_stage2_hidden": "next_hidden_stage1_stage2.npz",
        },
        "definitions": {
            "K": (
                "Final-layer dynamic rewrite+context mean keys after earlier-layer "
                "updates; stored sample-major as k_solver_mean[N,d_in]."
            ),
            "C0_inv_sqrt_K_spectrum": (
                "All singular values above sigma_max * max(d_in,N) * "
                "float32_eps are stored. C is evaluated as the Cholesky "
                "whitening L^{-1}K, which has the same singular values as "
                "the symmetric C0^{-1/2}K. The same relative numerical-rank "
                "rule is used for K. If direct Cholesky fails, the saved "
                "summary records the ridge applied to C0."
            ),
            "k_rewrite_prompt": (
                "Bare rewrite-prompt subject_last key at the same model state as K."
            ),
            "Delta": (
                "Effective float32 last-layer linear update in [d_out,d_in] "
                "orientation."
            ),
            "H_off": (
                "Inputs to the final rewritten module at every valid bare rewrite "
                "prompt token except subject_last; padding is excluded."
            ),
            "Delta_H_off": "Stored exactly as h_off @ Delta.T.",
            "W0_H_off": (
                "Stored exactly as h_off @ W0.T, where W0 is the original "
                "final-layer weight before that layer is edited; bias is excluded."
            ),
            "W0_H_off_original_model": (
                "Stored as W0 applied to off-subject hidden states collected "
                "from a completely unedited model before any editing layer is "
                "installed. It is saved in each off-token part under "
                "w0_h_off_original_model; padding and subject_last are excluded, "
                "and bias is excluded."
            ),
            "epsilon_fit": (
                "For each request i, compute ||Delta k_i-r_i||_2/||r_i||_2 "
                "using the solver key, then average equally over requests."
            ),
            "epsilon_sp": (
                "For each request i, compute ||Delta H_i^o||_F/"
                "||W0 H_i^o||_F over its bare rewrite-prompt off-subject "
                "tokens, then average equally over requests."
            ),
            "h_stage1": (
                "Final-norm hidden at the last bare-prompt token with the absolute "
                "Stage1 target injected at subject_last."
            ),
            "stage1_rewrite_prompts_correct": (
                "Fraction of bare rewrite prompts for which every target_new token "
                "is the teacher-forced argmax under the same absolute Stage1 "
                "injection used for h_stage1."
            ),
            "h_stage2": (
                "Final-norm hidden at the same position in the fully edited model."
            ),
        },
    }
    with (output_dir / "last_layer_tensor_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
