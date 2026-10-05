"""Joint residual optimization in the fixed spectral-scaled coordinate."""

from functools import partial
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .distributed_utils import (
    all_gather_columns,
    all_reduce_sum_,
    broadcast_from_main_,
    distributed_barrier,
    distributed_world_size,
    is_main_process,
    request_partition,
)
from .scro_hparams import SCROHyperParams


def _build_joint_scheduler(
    optimizer: torch.optim.Optimizer,
    hparams: SCROHyperParams,
):
    """Build the fixed cosine scheduler used by SCRO."""
    scheduler_t_max = hparams.v_num_grad_steps
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=scheduler_t_max,
        eta_min=hparams.v_lr * 0.1,
    )


def _get_hidden_size(model) -> int:
    if hasattr(model.config, "n_embd"):
        return model.config.n_embd
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    raise NotImplementedError("Cannot infer model hidden size")


def _edit_joint_output(
    cur_out,
    cur_layer,
    *,
    edited_layer_name,
    request_ids,
    clean_row_by_request,
    lookup_idxs,
    prompt_request_ids,
    target_init_cols,
    delta_basis,
    right_factor,
):
    if cur_layer != edited_layer_name:
        return cur_out

    for request_id in request_ids:
        if target_init_cols[request_id] is None:
            clean_row = clean_row_by_request[request_id]
            target_init_cols[request_id] = (
                cur_out[0][clean_row, lookup_idxs[clean_row]].detach().clone()
            )

    delta_matrix = _joint_residual(delta_basis, right_factor)
    for row_index, request_id in enumerate(prompt_request_ids):
        cur_out[0][row_index, lookup_idxs[row_index], :] += delta_matrix[
            :, request_id
        ]
    return cur_out


def _joint_kl_lookup_idxs(
    kl_rows: torch.Tensor,
    lookup_idxs: List[int],
) -> List[int]:
    return [lookup_idxs[row_index] for row_index in kl_rows.tolist()]


def _build_request_specs(
    model,
    tok,
    requests,
    context_templates,
    fact_token,
    request_indices=None,
):
    specs = []
    if request_indices is None:
        request_indices = range(len(requests))
    for request_index in request_indices:
        request = requests[request_index]
        target_ids = tok(
            request["target_new"]["str"], return_tensors="pt"
        ).to(model.device)["input_ids"][0]
        model_name = str(model.config._name_or_path).lower()
        tokenizer_type = str(type(tok)).lower()
        has_leading_bos = (
            target_ids.numel() > 0
            and tok.bos_token_id is not None
            and target_ids[0].item() == tok.bos_token_id
        )
        if any(
            model_family in model_name
            for model_family in ("mistral", "qwen", "deepseek")
        ):
            opt_target_ids = target_ids
        elif (
            "llama" in tokenizer_type
            or "llama-3.1" in model_name
            or "gemma" in model_name
        ):
            opt_target_ids = target_ids[1:] if has_leading_bos else target_ids
        else:
            opt_target_ids = target_ids
        if opt_target_ids.numel() == 0:
            raise ValueError(
                "Empty target after model-specific BOS handling: "
                f"model={model_name}, request_id={request_index}, "
                f"target={request['target_new']['str']!r}, "
                f"target_ids={target_ids.detach().cpu().tolist()}"
            )
        decoded_prefix = tok.decode(
            opt_target_ids[:-1],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        rewriting_prompts = [
            context.format(request["prompt"]) + decoded_prefix
            for context_group in context_templates
            for context in context_group
        ]
        rewrite_is_bare = [
            context == "{}"
            for context_group in context_templates
            for context in context_group
        ]
        if (
            len(rewrite_is_bare) != len(rewriting_prompts)
            or sum(rewrite_is_bare) != 1
            or not rewrite_is_bare[0]
        ):
            raise ValueError(
                "Joint rewrite loss expects exactly one leading bare '{}' "
                "template followed by context templates"
            )
        all_prompts = rewriting_prompts + ["{} is a"]
        subjects = [request["subject"]] * len(all_prompts)
        filled_prompts = [
            prompt.format(subject)
            for prompt, subject in zip(all_prompts, subjects)
        ]
        lookup_idxs = [
            find_fact_lookup_idx(
                prompt,
                subject,
                tok,
                fact_token,
                verbose=False,
                model_name=str(model.config._name_or_path).lower(),
            )
            for prompt, subject in zip(all_prompts, subjects)
        ]
        specs.append(
            {
                "request_id": request_index,
                "filled_prompts": filled_prompts,
                "lookup_idxs": lookup_idxs,
                "target_ids": opt_target_ids.detach().cpu(),
                "num_rewrite_rows": len(rewriting_prompts),
                "rewrite_is_bare": rewrite_is_bare,
            }
        )
    return specs


def _build_micro_batches(
    model,
    tok,
    request_specs,
    micro_batch_size,
    total_num_requests=None,
):
    batches = []
    local_num_requests = len(request_specs)
    num_requests = total_num_requests or local_num_requests
    for start in range(0, local_num_requests, micro_batch_size):
        batch_specs = request_specs[start : start + micro_batch_size]
        request_ids = [spec["request_id"] for spec in batch_specs]
        filled_prompts = []
        prompt_request_ids = []
        lookup_idxs = []
        rewrite_rows = []
        rewrite_is_bare = []
        kl_rows = []
        clean_row_by_request = {}
        row_target_ids = []

        for spec in batch_specs:
            base_row = len(filled_prompts)
            filled_prompts.extend(spec["filled_prompts"])
            prompt_request_ids.extend(
                [spec["request_id"]] * len(spec["filled_prompts"])
            )
            lookup_idxs.extend(spec["lookup_idxs"])
            clean_row_by_request[spec["request_id"]] = base_row
            for local_row in range(spec["num_rewrite_rows"]):
                rewrite_rows.append(base_row + local_row)
                rewrite_is_bare.append(spec["rewrite_is_bare"][local_row])
                row_target_ids.append(spec["target_ids"])
            kl_rows.append(base_row + spec["num_rewrite_rows"])
            row_target_ids.append(None)

        input_tokens = tok(
            filled_prompts, return_tensors="pt", padding=True
        ).to(model.device)
        rewriting_targets = torch.full(
            input_tokens["input_ids"].shape, -100, dtype=torch.long
        )
        target_lens = torch.ones(
            input_tokens["input_ids"].shape[0], dtype=torch.float32
        )
        for row_index, target_ids in enumerate(row_target_ids):
            if target_ids is None:
                continue
            example_length = input_tokens["attention_mask"][row_index].sum()
            rewriting_targets[
                row_index, example_length - len(target_ids) : example_length
            ] = target_ids
            target_lens[row_index] = float(target_ids.size(0))

        batches.append(
            {
                "request_ids": request_ids,
                "input_tokens": input_tokens,
                "rewriting_targets": rewriting_targets,
                "target_lens": target_lens,
                "prompt_request_ids": prompt_request_ids,
                "lookup_idxs": lookup_idxs,
                "rewrite_rows": torch.tensor(rewrite_rows, dtype=torch.long),
                "rewrite_is_bare": torch.tensor(
                    rewrite_is_bare, dtype=torch.bool
                ),
                "kl_rows": torch.tensor(kl_rows, dtype=torch.long),
                "clean_row_by_request": clean_row_by_request,
                "group_scale": len(request_ids) / num_requests,
            }
        )
    return batches


def _joint_residual(
    delta_basis: torch.Tensor,
    right_factor: torch.Tensor,
) -> torch.Tensor:
    return delta_basis @ right_factor


def _rewrite_nll(
    nll_each: torch.Tensor,
    rewrite_is_bare: torch.Tensor,
    bare_loss_alpha: float,
) -> torch.Tensor:
    """Mix the bare rewrite loss with the mean context rewrite loss."""
    if nll_each.ndim != 1:
        raise ValueError("nll_each must be a one-dimensional tensor")
    rewrite_is_bare = rewrite_is_bare.to(
        device=nll_each.device, dtype=torch.bool
    )
    if rewrite_is_bare.shape != nll_each.shape:
        raise ValueError("rewrite_is_bare must have the same shape as nll_each")
    if not bool(rewrite_is_bare.any().item()):
        raise ValueError("Rewrite loss requires at least one bare prompt")
    context_mask = ~rewrite_is_bare
    if not bool(context_mask.any().item()):
        raise ValueError("Rewrite loss requires at least one context prompt")

    bare_nll = nll_each[rewrite_is_bare].mean()
    context_mean_nll = nll_each[context_mask].mean()
    return (
        float(bare_loss_alpha) * bare_nll
        + (1.0 - float(bare_loss_alpha)) * context_mean_nll
    )


@torch.no_grad()
def _project_spectral_z(
    z: torch.Tensor, op_bound: float
) -> torch.Tensor:
    """Project A/Z onto the squared operator-norm ball."""
    if z.ndim != 2 or min(z.shape) == 0 or not torch.isfinite(z).all():
        raise ValueError("A/Z must be a finite, nonempty matrix")
    if not np.isfinite(op_bound) or op_bound <= 0:
        raise ValueError("Hard A/Z operator bound must be finite and positive")
    work = z.detach().double()
    left, values, right = torch.linalg.svd(work, full_matrices=False)
    cap = op_bound ** 0.5
    clipped = values.clamp(max=cap)
    if torch.equal(clipped, values):
        return z
    return ((left * clipped.unsqueeze(0)) @ right).to(z)


def compute_zs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: SCROHyperParams,
    layer: int,
    context_templates: List[str],
    V_r: torch.Tensor,
    spectral_values: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Learn the jointly optimized residual from a zero (cold) start."""
    if V_r is None:
        raise ValueError("Joint optimization requires a residual row-space basis")
    if spectral_values is None:
        raise ValueError("Spectral-scaled optimization requires spectral values")
    if hparams.v_num_grad_steps <= 0:
        raise ValueError("Stage1 requires at least one loop iteration")
    hparams.configure_rewrite_loss_mix()
    hparams.configure_z_constraints()
    op_bound = float(hparams.joint_spectral_z_op_bound)
    if not np.isfinite(op_bound) or op_bound <= 0:
        raise ValueError("Effective joint spectral Z operator bound must be positive")

    lm_w = nethook.get_parameter(
        model, f"{hparams.lm_head_module}.weight"
    ).T
    ln_f = nethook.get_module(model, hparams.ln_f_module)
    try:
        lm_b = nethook.get_parameter(
            model, f"{hparams.lm_head_module}.bias"
        )
        if lm_b is None:
            lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)
    except LookupError:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    num_requests = len(requests)
    world_size = distributed_world_size()
    main_process = is_main_process()
    if num_requests < world_size:
        raise ValueError(
            "Distributed joint optimization requires at least one request per rank"
        )
    local_start, local_end = request_partition(num_requests)
    request_indices = list(range(local_start, local_end))
    micro_batch_size = min(
        hparams.joint_z_micro_batch_size, len(request_indices)
    )
    request_specs = _build_request_specs(
        model,
        tok,
        requests,
        context_templates,
        hparams.fact_token,
        request_indices=request_indices,
    )
    micro_batches = _build_micro_batches(
        model,
        tok,
        request_specs,
        micro_batch_size,
        total_num_requests=num_requests,
    )
    bare_loss_alpha = hparams.joint_rewrite_bare_loss_alpha

    loss_layer = max(hparams.v_loss_layer, layer)
    hidden_size = _get_hidden_size(model)
    V_r = V_r.to(model.device)
    spectral_values = spectral_values.to(device=model.device, dtype=V_r.dtype)
    if (
        spectral_values.ndim != 1
        or spectral_values.numel() != V_r.size(1)
        or not bool(torch.isfinite(spectral_values).all().item())
        or bool((spectral_values <= 0).any().item())
    ):
        raise ValueError("spectral_values must be a finite positive rank vector")
    right_factor = spectral_values.unsqueeze(1) * V_r.T
    basis_shape = (hidden_size, right_factor.size(0))
    delta_basis = torch.zeros(
        basis_shape,
        device=model.device,
        dtype=right_factor.dtype,
        requires_grad=True,
    )

    target_init_cols = [None for _ in requests]
    target_init = None
    kl_distr_init = [None for _ in requests]
    optimizer = torch.optim.Adam([delta_basis], lr=hparams.v_lr)
    scheduler = _build_joint_scheduler(optimizer, hparams)
    nethook.set_requires_grad(False, model)

    num_updates = max(hparams.v_num_grad_steps - 1, 0)
    num_iterations = max(num_updates, 1)
    for iteration in range(num_iterations):
        update_enabled = iteration < num_updates
        optimizer.zero_grad()
        total_nll = 0.0
        total_kl = 0.0

        for micro_batch in micro_batches:
            input_tokens = {
                key: value.to(model.device)
                for key, value in micro_batch["input_tokens"].items()
            }
            rewriting_targets = micro_batch["rewriting_targets"].to(model.device)
            target_lens = micro_batch["target_lens"].to(model.device)
            rewrite_rows = micro_batch["rewrite_rows"].to(model.device)
            rewrite_is_bare = micro_batch["rewrite_is_bare"].to(model.device)
            kl_rows = micro_batch["kl_rows"].to(model.device)
            request_ids = micro_batch["request_ids"]
            lookup_idxs = micro_batch["lookup_idxs"]
            group_scale = micro_batch["group_scale"]

            kl_lookup_idxs = _joint_kl_lookup_idxs(kl_rows, lookup_idxs)
            edit_output = partial(
                _edit_joint_output,
                edited_layer_name=hparams.layer_module_tmp.format(layer),
                request_ids=request_ids,
                clean_row_by_request=micro_batch["clean_row_by_request"],
                lookup_idxs=lookup_idxs,
                prompt_request_ids=micro_batch["prompt_request_ids"],
                target_init_cols=target_init_cols,
                delta_basis=delta_basis,
                right_factor=right_factor,
            )
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.layer_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output,
            ) as traces:
                model_output = model(**input_tokens)
                logits = (
                    model_output
                    if isinstance(model_output, torch.Tensor)
                    else model_output.logits
                )

            kl_logits = torch.stack(
                [
                    logits[row_index, lookup_index, :]
                    for row_index, lookup_index in zip(kl_rows.tolist(), kl_lookup_idxs)
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            for local_index, request_id in enumerate(request_ids):
                if kl_distr_init[request_id] is None:
                    kl_distr_init[request_id] = (
                        kl_log_probs[local_index].detach().clone()
                    )
            kl_targets = torch.stack(
                [kl_distr_init[request_id] for request_id in request_ids], dim=0
            )

            loss_output = traces[
                hparams.layer_module_tmp.format(loss_layer)
            ].output
            if isinstance(loss_output, (list, tuple)):
                loss_output = loss_output[0]
            if loss_output.shape[1] != rewriting_targets.shape[1]:
                loss_output = loss_output.transpose(0, 1)
            full_repr = loss_output[rewrite_rows]
            rewrite_targets = rewriting_targets[rewrite_rows]
            rewrite_target_lens = target_lens[rewrite_rows]
            log_probs = torch.log_softmax(
                ln_f(full_repr) @ lm_w.to(full_repr.device)
                + lm_b.to(full_repr.device),
                dim=2,
            )
            gathered = torch.gather(
                log_probs,
                2,
                torch.where(rewrite_targets != -100, rewrite_targets, 0)
                .unsqueeze(2)
                .to(log_probs.device),
            ).squeeze(2)
            mask = (rewrite_targets != -100).float()
            nll_each = -(
                gathered * mask.to(gathered.device)
            ).sum(1) / rewrite_target_lens
            nll_loss = _rewrite_nll(
                nll_each,
                rewrite_is_bare,
                bare_loss_alpha,
            )
            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_targets,
                kl_log_probs,
                log_target=True,
                reduction="batchmean",
            )
            if update_enabled:
                (
                    group_scale
                    * (
                        nll_loss
                        + kl_loss.to(nll_loss.device)
                    )
                ).backward()
            total_nll += group_scale * nll_loss.item()
            total_kl += group_scale * kl_loss.item()

        if target_init is None:
            if any(target_init_cols[index] is None for index in request_indices):
                raise RuntimeError("Failed to collect local target_init columns")
            local_target_init = torch.stack(
                [target_init_cols[index] for index in request_indices], dim=1
            )
            target_init = all_gather_columns(local_target_init)
            if target_init.size(1) != num_requests:
                raise RuntimeError(
                    f"Expected {num_requests} target columns, got {target_init.size(1)}"
                )
        delta_matrix = _joint_residual(delta_basis, right_factor)
        target_norms = target_init.norm(dim=0).clamp_min(1e-6)
        delta_norms = delta_matrix.norm(dim=0)
        weight_decay = hparams.v_weight_decay * (
            delta_norms / target_norms.square()
        ).mean()
        if update_enabled:
            (weight_decay / world_size).backward()
            if delta_basis.grad is None:
                raise RuntimeError("Joint residual gradient was not produced")
            all_reduce_sum_(delta_basis.grad)

        metrics = delta_basis.new_tensor([total_nll, total_kl])
        all_reduce_sum_(metrics)
        total_nll, total_kl = [float(value) for value in metrics.tolist()]

        total_loss = total_nll + total_kl + weight_decay.item()
        if main_process:
            print(
                f"joint loss {np.round(total_loss, 3)} = "
                f"{np.round(total_nll, 3)} + "
                f"{np.round(total_kl, 3)} + "
                f"{np.round(weight_decay.item(), 3)}"
            )
        if not update_enabled:
            break

        should_stop = total_loss < 1e-2
        if not should_stop:
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            if main_process:
                physical_z = _project_spectral_z(delta_basis, op_bound)
                post_op = torch.linalg.matrix_norm(
                    physical_z, ord=2
                ).square().item()
                if post_op > op_bound:
                    safeguard = (
                        1 - 8 * torch.finfo(delta_basis.dtype).eps
                    ) / (post_op / op_bound) ** 0.5
                    physical_z = physical_z * safeguard
                delta_basis.copy_(physical_z)
            broadcast_from_main_(delta_basis)
        if should_stop:
            break

    delta_matrix = _joint_residual(delta_basis, right_factor)
    targets = target_init + delta_matrix
    if main_process:
        final_z_op_sq = torch.linalg.matrix_norm(
            delta_basis, ord=2
        ).square()
        if final_z_op_sq > op_bound * (
            1.0 + 64.0 * torch.finfo(delta_basis.dtype).eps
        ):
            raise RuntimeError(
                "Final Z hard operator constraint was violated: "
                f"{float(final_z_op_sq.item())} > {op_bound}"
            )
    distributed_barrier()
    return targets, delta_matrix, target_init


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    minus=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if fact_token_strategy != "subject_last":
        raise ValueError("SCRO requires fact_token=subject_last")
    layer_input, layer_output = repr_tools.get_reprs_at_word_tokens(
        track="both",
        subtoken="last",
        minus=minus,
        context_templates=context_templates,
        words=words,
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    return layer_input.detach(), layer_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose: bool = True,
    model_name: str = None,
) -> int:
    if fact_token_strategy != "subject_last":
        raise ValueError("SCRO requires fact_token=subject_last")
    index = repr_tools.get_words_idxs_in_templates(
        tok=tok,
        context_templates=[prompt],
        words=[subject],
        subtoken="last",
        model_name=model_name,
    )[0][0]
    if verbose:
        sentence = prompt.format(subject)
        print(
            f"Lookup index found: {index} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][index]),
        )
    return index
