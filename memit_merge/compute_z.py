from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .memit_hparams import MEMIT_MergeHyperParams
from statistics import fmean 
from operator import attrgetter


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMIT_MergeHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update in MEMIT-Merge.
    
    This function implements the core computation for finding the optimal value vector
    that minimizes the editing objective. It processes multiple requests and computes
    weighted gradients to find the best update direction.
    
    Args:
        model: The language model to edit
        tok: Tokenizer for the model
        requests: List of editing requests containing prompts and target outputs
        hparams: Hyperparameters for the MEMIT-Merge algorithm
        layer: Layer index where the edit will be applied
        context_templates: Templates for generating context variations
    
    Returns:
        Tuple of (key_matrix, value_matrix) for the rank-1 update
    """
    BATCH_SIZE = hparams.gradient_batchsize

    # Get model parameters
    lm_w, ln_f = (
        # nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        attrgetter(hparams.lm_head_module)(model).weight.T,
        # nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
        if lm_b == None:
            lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")
    device = model.device
    model_name = str(model.config._name_or_path).lower()

    # Create unique requests list and their corresponding weights
    new_requests = []
    request_weights = []
    for request in requests:
        if request not in new_requests:
            request_weights.append(1)
            new_requests.append(request)
        else:
            request_weights[new_requests.index(request)] += 1
    requests = new_requests
    request_weights = torch.tensor(request_weights, device=device)



    # Tokenize target into list of int token IDs
    target_idss = []
    target_prefixes = []
    for request in requests:
        target_ids = tok(request["target_new"], return_tensors="pt").to(device)["input_ids"][0]

        if (
            "mistral" in model_name
            or "qwen" in model_name
            or "deepseek" in model_name
        ):
            opt_target_ids = target_ids
        elif (
            "llama" in str(type(tok)).lower()
            or "llama-3.1" in model_name
            or "gemma" in model_name
        ):
            opt_target_ids = target_ids[1:]
        else:
            opt_target_ids = target_ids

        target_idss.append(opt_target_ids)
        if "llama-3.1" in model_name:
            tok_dcd = tok.decode(target_ids[:-1])
            bos_loc = tok_dcd.find("<|begin_of_text|>")
            if bos_loc != -1:
                tok_dcd = tok_dcd[bos_loc + len("<|begin_of_text|>"):]
            target_prefixes.append(tok_dcd)
        elif "gemma" in model_name:
            target_prefixes.append(tok.decode(opt_target_ids[:-1]))
        else:
            target_prefixes.append(tok.decode(target_ids[:-1]))
    # Compile list of rewriting and KL x/y pairs
    # Mix rewriting prompts into multiple requests
    rewriting_prompts, kl_prompts = [], []
    rewriting_subjects = []
    request_indicator_ls = []
    request_indicator_idxls = [0]
    for idx, (request, target_ids, target_prefix) in enumerate(zip(requests, target_idss, target_prefixes)):
        req_prompt = request["prompt"]
        prefixed_prompts = [
            context.format(req_prompt) + target_prefix
            for context_types in context_templates
            for context in context_types
        ]
        request_indicator_ls += [idx]*len(prefixed_prompts)  # to indicate which request the prompt belongs to
        request_indicator_idxls.append(request_indicator_idxls[-1] + len(prefixed_prompts))
        rewriting_prompts += prefixed_prompts
        rewriting_subjects += [request["subject"]] * len(prefixed_prompts)
    kl_prompts = ["{} is a"]  # KL prompts should be sufficient with one
    
    KL_LOCA_DEBUG = False
    if not KL_LOCA_DEBUG:
        kl_prompts = ["{} is a"]  # KL prompts should be sufficient with one
        kl_subjects = [requests[0]["subject"]]
    else:
        kl_prompts = ["{} is a"]*len(requests)  # KL prompts should be sufficient with one
        kl_subjects = [request["subject"] for request in requests]
    all_prompts = rewriting_prompts + kl_prompts
    subjects = rewriting_subjects + kl_subjects
    # Each knowledge edit has 6 rewriting_prompts, and there is one KL prompt in the large batch
    # Large batch refers to edit_size, which is the number of same-subject edits in the batch, unrelated to BATCH_SIZE=10
    # For example, editing 20 items simultaneously, even with BATCH_SIZE=10, all_prompts would be 121 items


    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=device)
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=device)
    else:
        raise NotImplementedError
    
    target_init, kl_distr_init = None, None
    # model.register_buffer("delta", delta)
    input_tok = tok(
        [prompt.format(subject) for prompt, subject in zip(all_prompts, subjects)],
        return_tensors="pt",
        padding=True,
    )

    # input_tok = input_tok.to(delta.device)

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    # Modify target ids for different requests
    for i in range(len(rewriting_prompts)):
        target_ids = target_idss[request_indicator_ls[i]]
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt,
            subject,
            tok,
            hparams.fact_token,
            verbose=(i == 0),
            model_name=model_name,
        )
        for i, (prompt, subject) in enumerate(zip(all_prompts, subjects))
    ]
    batch_idx = 0
    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init
        nonlocal batch_idx

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()  # More like the origin of key values, forward inference to value
                # Since a batch has multiple edits, there may be multiple initial values (if the sentences before the subject are not exactly the same, there will be multiple)
                # Currently using the init of the first edit as the init for all edits, may not be appropriate
            cur_lookup_idxs = lookup_idxs[batch_idx*BATCH_SIZE:(batch_idx+1)*BATCH_SIZE]
            batch_idx += 1
            # Add intervened delta
            for i, idx in enumerate(cur_lookup_idxs):
                # Note: delta and cur_out often have device conflicts
                # The main reason seems to be that cur_out's device is dynamically distributed during training, while delta is fixed
                # One method is to fix cur_out device. Another is to distribute delta to every device. Or inefficiently keep moving delta to cur_out
                if len(cur_lookup_idxs)!=len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta.to(cur_out[0].device)
                else:
                    cur_out[0][i, idx, :] += delta.to(cur_out[0].device)

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    init_edit = True
    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        kl_loss = torch.tensor(0, device=device)
        loss_sum = 0
        batch_idx = 0
        pre_kl_ind = 0
        for ind in range(0, len(lookup_idxs), BATCH_SIZE):
            inp_tok = {
                "input_ids": input_tok["input_ids"][ind:ind + BATCH_SIZE].to(device),
                "attention_mask": input_tok["attention_mask"][ind:ind + BATCH_SIZE].to(device),
            }
            # Forward propagation
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.layer_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:

                logits = model(**inp_tok).logits
            output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
            part_rewriting_targets = rewriting_targets[ind:ind+BATCH_SIZE]
            if output.shape[1] != part_rewriting_targets.shape[1]:
                # This should be due to the difference between GPT and llama structures? Maybe not
                output=torch.transpose(output, 0, 1)

            if ind+BATCH_SIZE > len(lookup_idxs)-len(kl_prompts):
                # Compute distribution for KL divergence, only compute kl logits in the last batch
                if len(lookup_idxs)-len(kl_prompts) > ind:
                    kl_logits = torch.stack(
                        [
                            logits[len(lookup_idxs)-len(kl_prompts)-ind+i, idx, :]
                            for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :ind+BATCH_SIZE])
                        ],
                        dim=0,
                    )
                    full_repr = output[:len(lookup_idxs)-len(kl_prompts)-ind]
                else:
                    # Pure kl_prompt calculation
                    kl_logits = torch.stack(
                        [
                            logits[i, idx, :]
                            for i, idx in enumerate(lookup_idxs[ind:ind+BATCH_SIZE])
                        ],
                        dim=0,
                    )
                    full_repr = output
                kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                if init_edit:
                    if kl_distr_init is None:
                        kl_distr_init = kl_log_probs.detach().clone()
                    else:
                        kl_distr_init = torch.cat([kl_distr_init,kl_log_probs.detach().clone()])

                kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                    kl_distr_init[pre_kl_ind:pre_kl_ind+len(kl_log_probs)], kl_log_probs, log_target=True, reduction="batchmean"
                )
                pre_kl_ind = pre_kl_ind+len(kl_log_probs)

            else:
                full_repr = output    

            log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
            loss = torch.gather(
                log_probs,
                2,
                torch.where(part_rewriting_targets != -100, part_rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
            ).squeeze(2)
            # Get corresponding weight for loss
            loss_weights = request_weights[request_indicator_ls[ind:ind+BATCH_SIZE]]
            # Multiply weight to loss
            loss = loss * loss_weights.unsqueeze(1)

            mask = (part_rewriting_targets != -100).float()
            # Aggregate total losses
            # More detailed approach would be to calculate the exact mean of target_ids.size(0) for each loss, but for simplification, can directly divide by the mean length of target_idss
            target_idsize_ls = [target_idss[request_indicator_ls[i]].size(0) for i in range(ind, min(ind+BATCH_SIZE, len(request_indicator_ls)))]
            
            nll_loss_each = torch.div(-(loss * mask.to(loss.device)).sum(1), torch.tensor(target_idsize_ls, device=loss.device))
            nll_loss = nll_loss_each.mean()
            


            target_init = target_init.to(delta.device)
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            floss = torch.tensor(0.0, device=device)
            if not torch.isnan(weight_decay):
                floss += weight_decay.to(floss.device)

            if not torch.isnan(kl_loss) and kl_loss:
                floss += kl_loss.to(floss.device)
            if not torch.isnan(nll_loss):
                floss += nll_loss.to(floss.device)

            print(
                f"loss {np.round(floss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg prob of [{request['target_new']}] "
                f"{torch.exp(-nll_loss_each).mean().item()} at index {ind}"  
            )
            floss.backward()
            loss_sum += floss.item()
            
        if loss_sum < 5e-2:
            break

        # Backpropagate
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
                

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    track=None,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
            )
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
            )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
    model_name: str = None,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
            model_name=model_name,
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
