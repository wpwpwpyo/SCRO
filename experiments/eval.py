import gc
import hashlib
import json
import os
import random
import re
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsets import (
    MENDQADataset,
    MultiCounterFactDataset,
    WikirecentDataset,
    HetionetDataset,
)
from experiments.py.eval_utils_counterfact_backdoor import (
    compute_generation_quality_counterfact,
    compute_prediction_quality_counterfact,
    compute_rewrite_quality_counterfact,
    n_gram_entropy as counterfact_ngram_entropy,
)
from experiments.py.eval_utils_wikirecent import compute_rewrite_quality_wikirecent
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from experiments.py.eval_utils_hetionet import (
    compute_prediction_quality_hetionet,
    compute_rewrite_quality_hetionet,
)
from util.generate import generate_standard
from util.globals import DATA_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DS_DICT = {
    "counterfact": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "zsre": (MENDQADataset, compute_rewrite_quality_zsre),
    "wikirecent": (WikirecentDataset, compute_rewrite_quality_wikirecent),
    "hetionet": (HetionetDataset, compute_rewrite_quality_hetionet),
}

PREDICTION_METRIC_KEYS = (
    "rewrite_prompts_correct",
    "paraphrase_prompts_correct",
    "neighborhood_prompts_correct",
    "rewrite_prompts_correct_token",
    "paraphrase_prompts_correct_token",
    "neighborhood_prompts_correct_token",
    "rewrite_prompts_probs",
    "paraphrase_prompts_probs",
    "neighborhood_prompts_probs",
)
GENERATION_ENTROPY_KEY = "generation_ngram_entropy"
GENERATION_TEXT_KEY = "generation_text"
LOCALITY_PRESERVATION_KEY = "locality_prompts_preservation"
LOCALITY_CACHE_VERSION = 1
LOCALITY_DATASET_KEYS = {"counterfact", "zsre", "hetionet"}

INCREMENTAL_EVAL_DICT = {
    "counterfact": {
        "prediction_method": compute_prediction_quality_counterfact,
        "generation_method": compute_generation_quality_counterfact,
        "entropy_method": counterfact_ngram_entropy,
    },
    "hetionet": {
        "prediction_method": compute_prediction_quality_hetionet,
    },
}

ALG_CHOICES = [
    "EAMET",
    "MEMIT",
    "PMET",
    "ROME",
    "FT",
    "MEND",
    "ALPHAEDIT",
    "ALPHAEDIT_ALIGNED",
    "EMMET",
    "MEMIT_MERGE",
    "SCRO",
    "Ori"
]


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def load_edited_model(
    edited_model_dir: str,
    device: int,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Path]:
    print(f"Loading edited model from {edited_model_dir}")
    model = AutoModelForCausalLM.from_pretrained(edited_model_dir)
    if device != -1:
        model = model.to(f"cuda:{device}")
    tok = AutoTokenizer.from_pretrained(edited_model_dir)

    print("Adding special tokens.")
    if "mistral" in str(model.config._name_or_path).lower():
        tok.pad_token = tok.eos_token
    else:
        if tok.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tok,
                model=model,
            )

        special_tokens_dict = {}

        if model.config.eos_token_id is not None:
            special_tokens_dict["eos_token"] = tok.convert_ids_to_tokens(model.config.eos_token_id)
        elif tok.eos_token is None:
            special_tokens_dict["eos_token"] = "</s>"

        if model.config.bos_token_id is not None:
            special_tokens_dict["bos_token"] = tok.convert_ids_to_tokens(model.config.bos_token_id)
        elif tok.bos_token is None:
            special_tokens_dict["bos_token"] = "<s>"

        if model.config.pad_token_id not in [-1, None]:
            special_tokens_dict["unk_token"] = tok.convert_ids_to_tokens(model.config.pad_token_id)
        elif tok.pad_token_id is not None:
            special_tokens_dict["unk_token"] = tok.convert_ids_to_tokens(tok.pad_token_id)
        elif tok.unk_token is None:
            special_tokens_dict["unk_token"] = "[UNK]"

        tok.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tok))

        if "gemma" not in str(model.config._name_or_path).lower():
            tok.add_bos_token = False
        else:
            tok.add_bos_token = True

    tok.padding_side = "right"
    print(f"padding side:{tok.padding_side}")
    return model, tok, edited_model_dir


def get_existing_case_ids(eval_dir: Path) -> set[int]:
    if not eval_dir.exists():
        return set()

    pattern = re.compile(r"^case_id(?P<case_id>\d+)\.json$")
    existing_case_ids = set()
    for case_file in eval_dir.iterdir():
        match = pattern.match(case_file.name)
        if match is None:
            continue
        existing_case_ids.add(int(match.group("case_id")))

    return existing_case_ids


def split_records_by_machine(
    records: List[Dict],
    total_machine: int,
    this_machine: int,
) -> List[Dict]:
    if total_machine <= 0:
        raise ValueError(f"total_machine must be positive, got {total_machine}")
    if this_machine < 0 or this_machine >= total_machine:
        raise ValueError(
            f"this_machine must be in [0, {total_machine - 1}], got {this_machine}"
        )

    return records[this_machine::total_machine]


def load_all_case_metrics(eval_dir: Path) -> List[Dict]:
    case_files = sorted(
        eval_dir.glob("case_id*.json"),
        key=lambda path: int(path.stem.replace("case_id", "")),
    )
    all_case_metrics = []
    for case_file in case_files:
        with open(case_file, "r") as f:
            all_case_metrics.append(json.load(f))

    return all_case_metrics


def _case_file_error(case_file: Path, message: str) -> ValueError:
    return ValueError(f"Invalid evaluation case file {case_file}: {message}")


def load_case_metrics_for_resume(
    case_file: Path,
    expected_case_id: int,
) -> Dict:
    try:
        with open(case_file, "r") as f:
            payload = json.load(f)
    except json.JSONDecodeError as exc:
        raise _case_file_error(
            case_file,
            f"malformed JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
        ) from exc
    except OSError as exc:
        raise _case_file_error(case_file, f"could not be read: {exc}") from exc

    if not isinstance(payload, dict):
        raise _case_file_error(case_file, "top-level JSON value must be an object")
    if "case_id" not in payload:
        raise _case_file_error(case_file, "missing top-level 'case_id'")
    if payload["case_id"] != expected_case_id:
        raise _case_file_error(
            case_file,
            f"case_id={payload['case_id']!r} does not match expected "
            f"case_id={expected_case_id!r}",
        )
    if "metrics" not in payload:
        raise _case_file_error(case_file, "missing top-level 'metrics'")
    if not isinstance(payload["metrics"], dict):
        raise _case_file_error(case_file, "'metrics' must be a JSON object")

    metrics = payload["metrics"]
    for key in PREDICTION_METRIC_KEYS + (
        GENERATION_ENTROPY_KEY,
        LOCALITY_PRESERVATION_KEY,
    ):
        if key in metrics and (
            isinstance(metrics[key], bool)
            or not isinstance(metrics[key], (int, float))
        ):
            raise _case_file_error(case_file, f"metric '{key}' must be numeric")
    if GENERATION_TEXT_KEY in metrics and (
        not isinstance(metrics[GENERATION_TEXT_KEY], list)
        or not all(isinstance(text, str) for text in metrics[GENERATION_TEXT_KEY])
    ):
        raise _case_file_error(
            case_file,
            f"metric '{GENERATION_TEXT_KEY}' must be a list of strings",
        )
    return payload


def write_json_atomic(path: Path, payload: Dict) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(temporary_path, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _safe_cache_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"


def locality_reference_cache_dir(model_name: str, ds_name: str) -> Path:
    return (
        Path(DATA_DIR)
        / "locality_reference_cache"
        / f"v{LOCALITY_CACHE_VERSION}"
        / _safe_cache_component(model_name)
        / _safe_cache_component(ds_name)
    )


def _locality_prompt_target_pairs(record: Dict, ds_key: str) -> List[Tuple[str, str]]:
    if "locality_prompt" in record and "locality_ground_truth" in record:
        pairs = [(record["locality_prompt"], record["locality_ground_truth"])]
    else:
        pairs = []
        target_true = record.get("requested_rewrite", {}).get("target_true", {}).get("str")
        for item in record.get("neighborhood_prompts", []):
            if isinstance(item, dict):
                prompt = item.get("prompt")
                target = item.get("target")
            else:
                prompt = item
                target = target_true
            pairs.append((prompt, target))

    normalized_pairs = []
    for prompt, target in pairs:
        if not isinstance(prompt, str) or not isinstance(target, str):
            raise ValueError(
                f"Invalid {ds_key} locality prompt/target pair for "
                f"case_id={record.get('case_id')}: {(prompt, target)!r}"
            )
        if target == "":
            raise ValueError(
                f"Empty locality target for case_id={record.get('case_id')}"
            )
        normalized_pairs.append((prompt, target))
    if not normalized_pairs:
        raise ValueError(
            f"No locality prompts for {ds_key} case_id={record.get('case_id')}"
        )
    return normalized_pairs


def _locality_record_signature(pairs: List[Tuple[str, str]]) -> str:
    serialized = json.dumps(
        pairs,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _teacher_forced_top1_sequences(
    model,
    tok,
    pairs: List[Tuple[str, str]],
    batch_size: int = 16,
) -> List[List[int]]:
    if not pairs:
        return []
    device = next(model.parameters()).device
    previous_padding_side = tok.padding_side
    was_training = model.training
    outputs = []
    model.eval()
    tok.padding_side = "left"
    try:
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            prompts = [prompt for prompt, _ in batch]
            prompt_targets = [f"{prompt} {target}" for prompt, target in batch]
            prompt_ids = tok(prompts, add_special_tokens=True)["input_ids"]
            encoded = tok(
                prompt_targets,
                padding=True,
                truncation=False,
                add_special_tokens=True,
                return_tensors="pt",
            )
            model_inputs = {
                key: value.to(device)
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask"}
            }
            with torch.no_grad():
                logits = model(**model_inputs).logits

            sequence_width = model_inputs["input_ids"].size(1)
            combined_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()
            for row, (prompt_tokens, combined_length) in enumerate(
                zip(prompt_ids, combined_lengths)
            ):
                padding_length = sequence_width - int(combined_length)
                target_start = padding_length + len(prompt_tokens)
                target_end = padding_length + int(combined_length)
                if target_start < 1 or target_end <= target_start:
                    raise RuntimeError(
                        "Could not align locality target tokens for "
                        f"prompt={batch[row][0]!r}, target={batch[row][1]!r}"
                    )
                predicted = logits[
                    row,
                    target_start - 1 : target_end - 1,
                ].argmax(dim=-1)
                outputs.append(predicted.detach().cpu().tolist())
    finally:
        tok.padding_side = previous_padding_side
        model.train(was_training)
    return outputs


def _load_locality_reference(
    cache_file: Path,
    signature: str,
    expected_pair_count: int,
) -> Optional[List[List[int]]]:
    if not cache_file.exists():
        return None
    try:
        with cache_file.open() as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    outputs = payload.get("top1_token_ids")
    if (
        payload.get("version") != LOCALITY_CACHE_VERSION
        or payload.get("record_signature") != signature
        or not isinstance(outputs, list)
        or len(outputs) != expected_pair_count
        or not all(
            isinstance(sequence, list)
            and sequence
            and all(isinstance(token_id, int) for token_id in sequence)
            for sequence in outputs
        )
    ):
        return None
    return outputs


def ensure_locality_reference_cache(
    model,
    tok,
    records: List[Dict],
    model_name: str,
    ds_name: str,
    total_machine: int = 1,
    this_machine: int = 0,
) -> Path:
    ds_key = "hetionet" if "hetionet" in ds_name else ds_name
    if ds_key not in LOCALITY_DATASET_KEYS:
        raise ValueError(f"Locality preservation is not defined for {ds_name}")

    cache_dir = locality_reference_cache_dir(model_name, ds_name)
    cache_dir.mkdir(parents=True, exist_ok=True)
    assigned_records = split_records_by_machine(records, total_machine, this_machine)
    pending = []
    flat_pairs = []
    for record in assigned_records:
        pairs = _locality_prompt_target_pairs(record, ds_key)
        signature = _locality_record_signature(pairs)
        cache_file = cache_dir / f"case_id{record['case_id']}.json"
        if _load_locality_reference(cache_file, signature, len(pairs)) is not None:
            continue
        start = len(flat_pairs)
        flat_pairs.extend(pairs)
        pending.append((record["case_id"], signature, cache_file, start, len(flat_pairs)))

    if pending:
        print(
            f"Computing original-model locality references for {len(pending)} "
            f"{ds_name} cases ({len(flat_pairs)} prompt/target pairs)"
        )
        flat_outputs = _teacher_forced_top1_sequences(model, tok, flat_pairs)
        for case_id, signature, cache_file, start, end in pending:
            write_json_atomic(
                cache_file,
                {
                    "version": LOCALITY_CACHE_VERSION,
                    "case_id": case_id,
                    "record_signature": signature,
                    "top1_token_ids": flat_outputs[start:end],
                },
            )
    else:
        print(f"Reusing original-model locality references from {cache_dir}")
    return cache_dir


def compute_locality_preservation(
    model,
    tok,
    record: Dict,
    ds_key: str,
    reference_cache_dir: Path,
) -> float:
    pairs = _locality_prompt_target_pairs(record, ds_key)
    signature = _locality_record_signature(pairs)
    cache_file = reference_cache_dir / f"case_id{record['case_id']}.json"
    reference_outputs = _load_locality_reference(cache_file, signature, len(pairs))
    if reference_outputs is None:
        raise RuntimeError(
            "Missing or stale original-model locality reference: "
            f"{cache_file}. Build the reference cache before evaluating the edited model."
        )
    edited_outputs = _teacher_forced_top1_sequences(model, tok, pairs)
    pair_scores = []
    for pair_index, (reference, edited) in enumerate(
        zip(reference_outputs, edited_outputs)
    ):
        if len(reference) != len(edited):
            raise RuntimeError(
                "Locality token count changed between original and edited model for "
                f"case_id={record['case_id']}, pair={pair_index}: "
                f"{len(reference)} vs {len(edited)}"
            )
        pair_scores.append(float(np.mean(np.equal(reference, edited))))
    return float(np.mean(pair_scores))


def load_evaluation_records(
    tok,
    ds_name: str,
    dataset_size_limit: Optional[int],
    model_name_config: str,
    randomize_editing_sequence: bool = False,
    shuffle_seed: int = 0,
) -> Tuple[str, List[Dict]]:
    ds_key = "hetionet" if "hetionet" in ds_name else ds_name
    ds_class, _ = DS_DICT[ds_key]
    ds = ds_class(
        DATA_DIR,
        tok=tok,
        size=dataset_size_limit,
        trigger=ds_name,
        model_name_config=model_name_config,
        randomize_editing_sequence=randomize_editing_sequence,
        shuffle_seed=shuffle_seed,
        relation_count=ds_name.split("_")[-1] if "_" in ds_name else None,
    )
    return ds_key, [ds[i] for i in range(len(ds))]


def complete_cached_case_metrics(
    case_file: Path,
    expected_case_id: int,
    model,
    tok,
    record: Dict,
    assigned_prefix_len: int,
    cached_prefix,
    incremental_eval_config: Optional[Dict],
    ds_key: str,
    locality_cache_dir: Path,
) -> Tuple[Dict, List[str]]:
    payload = load_case_metrics_for_resume(case_file, expected_case_id)
    metrics = payload["metrics"]
    filled_fields = []

    if incremental_eval_config is not None:
        missing_prediction_keys = [
            key for key in PREDICTION_METRIC_KEYS if key not in metrics
        ]
        if missing_prediction_keys:
            prediction_metrics = incremental_eval_config["prediction_method"](
                model,
                tok,
                record,
                assigned_prefix_len,
                cached_prefix=cached_prefix,
            )
            unavailable_keys = [
                key for key in missing_prediction_keys if key not in prediction_metrics
            ]
            if unavailable_keys:
                raise RuntimeError(
                    "Prediction evaluator did not return required fields: "
                    + ", ".join(unavailable_keys)
                )
            for key in missing_prediction_keys:
                metrics[key] = prediction_metrics[key]
                filled_fields.append(key)

        generation_enabled = "generation_method" in incremental_eval_config
        text_missing = generation_enabled and GENERATION_TEXT_KEY not in metrics
        entropy_missing = generation_enabled and GENERATION_ENTROPY_KEY not in metrics
        if text_missing:
            generation_metrics = incremental_eval_config["generation_method"](
                model,
                tok,
                record,
            )
            if GENERATION_TEXT_KEY not in generation_metrics:
                raise RuntimeError(
                    f"Generation evaluator did not return '{GENERATION_TEXT_KEY}'"
                )
            metrics[GENERATION_TEXT_KEY] = generation_metrics[GENERATION_TEXT_KEY]
            filled_fields.append(GENERATION_TEXT_KEY)
            if entropy_missing:
                if GENERATION_ENTROPY_KEY not in generation_metrics:
                    raise RuntimeError(
                        f"Generation evaluator did not return '{GENERATION_ENTROPY_KEY}'"
                    )
                metrics[GENERATION_ENTROPY_KEY] = generation_metrics[
                    GENERATION_ENTROPY_KEY
                ]
                filled_fields.append(GENERATION_ENTROPY_KEY)
        elif entropy_missing:
            metrics[GENERATION_ENTROPY_KEY] = float(
                incremental_eval_config["entropy_method"](
                    metrics[GENERATION_TEXT_KEY]
                )
            )
            filled_fields.append(GENERATION_ENTROPY_KEY)

    if ds_key in LOCALITY_DATASET_KEYS and LOCALITY_PRESERVATION_KEY not in metrics:
        metrics[LOCALITY_PRESERVATION_KEY] = compute_locality_preservation(
            model,
            tok,
            record,
            ds_key,
            locality_cache_dir,
        )
        filled_fields.append(LOCALITY_PRESERVATION_KEY)

    return payload, filled_fields


def load_execution_time(results_dir: str) -> float:
    metadata_path = Path(results_dir) / "edit-metadata.json"
    if not metadata_path.exists():
        return 0

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata.get("execution_time", 0)


def build_overall_record(
    alg_name: str,
    ds_name: str,
    all_case_metrics,
    execution_time: float,
    assigned_prefix_len: int,
    dataset_size: Optional[int],
):
    if len(all_case_metrics) == 0:
        raise ValueError("No case metrics were produced.")

    def avg(metric_name: str):
        return sum(metrics["metrics"][metric_name] for metrics in all_case_metrics) / len(all_case_metrics)

    overall_record = {
        "alg_name": alg_name,
        "execution_time": execution_time,
        "assigned_prefix_len": assigned_prefix_len,
        "num_edits": dataset_size,
    }

    if "counterfact" in ds_name or "hetionet" in ds_name:
        overall_record.update(
            {
                "rewrite_prompts_correct": avg("rewrite_prompts_correct"),
                "paraphrase_prompts_correct": avg("paraphrase_prompts_correct"),
                "neighborhood_prompts_correct": avg("neighborhood_prompts_correct"),
                "rewrite_prompts_correct_token": avg("rewrite_prompts_correct_token"),
                "paraphrase_prompts_correct_token": avg("paraphrase_prompts_correct_token"),
                "neighborhood_prompts_correct_token": avg("neighborhood_prompts_correct_token"),
                "rewrite_prompts_probs": avg("rewrite_prompts_probs"),
                "paraphrase_prompts_probs": avg("paraphrase_prompts_probs"),
                "neighborhood_prompts_probs": avg("neighborhood_prompts_probs"),
                LOCALITY_PRESERVATION_KEY: avg(LOCALITY_PRESERVATION_KEY),
            }
        )
        if "counterfact" in ds_name:
            overall_record["generation_ngram_entropy"] = avg(
                "generation_ngram_entropy"
            )
    elif "zsre" in ds_name:
        overall_record.update(
            {
                "rewrite_prompts_correct": avg("rewrite_prompts_correct"),
                "paraphrase_prompts_correct": avg("paraphrase_prompts_correct"),
                "rewrite_prompts_token": avg("rewrite_prompts_token"),
                "paraphrase_prompts_token": avg("paraphrase_prompts_token"),
                "neighborhood_prompts_correct": avg("neighborhood_prompts_correct"),
                LOCALITY_PRESERVATION_KEY: avg(LOCALITY_PRESERVATION_KEY),
            }
        )
    elif "wikirecent" in ds_name:
        overall_record.update(
            {
                "rewrite_prompts_correct": avg("rewrite_prompts_correct"),
                "portability_prompts_correct": avg("portability_prompts_correct"),
                "locality_prompts_correct": avg("locality_prompts_correct"),
                "generation_ngram_entropy": avg("generation_ngram_entropy"),
            }
        )

    return overall_record


def evaluate_loaded_model(
    model,
    tok,
    alg_name: str,
    ds_name: str,
    dataset_size_limit: Optional[int],
    results_dir: str,
    model_name: Optional[str] = None,
    assigned_prefix_len: int = 10,
    randomize_editing_sequence: bool = False,
    shuffle_seed: int = 0,
    total_machine: int = 1,
    this_machine: int = 0,
    execution_time: Optional[float] = None,
):
    distributed_eval = dist.is_available() and dist.is_initialized()
    if distributed_eval:
        total_machine = dist.get_world_size()
        this_machine = dist.get_rank()

    run_dir = Path(results_dir)
    model_name_config = model_name if model_name is not None else model.config._name_or_path
    ds_key, all_records = load_evaluation_records(
        tok=tok,
        ds_name=ds_name,
        dataset_size_limit=dataset_size_limit,
        model_name_config=model_name_config,
        randomize_editing_sequence=randomize_editing_sequence,
        shuffle_seed=shuffle_seed,
    )
    _, ds_eval_method = DS_DICT[ds_key]
    locality_cache_dir = (
        locality_reference_cache_dir(model_name_config, ds_name)
        if ds_key in LOCALITY_DATASET_KEYS
        else None
    )

    eval_dir = run_dir / "eval" / f"ds_{dataset_size_limit}_prefix_{assigned_prefix_len}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    assigned_records = split_records_by_machine(
        all_records, total_machine=total_machine, this_machine=this_machine
    )

    print(f"Found {len(all_records)} total records in dataset")
    print(f"Found {len(all_records)} pending records after resume filtering")
    print(
        f"Machine {this_machine}/{total_machine} will evaluate "
        f"{len(assigned_records)} records on {next(model.parameters()).device}"
    )

    prefixes = None
    if assigned_prefix_len != 0:
        prefixes = [
            prefix + ". "
            for prefix in generate_standard(
                model,
                ["The", "Therefore", "You", "However", "And", "While", "To", "Nevertheless", "Never", "He"],
                tok,
                max_new_tokens=assigned_prefix_len,
                do_sample=True,
            )
        ]

    start = time()
    incremental_eval_config = INCREMENTAL_EVAL_DICT.get(ds_key)
    for ind, record in enumerate(assigned_records):
        if ind % 10 == 0:
            print(f"evaluating {ind}th record")
        case_out_file = eval_dir / f"case_id{record['case_id']}.json"
        if case_out_file.exists():
            cached_payload, filled_fields = complete_cached_case_metrics(
                case_file=case_out_file,
                expected_case_id=record["case_id"],
                model=model,
                tok=tok,
                record=record,
                assigned_prefix_len=assigned_prefix_len,
                cached_prefix=prefixes,
                incremental_eval_config=incremental_eval_config,
                ds_key=ds_key,
                locality_cache_dir=locality_cache_dir,
            )
            if filled_fields:
                write_json_atomic(case_out_file, cached_payload)
                print(
                    f"Completed cached case_id={record['case_id']} fields: "
                    + ", ".join(filled_fields)
                )
            continue
        metrics = ds_eval_method(
            model,
            tok,
            record,
            assigned_prefix_len,
            cached_prefix=prefixes,
        )
        if locality_cache_dir is not None:
            metrics[LOCALITY_PRESERVATION_KEY] = compute_locality_preservation(
                model,
                tok,
                record,
                ds_key,
                locality_cache_dir,
            )
        case_payload = {
            "case_id": record["case_id"],
            "metrics": metrics,
        }
        write_json_atomic(case_out_file, case_payload)

    if distributed_eval:
        dist.barrier()
        if this_machine != 0:
            dist.barrier()
            return None

    all_case_metrics = load_all_case_metrics(eval_dir)
    overall_record = build_overall_record(
        alg_name=alg_name,
        ds_name=ds_name,
        all_case_metrics=all_case_metrics,
        execution_time=(
            load_execution_time(results_dir)
            if execution_time is None
            else execution_time
        ),
        assigned_prefix_len=assigned_prefix_len,
        dataset_size=len(all_case_metrics),
    )
    with open(eval_dir / "all-metrics-result.json", "w") as f:
        json.dump(overall_record, f, indent=1)

    print("Stage2 evaluation metrics", overall_record)
    print("Evaluation took", time() - start)
    if distributed_eval:
        dist.barrier()
    return overall_record


def main(
    alg_name: str,
    ds_name: str,
    dataset_size_limit: Optional[int],
    results_dir: str,
    model_name: Optional[str] = None,
    assigned_prefix_len: int = 10,
    store_case_metrics: bool = False,
    randomize_editing_sequence: bool = False,
    shuffle_seed: int = 0,
    total_machine: int = 1,
    this_machine: int = 0,
    device: int = 0,
    original: bool = False
):
    os.chdir(PROJECT_ROOT)
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    set_seed(42)

    if model_name is None:
        raise ValueError(
            "--model_name is required to compute original-model locality preservation"
        )
    base_model_dir = PROJECT_ROOT.parent / "models" / model_name
    ds_key = "hetionet" if "hetionet" in ds_name else ds_name
    base_model = None
    base_tok = None
    if ds_key in LOCALITY_DATASET_KEYS:
        base_model, base_tok, _ = load_edited_model(base_model_dir, device)
        _, base_records = load_evaluation_records(
            tok=base_tok,
            ds_name=ds_name,
            dataset_size_limit=dataset_size_limit,
            model_name_config=model_name,
            randomize_editing_sequence=randomize_editing_sequence,
            shuffle_seed=shuffle_seed,
        )
        ensure_locality_reference_cache(
            model=base_model,
            tok=base_tok,
            records=base_records,
            model_name=model_name,
            ds_name=ds_name,
            total_machine=total_machine,
            this_machine=this_machine,
        )

    if original:
        results_dir = '/'.join(results_dir.split('/')[:3]) + '/pre/baseline'
        if base_model is None:
            base_model, base_tok, _ = load_edited_model(base_model_dir, device)
        model, tok = base_model, base_tok
    else:
        edited_model_dir = Path(results_dir) / "edited_model" / f"ds{dataset_size_limit}"
        if not edited_model_dir.exists():
            raise FileNotFoundError(f"Edited model not found at {edited_model_dir}")
        if base_model is not None:
            del base_model, base_tok
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        model, tok, _ = load_edited_model(edited_model_dir, device)

    return evaluate_loaded_model(
        model=model,
        tok=tok,
        alg_name=alg_name,
        ds_name=ds_name,
        dataset_size_limit=dataset_size_limit,
        results_dir=results_dir,
        model_name=model_name,
        assigned_prefix_len=assigned_prefix_len,
        randomize_editing_sequence=randomize_editing_sequence,
        shuffle_seed=shuffle_seed,
        total_machine=total_machine,
        this_machine=this_machine,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=ALG_CHOICES,
        default="EAMET",
        help="Editing algorithm used to produce the edited model.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Original model name, used only for dataset tokenization rules.",
    )
    parser.add_argument(
        "--ds_name",
        # choices=["counterfact", "zsre", "wikirecent"],
        default="counterfact",
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing edited_model and evaluation outputs.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate dataset to first n records.",
    )
    parser.add_argument(
        "--assigned_prefix_len",
        type=int,
        default=10,
        help="Length of testing prefix.",
    )
    parser.add_argument(
        "--store_case_metrics",
        dest="store_case_metrics",
        action="store_true",
        help="Store per-case metrics.",
    )
    parser.add_argument(
        "--randomize_editing_sequence",
        dest="randomize_editing_sequence",
        action="store_true",
        help="Randomize evaluation sequence.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help="Shuffle seed.",
    )
    parser.add_argument(
        "--total_machine",
        type=int,
        default=1,
        help="Total number of machines used to split pending records.",
    )
    parser.add_argument(
        "--this_machine",
        type=int,
        default=0,
        help="Index of the current machine, starting from 0.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index assigned to this process.",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="Test original model.",
    )
    args = parser.parse_args()

    main(
        alg_name=args.alg_name,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        results_dir=args.results_dir,
        model_name=args.model_name,
        assigned_prefix_len=args.assigned_prefix_len,
        store_case_metrics=args.store_case_metrics,
        randomize_editing_sequence=args.randomize_editing_sequence,
        shuffle_seed=args.shuffle_seed,
        total_machine=args.total_machine,
        this_machine=args.this_machine,
        device=args.device,
        original=args.original
    )
