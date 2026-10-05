import json
from pathlib import Path
from time import time
from typing import Optional

from glue_eval.glue_eval import GLUEEval
from experiments.eval import load_edited_model

import numpy as np
import random
import torch
import os

def run_glue_eval(
    model,
    tok,
    output_dir: Path,
    output_name: str,
    number_of_tests: int = 100,
    nli_flag: bool = True,
    sst_flag: bool = True,
    cola_flag: bool = True,
    rte_flag: bool = True,
    mmlu_flag: bool = True,
    mrpc_flag: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    glue_results = {"edit_num": -1}
    out_file = output_dir / f"{output_name}.json"
    glue_eval = GLUEEval(model, tok, number_of_tests=number_of_tests)
    glue_results = glue_eval.evaluate(
        glue_results,
        str(out_file),
        nli_flag=nli_flag,
        sst_flag=sst_flag,
        cola_flag=cola_flag,
        rte_flag=rte_flag,
        mmlu_flag=mmlu_flag,
        mrpc_flag=mrpc_flag,
    )

    output_filename = str(out_file).replace(".json", "_glue.json")
    with open(output_filename, "w") as f:
        json.dump(glue_results, f, indent=4)

    return glue_results


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

def main(
    results_dir: str,
    dataset_size_limit: Optional[int],
    alg_name: Optional[str] = None,
    number_of_tests: int = 100,
    output_name: str = "base",
    device : int = None,
    original: bool = False,
    model_name: str = None
):
    set_seed(42)

    start = time()

    if original:
        results_dir = '/'.join(results_dir.split('/')[:3]) + '/pre/baseline'
        edited_model_dir = f'../models/{model_name}'
        model, tok, _ = load_edited_model(edited_model_dir, device)
        model.config._name_or_path = model.config._name_or_path.split('/')[-1]
    else:
        edited_model_dir = Path(results_dir) / "edited_model" / f"ds{dataset_size_limit}"
        model, tok, _ = load_edited_model(edited_model_dir, device)
        model.config._name_or_path = model.config._name_or_path.split('/')[1]

    model.generation_config.max_new_tokens = None
    
    output_dir = Path(results_dir) / "glue_eval" / f"ds{dataset_size_limit}"
    glue_results = run_glue_eval(
        model,
        tok,
        output_dir,
        output_name,
        number_of_tests=number_of_tests,
    )

    if alg_name is not None:
        metadata_file = output_dir / f"{output_name}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(
                {
                    "alg_name": alg_name,
                    "dataset_size_limit": dataset_size_limit,
                    "number_of_tests": number_of_tests,
                    "output_name": output_name,
                },
                f,
                indent=4,
            )

    print("GLUE evaluation took", time() - start)
    return glue_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing edited_model and GLUE outputs.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Dataset size used by edit.py. This selects edited_model/ds{dataset_size_limit}.",
    )
    parser.add_argument(
        "--alg_name",
        # choices=ALG_CHOICES,
        default=None,
        help="Editing algorithm used to produce the edited model.",
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help="Original model name, used only for dataset tokenization rules.",
    )
    parser.add_argument(
        "--number_of_tests",
        type=int,
        default=100,
        help="Number of samples per GLUE task.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="base",
        help="Output prefix under results_dir/glue_eval/ds_{dataset_size_limit}.",
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
        results_dir=args.results_dir,
        dataset_size_limit=args.dataset_size_limit,
        alg_name=args.alg_name,
        number_of_tests=args.number_of_tests,
        output_name=args.output_name,
        device=args.device,
        original=args.original,
        model_name=args.model_name
    )
