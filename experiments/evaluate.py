from experiments import edit as edit_entry
from experiments import eval as eval_entry
import torch

import numpy as np
import random
import os
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

def main(args):
    set_seed(42)

    edit_entry.main(
        alg_name=args.alg_name,
        model_name=args.model_name,
        hparams_fname=args.hparams_fname,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        results_dir=args.results_dir,
        continue_from_run=args.continue_from_run,
        use_cache=args.use_cache,
        motivation_exp=args.motivation_exp,
        cache_motivation=args.cache_motivation,
        randomize_editing_sequence=args.randomize_editing_sequence,
        shuffle_seed=args.shuffle_seed,
    )
    eval_entry.main(
        alg_name=args.alg_name,
        ds_name=args.ds_name,
        dataset_size_limit=args.dataset_size_limit,
        results_dir=args.results_dir,
        model_name=args.model_name,
        assigned_prefix_len=args.assigned_prefix_len,
        store_case_metrics=args.store_case_metrics,
        randomize_editing_sequence=args.randomize_editing_sequence,
        shuffle_seed=args.shuffle_seed,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper. Prefer running experiments.edit first, "
            "then experiments.eval."
        )
    )
    parser.add_argument(
        "--alg_name",
        choices=eval_entry.ALG_CHOICES,
        default="EAMET",
        required=True,
        help="Editing algorithm to use.",
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        required=True,
        help="Model to edit.",
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        required=True,
        help="Name of hyperparameters file, located in hparams/<alg_name>.",
    )
    parser.add_argument(
        "--ds_name",
        choices=["counterfact", "zsre", "wikirecent"],
        default="counterfact",
        help="Dataset to edit and evaluate.",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory where edited model and evaluation results are stored.",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, load params.json from --results_dir.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate dataset to first n records.",
    )
    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs.",
    )
    parser.add_argument(
        "--assigned_prefix_len",
        type=int,
        default=10,
        help="Length of testing prefix.",
    )
    parser.add_argument(
        "--motivation_exp",
        dest="motivation_exp",
        action="store_true",
        help="Activate motivation experiments.",
    )
    parser.add_argument(
        "--cache_motivation",
        dest="cache_motivation",
        action="store_true",
        help="Cache motivation experiment data.",
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
        help="Randomize editing/evaluation sequence.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help="Shuffle seed.",
    )
    args = parser.parse_args()

    if args.alg_name == "MEND":
        from baselines.mend import MENDHyperParams, MendRewriteExecutor

        edit_entry.ALG_DICT["MEND"] = (MENDHyperParams, MendRewriteExecutor().apply_to_model)

    main(args)
