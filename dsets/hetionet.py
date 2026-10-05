import json
import typing
from pathlib import Path

import torch
from torch.utils.data import Dataset
import copy
import random

from util.globals import *

REMOTE_ROOT = f"{REMOTE_ROOT_URL}/data/dsets"

class HetionetDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        multi: bool = False,
        size: typing.Optional[int] = None,
        data_range: list = None,
        *args,
        **kwargs,
    ):
        data_dir = Path(data_dir)
        hetionet_loc = data_dir / f"hetionet/hetionet_{kwargs['relation_count']}.json" 
        self.model_name_config = kwargs.get("model_name_config", None)
        
        if not hetionet_loc.exists():
            assert False, f"Hetionet dataset not found at {hetionet_loc}"

        with open(hetionet_loc, "r") as f:
            self.data = json.load(f)
        
        data = []
        ind = 0
        for d in self.data:
            subject = d["subject_name"]
            for record in d["fine_relations"]:
                data.append(
                    {
                        "case_id": ind,
                        "requested_rewrite": {
                            "prompt": record["prompt"].replace(subject, "{}"),
                            "subject": subject,
                            "target_new": {"str": record["object_list"][0]},
                            "target_true": {"str": "<|endoftext|>"},
                            # "train_paraphrase_prompts": record["train_paraphrase_prompt"],
                            "paraphrase_prompts": record["paraphrase_prompt"],
                        },
                        "paraphrase_prompts": record["paraphrase_prompt"],
                        "neighborhood_prompts":[
                            v for k, v in record["neighborhood_prompts"].items()
                        ],
                        "neighborhood_prompts":[
                            {
                                "prompt": v["prompt"],
                                "target": v["ans_list"][0]
                            } for k, v in record["neighborhood_prompts"].items()
                        ],
                        "generation_prompts": record["generation_prompt"],
                    }
                )
                ind += 1
        self.data = data

        if size is not None and data_range is not None:
            assert False, f"size and range can not be not None together"
        elif size is not None:
            self.data = self.data[:size]
        elif data_range is not None:
            self.data = [self.data[i] for i in data_range]

        if kwargs.get("randomize_editing_sequence", False):
            random.seed(kwargs.get("shuffle_seed", 0))
            random.shuffle(self.data)

        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]