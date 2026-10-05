# Reproducing Baselines and SCRO

This example uses Qwen2.5-7B and the local HetionetEdit dataset.

## Docker image installation

With Docker and the NVIDIA Container Toolkit installed on the host, build the image from the included `docker/Dockerfile` and `docker/environment.yml`. The image and its Conda environment are both named `scro`.

```bash
docker build -t scro:latest ./docker
docker run --rm -it --gpus all \
  -v "$PWD":/workspace/SCRO \
  -v "$PWD/../models":/workspace/models \
  -v "$PWD/../cache":/workspace/cache \
  -w /workspace/SCRO scro:latest
```

The model weights must be under `../models/Qwen2.5-7B/`. Each method reads its Qwen configuration from `hparams/<METHOD>/Qwen-7B.json`; SCRO uses `hparams/SCRO/Qwen-7B.json`. The first run may need to prepare or download covariance statistics, so allow sufficient disk space and time.

## Run the editing methods

Set `N=10000` for a full-size run. Replace GPU `0` with a free physical GPU ID.

```bash
N=10000
DATASET=hetionet_mix
MODEL=Qwen2.5-7B
GPU=0

# Each baseline runs editing, saves the edited model, then evaluates it.
for METHOD in FT MEMIT EMMET PMET ALPHAEDIT MEMIT_MERGE EAMET; do
  bash baselines.sh "$METHOD" "$MODEL" "$DATASET" "$N" edit_eval 1 0 "$GPU"
done

# SCRO: GPU_DEVICES, micro-batch size, cache flag, and squared Z operator-norm bound.
bash scro.sh "$MODEL" "$DATASET" "$N" edit_eval "$GPU" 2 1 25 off
```

The baseline arguments `1 0 "$GPU"` mean one machine, machine index zero, and the selected GPU. `scro.sh` accepts comma-separated GPUs (for example `0,1`) for single-node distributed execution. Do not run multiple jobs on the same GPUs simultaneously unless that is intentional.

## Results

Baseline outputs are under `results/Qwen2.5-7B/hetionet_mix/<METHOD>/baseline/`; SCRO outputs are under `results/Qwen2.5-7B/hetionet_mix/SCRO/scro_hparams<hash>_zop_h25_micro_batch2_ds<N>/`. The SCRO hash is derived from the current Qwen hyperparameter file, so its value may change when that file changes.

For either method, the aggregate evaluation is `eval/ds_<N>_prefix_0/all-metrics-result.json` inside its run directory. The edited model is saved in `edited_model/ds<N>/`. 

Baseline logs are under `logs/Qwen2.5-7B/hetionet_mix/<METHOD>/baseline/ds_<N>/`; SCRO logs are under `logs/Qwen2.5-7B/hetionet_mix/SCRO/<matching-run-name>/edit_eval.log`. Keep the configuration JSON, logs, and result directory together when reporting an experiment.

## Acknowledgements

This repository is built upon the official [EAMET](https://github.com/ybdai7/EAMET-massive-editing) codebase. We thank the EAMET authors for making their code publicly available.
