from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

with (PROJECT_ROOT / "globals.yml").open("r") as stream:
    data = yaml.safe_load(stream)


def project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


(RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR, KV_DIR, MOTIVATION_DIR, P_DIR) = (
    project_path(z)
    for z in [
        data["RESULTS_DIR"],
        data["DATA_DIR"],
        data["STATS_DIR"],
        data["HPARAMS_DIR"],
        data["KV_DIR"],
        data["MOTIVATION_DIR"],
        data["P_DIR"],
    ]
)

REMOTE_ROOT_URL = data["REMOTE_ROOT_URL"]
