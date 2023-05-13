import os

import wandb
from speechemotionrecognition.constants import DEFAULT_MAX_SHARD_SIZE
from speechemotionrecognition.dataset_helpers import load_iemocap

args = {
    "project": "iemocap",
    "job_type": "load",
}
package_dir = os.getcwd()
save_dir = package_dir + "/data/raw/iemocap"
data_dir = package_dir + "/data/raw/IEMOCAP"

with wandb.init(**args) as run:
    artifact = wandb.Artifact("raw", type="dataset")

    dataset = load_iemocap(data_dir)
    dataset.save_to_disk(save_dir, max_shard_size=DEFAULT_MAX_SHARD_SIZE)

    artifact.add_dir(local_path=save_dir)
    run.log_artifact(artifact)
