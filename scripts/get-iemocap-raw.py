import os

import wandb
from speechemotionrecognition.constants import DEFAULT_MAX_SHARD_SIZE
from speechemotionrecognition.dataset_helpers import load_iemocap

args = {
    "project": "iemocap",
    "job_type": "load",
}
filename = "iemocap_raw.arrow"
package_dir = os.getcwd()
save_dir = package_dir + "/data/raw/iemocap"
data_dir = package_dir + "/data/raw/IEMOCAP"

with wandb.init(**args) as run:
    artifact = wandb.Artifact("raw", type="dataset")

    dataset = load_iemocap(data_dir)
    savepath = os.path.join(save_dir, filename)
    dataset.save_to_disk(savepath, max_shard_size=DEFAULT_MAX_SHARD_SIZE)

    artifact.add_dir(local_path=savepath, name=filename)
    run.log_artifact(artifact)
