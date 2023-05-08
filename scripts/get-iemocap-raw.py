import os

import wandb
from speechemotionrecognition.dataset_helpers import load_iemocap

args = {
    "project": "iemocap",
    "job_type": "load",
}
filename = "iemocap_raw.arrow"
package_dir = os.getcwd()
cache_dir = package_dir + "/cache/"
data_dir = package_dir + "/data/raw/IEMOCAP"

with wandb.init(**args) as run:
    artifact = wandb.Artifact("raw", type="dataset")

    dataset = load_iemocap(data_dir, cache_dir=cache_dir, filename=filename)
    artifact.add_file(local_path=os.path.join(cache_dir, filename), name=filename)
    run.log_artifact(artifact)
