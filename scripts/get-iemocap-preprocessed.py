import os

import wandb
from datasets import load_from_disk
from speechemotionrecognition.constants import DEFAULT_MAX_SHARD_SIZE
from speechemotionrecognition.dataset_helpers import process_dataset
from transformers import AutoProcessor

args = {
    "project": "iemocap",
    "job_type": "preprocess",
}
package_dir = os.getcwd()
save_dir_template = package_dir + "/data/processed/iemocap/{}"
model_names = ["openai/whisper-base", "facebook/wav2vec2-base-960h"]

for model_name in model_names:
    processor = AutoProcessor.from_pretrained(model_name)
    processor_name = type(processor).__name__.lower()
    savepath = save_dir_template.format(processor_name)

    with wandb.init(**args) as run:
        input_artifact_name = f"iemocap/raw:latest"
        raw_dataset_dir = run.use_artifact(input_artifact_name).download()

        new_artifact_name = f"processed-{processor_name}"
        artifact = wandb.Artifact(new_artifact_name, type="dataset")

        dataset = load_from_disk(raw_dataset_dir)
        dataset = process_dataset(dataset, processor)
        dataset.save_to_disk(savepath, max_shard_size=DEFAULT_MAX_SHARD_SIZE)

        artifact.add_dir(local_path=savepath)
        run.log_artifact(artifact)
