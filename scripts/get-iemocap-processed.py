import os

import wandb
from datasets import Dataset
from speechemotionrecognition.dataset_helpers import process_dataset
from transformers import AutoProcessor

args = {
    "project": "iemocap",
    "job_type": "process",
}
package_dir = os.getcwd()
cache_dir = package_dir + "/cache/"
data_dir = package_dir + "/data/raw/IEMOCAP"
filename_template = "iemocap_{}_processed.arrow"
model_names = ["openai/whisper-base", "facebook/wav2vec2-base-960h"]

for model_name in model_names:
    processor = AutoProcessor.from_pretrained(model_name)
    processor_name = type(processor).__name__.lower()
    filename = filename_template.format(processor_name)

    with wandb.init(**args) as run:
        raw_dataset_dir = run.use_artifact("iemocap/raw:latest").download()
        artifact = wandb.Artifact(f"processed-{processor_name}", type="dataset")

        dataset = Dataset.from_file(raw_dataset_dir + "/iemocap_raw.arrow")
        dataset = process_dataset(dataset, processor, cache_dir=cache_dir, filename=filename)

        artifact.add_file(local_path=os.path.join(cache_dir, filename), name="iemocap_processed.arrow")
        run.log_artifact(artifact)
