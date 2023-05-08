import os

import wandb
from datasets import load_dataset
from speechemotionrecognition.dataset_helpers import process_dataset
from transformers import AutoProcessor

args = {
    "project": "iemocap",
    "job_type": "process",
}
cache_dir = "../cache/"
filename_template = "iemocap_{}_processed.arrow"
model_names = ["openai/whisper-base", "facebook/wav2vec2-base-960h"]

for model_name in model_names:
    processor = AutoProcessor.from_pretrained(model_name)
    processor_name = type(processor).__name__.lower()
    filename = filename_template.format(processor_name)

    with wandb.init(**args) as run:
        raw_dataset_dir = run.use_artifact("iemocap/raw:latest").download()
        artifact = wandb.Artifact(f"processed/{processor_name}", type="dataset")

        dataset = load_dataset(raw_dataset_dir)
        dataset = process_dataset(dataset, processor, cache_dir=cache_dir, filename=filename)

        artifact.add_file(local_path=os.path.join(cache_dir, filename), name=filename)
        run.log_artifact(artifact)

