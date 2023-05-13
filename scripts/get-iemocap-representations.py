import os

import wandb
from datasets import Dataset
from speechemotionrecognition.constants import DEFAULT_MAX_SHARD_SIZE
from speechemotionrecognition.dataset_helpers import get_representations
from transformers import AutoProcessor, AutoModel

args = {
    "project": "iemocap",
    "job_type": "represent",
}
package_dir = os.getcwd()
data_dir = package_dir + "/data/raw/IEMOCAP"
save_dir = package_dir + "/data/processed/iemocap"
filename_template = "iemocap_{}_representations.arrow"
model_names = ["openai/whisper-base", "facebook/wav2vec2-base-960h"]

for model_name in model_names:
    model = AutoModel.from_pretrained(model_name)
    processor_name = type(AutoProcessor.from_pretrained(model_name)).__name__.lower()
    filename = filename_template.format(processor_name)

    with wandb.init(**args) as run:
        processed_dataset = run.use_artifact(f"iemocap/processed-{processor_name}:latest").download()
        artifact = wandb.Artifact(f"representations-{model_name.split('/')[-1]}", type="dataset")

        dataset = Dataset.from_file(processed_dataset + "/iemocap_processed.arrow")
        dataset = get_representations(dataset, model)
        savepath = os.path.join(save_dir, filename)
        dataset.save_to_disk(savepath, max_shard_size=DEFAULT_MAX_SHARD_SIZE)

        artifact.add_dir(local_path=savepath, name="iemocap_representations.arrow")
        run.log_artifact(artifact)
    del model, dataset
