import os

import wandb
from datasets import load_from_disk
from speechemotionrecognition.constants import DEFAULT_MAX_SHARD_SIZE
from speechemotionrecognition.dataset_helpers import get_representations
from transformers import AutoProcessor, AutoModel

args = {
    "project": "iemocap",
    "job_type": "represent",
}
package_dir = os.getcwd()
save_dir_template = package_dir + "/data/processed/iemocap/{}"
model_names = ["openai/whisper-base", "facebook/wav2vec2-base-960h"]

for model_name in model_names:
    model = AutoModel.from_pretrained(model_name)
    processor_name = type(AutoProcessor.from_pretrained(model_name)).__name__.lower()
    savepath = save_dir_template.format(processor_name)

    with wandb.init(**args) as run:
        input_artifact_name = f"iemocap/preprocessed-{processor_name}:latest"
        preprocessed_dataset_dir = run.use_artifact(input_artifact_name).download()

        new_artifact_name = f"representations-{model_name.split('/')[-1]}"
        artifact = wandb.Artifact(new_artifact_name, type="dataset")

        dataset = load_from_disk(preprocessed_dataset_dir)
        dataset = get_representations(dataset, model)
        dataset.save_to_disk(savepath, max_shard_size=DEFAULT_MAX_SHARD_SIZE)

        artifact.add_dir(local_path=savepath)
        run.log_artifact(artifact)
    del model, dataset
