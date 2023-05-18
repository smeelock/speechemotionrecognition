"""
A script to run baseline model on IEMOCAP dataset.
The baseline model is a simple classifier on top of the speech representations from e.g. wav2vec2, whipser or hubert.
"""

import os

import speechemotionrecognition.utils as utils
import torch
import wandb
from datasets import load_from_disk, DatasetDict
from sklearn.model_selection import LeaveOneGroupOut
from speechemotionrecognition.constants import DEFAULT_IEMOCAP_LABEL_LIST, DEFAULT_IEMOCAP_LABEL2ID, \
    DEFAULT_IEMOCAP_ID2LABEL
from speechemotionrecognition.constants import DEFAULT_METRICS
from speechemotionrecognition.models import SpeechClassificationHead
from tqdm.notebook import tqdm
from transformers import AutoConfig, Trainer, TrainingArguments

# initialization
api = wandb.Api()
os.environ["WANDB_PROJECT"] = "representation-fusion"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

package_dir = os.getcwd()
output_dir = os.path.join(package_dir, "runs/")

# config
model_name = "openai/whisper-base"
label_list = DEFAULT_IEMOCAP_LABEL_LIST

batch_size = 50  # ~1% of iemocap dataset
epochs = 10
learning_rate = 5e-5

config = AutoConfig.from_pretrained(
    model_name,
    num_labels=len(label_list),
    label2id=DEFAULT_IEMOCAP_LABEL2ID,
    id2label=DEFAULT_IEMOCAP_ID2LABEL
)
setattr(config, "num_encoder_layers", 1000)  # set to absurd number to get all layers
setattr(config, "merged_strategy", "max")

# download artifact
artifact_name = "{username}/{project}/{artifact_name}:{version}".format(
    username="tsinghua-ser",
    project="iemocap",
    artifact_name=f"representations-{model_name.split('/')[-1]}",
    version="v1"
)
artifact = api.artifact(artifact_name)
artifact_dir = artifact.download()

# load dataset
dataset = load_from_disk(artifact_dir)
dataset = dataset.rename_column("representations", "input_values")

# leave-one-speaker-out cross-validation
logo = LeaveOneGroupOut()
splits = logo.split(
    X=torch.zeros(len(dataset)),
    y=dataset["label"],
    groups=dataset["speaker"]
)

# train one model for each cross-validation split
for train_index, test_index in tqdm(splits):
    ds = DatasetDict({
        "train": dataset.select(train_index),
        "test": dataset.select(test_index)
    })

    # model
    model = SpeechClassificationHead(config=config)

    # trainer
    training_args = TrainingArguments(
        output_dir=output_dir,
        label_names=label_list,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        evaluation_strategy="steps",  # should enable do_eval
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),  # whether to use fp16 16-bit (mixed) precision training
        # instead of 32-bit training
        save_steps=100,
        eval_steps=10,
        logging_steps=50,
        report_to=["wandb"],
        half_precision_backend="auto",  # should be 'cuda_amp' half precision backend
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=utils.get_compute_metrics(DEFAULT_METRICS),
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
    )

    # train
    trainer.train()
