import os

import click
import torch
from datasets import DatasetDict
from sklearn.model_selection import LeaveOneGroupOut
from transformers import AutoConfig, TrainingArguments, Trainer, AutoProcessor

from .constants import DEFAULT_WANDB_WATCH, DEFAULT_WANDB_LOG_MODEL, DEFAULT_WHISPER_MODEL_NAME, DEFAULT_OUTPUT_DIR, \
    DEFAULT_TEST_SPLIT_SIZE, DEFAULT_SEED, DEFAULT_IEMOCAP_LABEL_LIST, DEFAULT_IEMOCAP_LABEL2ID, \
    DEFAULT_IEMOCAP_ID2LABEL, DEFAULT_WANDB_PROJECT, DEFAULT_IEMOCAP_DIR, \
    DEFAULT_METRICS, DEFAULT_CACHE_DIR
from .dataset_helpers import load_iemocap, process_dataset, get_representations
from .models import WhisperEncoderForSpeechClassification
from .trainers import DataCollatorCTCWithPadding
from .utils import get_compute_metrics


@click.command()
@click.option("--batch-size", default=4, type=int, help="Batch size")
@click.option("--cache-dir", default=DEFAULT_CACHE_DIR, type=str, help="Cache directory")
@click.option("--data-dir", default=DEFAULT_IEMOCAP_DIR, type=str, help="Data directory")
@click.option("--dataset", default="iemocap", type=click.Choice(["iemocap"], case_sensitive=False), help="Dataset name")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--epochs", default=2, type=int, help="Number of epochs")
@click.option("--learning-rate", default=5e-5, type=float, help="Learning rate")
@click.option("--model-name-or-path", default=DEFAULT_WHISPER_MODEL_NAME, type=str, help="Model name or path")
@click.option("--num-encoder-layers", default=4, type=int, help="Number of encoder layers")
@click.option("--output-dir", default=DEFAULT_OUTPUT_DIR, type=str, help="Output directory")
@click.option("--seed", default=DEFAULT_SEED, type=int, help="Seed")
@click.option("--test-split-size", default=DEFAULT_TEST_SPLIT_SIZE, type=float, help="Test split size")
@click.option("--wandb-disabled", is_flag=True, help="Disable wandb")
@click.option("--wandb-log-model", default=DEFAULT_WANDB_LOG_MODEL, type=str, help="Wandb log model")
@click.option("--wandb-project", default=DEFAULT_WANDB_PROJECT, type=str, help="Wandb project")
@click.option("--wandb-watch", default=DEFAULT_WANDB_WATCH, type=str, help="Wandb watch")
def main(
    batch_size,
    cache_dir,
    data_dir,
    dataset,
    debug,
    epochs,
    learning_rate,
    model_name_or_path,
    num_encoder_layers,
    output_dir,
    seed,
    test_split_size,
    wandb_disabled,
    wandb_log_model,
    wandb_project,
    wandb_watch,
):
    # configuration
    metrics = DEFAULT_METRICS
    if dataset == "iemocap":
        label_list = DEFAULT_IEMOCAP_LABEL_LIST
        label2id = DEFAULT_IEMOCAP_LABEL2ID
        id2label = DEFAULT_IEMOCAP_ID2LABEL
    else:
        raise NotImplementedError

    if not wandb_disabled:
        if debug:
            wandb_project = "test"
        os.environ["WANDB_PROJECT"] = wandb_project
        os.environ["WANDB_WATCH"] = wandb_watch
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(label_list),
        label2id=label2id,
        id2label=id2label
    )
    setattr(config, "num_encoder_layers", num_encoder_layers)

    # model
    model = WhisperEncoderForSpeechClassification.from_pretrained(
        model_name_or_path,
        config=config
    )
    model.freeze_encoder()

    # dataset
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    raw_dataset = load_iemocap(data_dir)
    dataset = process_dataset(raw_dataset, processor, model)

    # leave-one-speaker-out cross-validation
    logo = LeaveOneGroupOut()
    splits = logo.split(
        X=torch.zeros(dataset["train"].num_rows),
        y=dataset["train"]["label"],
        groups=dataset["train"]["speaker"]
    )
    for train_index, test_index in splits:
        ds = DatasetDict({
            "train": dataset["train"].select(train_index),
            "test": dataset["train"].select(test_index)
        })

        # trainer
        training_args = TrainingArguments(
            output_dir=output_dir,
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
            report_to=[] if wandb_disabled else ["wandb"],
            half_precision_backend="auto",  # should be 'cuda_amp' half precision backend
            gradient_checkpointing=True,  # use gradient checkpointing to save memory at the expense
            # of slower backward pass
        )

        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=get_compute_metrics(metrics),
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
            tokenizer=processor.feature_extractor,
        )

        # train
        trainer.train()


if __name__ == "__main__":
    main()
