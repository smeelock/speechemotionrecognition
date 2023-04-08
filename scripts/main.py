import os

import torch
from torch.utils.data import random_split
from torchaudio.datasets import IEMOCAP
from transformers import AutoConfig, WhisperProcessor, TrainingArguments

import utils
from constants import DEFAULT_WANDB_WATCH, DEFAULT_WANDB_LOG_MODEL, DEFAULT_WHISPER_MODEL_NAME, DEFAULT_DATA_DIR, \
    DEFAULT_OUTPUT_DIR, DEFAULT_TEST_SPLIT_SIZE, DEFAULT_SEED, DEFAULT_IEMOCAP_LABEL_LIST, DEFAULT_IEMOCAP_LABEL2ID, \
    DEFAULT_IEMOCAP_ID2LABEL
from datasets import ProcessedIEMOCAP
from models import WhisperEncoderForSpeechClassification
from trainers import DataCollatorCTCWithPadding, CTCTrainer

# ========= Configuration =========
os.environ["WANDB_WATCH"] = DEFAULT_WANDB_WATCH
os.environ["WANDB_LOG_MODEL"] = DEFAULT_WANDB_LOG_MODEL

model_name_or_path = DEFAULT_WHISPER_MODEL_NAME
data_dir = DEFAULT_DATA_DIR
output_dir = DEFAULT_OUTPUT_DIR
test_split_size = DEFAULT_TEST_SPLIT_SIZE
num_encoder_layers = 4
seed = DEFAULT_SEED
label_list = DEFAULT_IEMOCAP_LABEL_LIST
label2id = DEFAULT_IEMOCAP_LABEL2ID
id2label = DEFAULT_IEMOCAP_ID2LABEL

config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=len(label_list),
    label2id=label2id,
    id2label=id2label
)
setattr(config, "num_encoder_layers", num_encoder_layers)

# ========= Model =========
model = WhisperEncoderForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config
)
model.freeze_encoder()

# ========= Dataset =========
processor = WhisperProcessor.from_pretrained(model_name_or_path)
iemocap = IEMOCAP(root=data_dir)  # in function, path = root / "IEMOCAP"
dataset = ProcessedIEMOCAP(data=iemocap, processor=processor)
train_ds, test_ds = random_split(
    dataset,
    [1 - test_split_size, test_split_size],
    generator=torch.Generator().manual_seed(seed)
)

# ========= Trainer =========
training_args = TrainingArguments(
    output_dir=output_dir,
    label_names=label_list,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",  # should enable do_eval
    num_train_epochs=1.0,
    learning_rate=1e-4,
    fp16=torch.cuda.is_available(),  # whether to use fp16 16-bit (mixed) precision training instead of 32-bit training
    save_steps=100,
    eval_steps=10,
    logging_steps=50,
    report_to=["wandb"],
    half_precision_backend="auto",  # should be 'cuda_amp' half precision backend
    gradient_checkpointing=True,  # use gradient checkpointing to save memory at the expense of slower backward pass
)

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=utils.compute_metrics,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=processor.feature_extractor,
)

# ========= Training =========
trainer.train()
