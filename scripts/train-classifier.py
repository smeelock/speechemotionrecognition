"""
A script to train a Speech Emotion Recognition classifier on the IEMOCAP dataset using
OpenAI's whisper model as a feature extractor.

Original file is located at
    https://colab.research.google.com/drive/1-vw1bNt-e8DdzAk74tEqbIkBiL2hfaOf
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List, Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import CrossEntropyLoss
from torch.utils.data import Subset, random_split
from torchaudio.datasets import IEMOCAP  # https://pytorch.org/audio/master/generated/torchaudio.datasets.IEMOCAP.html
from transformers import WhisperPreTrainedModel, AutoConfig, WhisperProcessor, WhisperModel, TrainingArguments, \
    EvalPrediction, Trainer, is_apex_available
from transformers.utils import ModelOutput

if is_apex_available():
    from apex import amp

# ========= Configuration =========
# wandb
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data/raw")
output_dir = os.path.join(base_dir, "runs/")

# training
debug_size = 1  # 0.1 = 10% of the dataset
test_split_size = 0.2

feature_to_idx = {key: i for i, key in enumerate(["wav", "sampling_rate", "filename", "label", "speaker"])}
label_list = ["neu", "hap", "ang", "sad", "exc", "fru"]
num_labels = len(label_list)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# evaluation
metrics = {
    "unweighted_accuracy": accuracy_score,
    "weighted_accuracy": balanced_accuracy_score,
    "micro_f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"),
    "macro_f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")
}

# ========= Model =========
model_name_or_path = "openai/whisper-base"
num_encoder_layers = 4
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
)
setattr(config, "num_encoder_layers", num_encoder_layers)


@dataclass
class ClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class ClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class WhisperForSpeechClassification(WhisperPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.encoder = WhisperModel(config).encoder

        # only keep first n encoding layers
        self.encoder.layers = self.encoder.layers[:config.num_encoder_layers]
        self.classifier = ClassificationHead(config)

        self.init_weights()

    def freeze_encoder(self):
        self.encoder._freeze_parameters()

    def forward(
        self,
        input_features,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = torch.max(hidden_states, dim=1)[0]
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return ClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


model = WhisperForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config
)
model.freeze_encoder()

# ========= IEMOCAP =========
processor = WhisperProcessor.from_pretrained(model_name_or_path)
target_sampling_rate = processor.feature_extractor.sampling_rate


class CustomIEMOCAP(torch.utils.data.Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __getitem__(self, index):
        wav, _, _, label, _ = self.data[index]
        inputs = self.processor(wav.squeeze(), sampling_rate=target_sampling_rate)
        inputs["labels"] = label2id[label]
        return inputs

    def __len__(self):
        return len(self.data)


iemocap = IEMOCAP(root=data_dir)  # in function, path = root / "IEMOCAP"

# ========= Dataset =========
dataset = CustomIEMOCAP(data=iemocap, processor=processor)
train_ds, test_ds = random_split(dataset, [1 - test_split_size, test_split_size],
                                 generator=torch.Generator().manual_seed(42))

# ========= Training =========
# training parameters
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


@dataclass
class DataCollatorCTCWithPadding:
    processor: WhisperProcessor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"][0]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        d_type = torch.long if isinstance(label_features[0], int) else torch.float

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)
        return batch


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {k: metric(p.label_ids, preds) for k, metric in metrics.items()}


class CTCTrainer(Trainer):
    def training_step(self, model, inputs) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if hasattr(self, 'use_amp') and self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if hasattr(self, 'use_amp') and self.use_amp:
            self.scaler.scale(loss).backward()
        elif hasattr(self, 'use_apex') and self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif hasattr(self, 'deepspeed') and self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys) -> torch.Tensor:
        model.eval()
        inputs = self._prepare_inputs(inputs)

        labels = inputs.get("labels")

        with autocast():
            outputs = model(**inputs)
            logits = outputs.get("logits")
            loss = outputs.get("loss")

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if hasattr(self, 'use_amp') and self.use_amp:
            self.scaler.scale(loss).backward()
        elif hasattr(self, 'use_apex') and self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif hasattr(self, 'deepspeed') and self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        with torch.no_grad():
            torch.cuda.empty_cache()

        if prediction_loss_only:
            return loss.detach()
        return (loss.detach(), logits.detach(), labels.detach())


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=processor.feature_extractor,
)

trainer.train()
