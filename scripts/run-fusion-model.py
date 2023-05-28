import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import wandb
from datasets import load_from_disk, DatasetDict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from speechemotionrecognition import utils
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, TrainingArguments, Trainer, PretrainedConfig, PreTrainedModel, \
    DataCollatorWithPadding, PreTrainedTokenizerBase
from transformers.utils import ModelOutput

# configuration
model_names = ("openai/whisper-tiny", "facebook/wav2vec2-base-960h")
hidden_dim = 512
num_heads = 1

cache_dir = os.path.join(os.getcwd(), "cache")
output_dir = os.path.join(os.getcwd(), "runs")

batch_size = 10
epochs = 5
learning_rate = 5e-5
debug_size = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_cv_groups = 4  # leave 3 groups out -> only 4 models are trained bc there are 10 speakers in total

metrics = {
    "unweighted_accuracy": accuracy_score,
    "weighted_accuracy": balanced_accuracy_score,
    "micro_f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"),
    "macro_f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
    "confusion_matrix": confusion_matrix
}
label_list = ["neu", "hap", "ang", "sad", "exc"]  # exc & hap are merged together
label2id = {label: i for i, label in enumerate(label_list)},
id2label = {i: label for i, label in enumerate(label_list)}

# wandb
api = wandb.Api()
os.environ["WANDB_PROJECT"] = "representation-fusion"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def _clean_model_name(model_name):
    return model_name.split('/')[-1]


# initialize models & processors
models = {}
processors = {}
for name in model_names:
    if name in models:
        continue

    models[name] = AutoModel.from_pretrained(name).to(device)
    if 'whisper' in name:
        models[name] = models[name].encoder
    try:
        processors[name] = AutoProcessor.from_pretrained(name)
    except OSError as e:
        print("Catched: ", e)
        print("Loading wav2vec2 processor")
        processors[name] = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

# load dataset
artifact = api.artifact("tsinghua-ser/iemocap/raw:v3")
raw_dataset_dir = artifact.download()
raw_dataset = load_from_disk(raw_dataset_dir)


# /!\ uncomment the following line for production
# n = int(debug_size * len(raw_dataset))
# raw_dataset = raw_dataset.select(torch.randint(low=0, high=len(raw_dataset), size=(n,)))  # for debug only


def _process(batch):
    for (processor_name, processor), (model_name, model) in zip(processors.items(), models.items()):
        model_name_clean = model_name.split('/')[-1]

        target_sampling_rate = 16_000
        if hasattr(processor, "feature_encoder"):
            target_sampling_rate = processor.feature_encoder.sampling_rate

        inputs = processor(batch["audio"]["array"], sampling_rate=target_sampling_rate, return_tensors="pt")
        with torch.no_grad():
            if hasattr(inputs, "input_values"):
                input_values = inputs.input_values
            elif hasattr(inputs, "input_features"):
                input_values = inputs.input_features
            else:
                raise NotImplementedError("found none of: ['input_values', 'input_features']")
            input_values = input_values.to(device)
            # `input_values` in batch is replaced with the name of the model
            batch[model_name_clean] = model(input_values).last_hidden_state
    return batch


# get dataset representations from all models
helper_name = "X".join([_clean_model_name(m) for m in models])
description = "Processing IEMOCAP dataset with: " + \
              ",".join([f"{p}/{_clean_model_name(m)}" for p, m in zip(processors.keys(), models.keys())])

dataset = raw_dataset.map(
    function=_process,
    desc=description,
    remove_columns=["audio"],
    cache_file_name=os.path.join(cache_dir, f"{helper_name}/iemocap.arrow")
)

dataset = dataset.rename_columns({_clean_model_name(k): f"rep{i}" for i, k in enumerate(models.keys(), 1)})


# cross-attention model

@dataclass
class FusionModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class FusionConfig(PretrainedConfig):
    def __init__(self, embed_dim=768, hidden_dim=512, num_classes=4, num_heads=1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_classes = num_classes


class FusionModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.attention = nn.MultiheadAttention(embed_dim=config.embed_dim, num_heads=config.num_heads)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=config.embed_dim, out_features=config.hidden_dim),
            nn.Tanh(),
            nn.Linear(in_features=config.hidden_dim, out_features=config.num_classes),
        )
        self.init_weights()

    def forward(
        self,
        input_values: Optional[Tuple[torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,  # labels are needed for loss computation
    ) -> Union[Tuple, FusionModelOutput]:
        assert isinstance(input_values, tuple), "input_values must be a list of Tensors"
        assert len(input_values) == 2, "input_values must have 2 entries: rep1 and rep2"
        rep1, rep2 = input_values

        attended, attn_weights = self.attention(rep2, rep1, rep1)  # cross-attention
        attended = self.norm(attended + rep2)  # Add and Norm

        # Max pooling
        attended = F.adaptive_max_pool2d(attended, output_size=(1, attended.size(-1)))
        attended = attended.squeeze(dim=-2)

        # Classifier
        logits = self.classifier(attended)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))

        return FusionModelOutput(
            loss=loss,
            last_hidden_state=logits,
            hidden_states=None,
            attentions=None,
        )


# data collator
class FusionDataCollator(DataCollatorWithPadding):
    def __init__(
        self,
        fusion_strategy,
        tokenizer: PreTrainedTokenizerBase = None,
        padding: Union[bool, str, transformers.utils.generic.PaddingStrategy] = True,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: str = 'pt'
    ):
        super().__init__(tokenizer, padding, max_length, pad_to_multiple_of, return_tensors)
        self.fusion_strategy = fusion_strategy

    def __call__(self, features):
        labels = torch.tensor([feature['label'] for feature in features], dtype=torch.long, device=device)
        rep1s = [torch.tensor(feature['rep1'], device=device).squeeze() for feature in features]
        rep2s = [torch.tensor(feature['rep2'], device=device).squeeze() for feature in features]

        # Pad sequences independently
        padded_rep1s = pad_sequence(rep1s, batch_first=True)  # [n_samples, time, embed_dim]
        padded_rep2s = pad_sequence(rep2s, batch_first=True)  # [n_samples, time, embed_dim]

        # Apply the fusion strategy
        # fix time dimensions of rep1 & rep2
        if self.fusion_strategy == 'padding':
            pass

        elif self.fusion_strategy == 'max_pooling':
            _pool = lambda x, size: F.adaptive_max_pool1d(x.permute(0, 2, 1), output_size=size)
            time_dim1 = padded_rep1s.size(1)
            time_dim2 = padded_rep2s.size(1)

            if time_dim1 == time_dim2:
                rep1s = padded_rep1s
                rep2s = padded_rep2s
            elif time_dim1 > time_dim2:
                pooled_rep1 = _pool(padded_rep1s, time_dim2)
                pooled_rep1 = pooled_rep1.permute(0, 2, 1)
                rep1s = pooled_rep1
                rep2s = padded_rep2s
            elif time_dim2 > time_dim1:
                pooled_rep2 = _pool(padded_rep2s, time_dim1)
                pooled_rep2 = pooled_rep2.permute(0, 2, 1)
                rep1s = padded_rep1s
                rep2s = pooled_rep2

        else:
            raise ValueError("Invalid fusion strategy")

        # fix embedding dimensions of rep1 & rep2
        embed_dim1 = rep1s.size(2)
        embed_dim2 = rep2s.size(2)
        if embed_dim1 > embed_dim2:
            rep2s = torch.tile(rep2s, (1, 1, embed_dim1 // embed_dim2))
        elif embed_dim2 > embed_dim1:
            rep1s = torch.tile(rep1s, (1, 1, embed_dim2 // embed_dim1))
        assert rep1s.size(2) == rep2s.size(2), f"invalid sizes: {rep1s.size(2)} vs {rep2s.size(2)}"

        batch = {
            'labels': labels,
            'input_values': (rep1s, rep2s)  # tuple
        }
        return batch


# leave-one-group(of speakers)-out cross-validation
splits = utils.get_cv_splits(dataset, n_cv_groups=n_cv_groups)

# get embed dim (using actual vectors)
train_idx, test_idx = next(splits)
ds = DatasetDict({
    "train": dataset.select(train_idx),
    "test": dataset.select(test_idx)
})
sample = ds['train'][0]
rep1, rep2 = torch.Tensor(sample['rep1']).squeeze(), torch.Tensor(sample['rep2']).squeeze()
print("representation sizes: ", rep1.shape, rep2.shape)

data_collator = FusionDataCollator(fusion_strategy="max_pooling")
examples = [ds['train'][0], ds['train'][1], ds['train'][2]]
collated = data_collator(examples)
print("collated sizes: ", collated['input_values'][0].shape)
embed_dim = collated['input_values'][0].size(-1)

fusion_config = FusionConfig(
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    num_heads=num_heads,
    num_classes=len(label_list)
)

for train_index, test_index in tqdm(splits):
    args = {
        "project": os.environ["WANDB_PROJECT"],
        "tags": ["fusion", *model_names],
        "group": "X".join([_clean_model_name(m) for m in model_names])
    }
    with wandb.init(**args) as run:
        ds = DatasetDict({
            "train": dataset.select(train_index),
            "test": dataset.select(test_index)
        })

        _get_speakers = lambda s: np.unique(ds[s]['speaker'])
        print("train speakers: ", _get_speakers("train"))
        print("test speakers: ", _get_speakers("test"))

        # model
        fusion_model = FusionModel(fusion_config)

        # trainer
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, run.id),
            dataloader_pin_memory=False,
            # fix: RuntimeError: cannot pin 'torch.cuda.LongTensor' only dense CPU tensors can be pinned
            remove_unused_columns=False,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            evaluation_strategy="steps",  # should enable do_eval
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),  # whether to use fp16 16-bit (mixed) precision training
            # instead of 32-bit training
            save_steps=50,
            eval_steps=10,
            logging_steps=50,
            report_to=["wandb"],
            half_precision_backend="auto",  # should be 'cuda_amp' half precision backend
        )

        data_collator = FusionDataCollator(fusion_strategy="max_pooling")
        trainer = Trainer(
            model=fusion_model,
            args=training_args,
            data_collator=data_collator,
            compute_metrics=utils.get_compute_metrics(metrics),
            train_dataset=ds["train"],
            eval_dataset=ds["test"],
        )

        # train
        trainer.train()
