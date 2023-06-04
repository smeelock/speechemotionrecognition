import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import load_from_disk, DatasetDict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.svm import SVC
from speechemotionrecognition import utils
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

# configuration
model_names = ("openai/whisper-tiny", "facebook/wav2vec2-base-960h")

cache_dir = os.path.join(os.getcwd(), "cache")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
debug_size = 0.001
n_cv_groups = 3  # leave 3 groups out -> only 4 models are trained bc there are 10 speakers in total

metrics = {
    "unweighted_accuracy": accuracy_score,
    "weighted_accuracy": balanced_accuracy_score,
    "micro_f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="micro"),
    "macro_f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro")
}
label_list = ["neu", "hap", "ang", "sad", "exc"]  # exc & hap are merged together
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

# wandb
api = wandb.Api()
os.environ["WANDB_PROJECT"] = "representation-fusion"
os.environ["WANDB_WATCH"] = "all"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


# utils
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


def collate(features):
    labels = torch.tensor([feature['label'] for feature in features], dtype=torch.long)
    rep1s = [torch.tensor(feature['rep1']).squeeze() for feature in features]
    rep2s = [torch.tensor(feature['rep2']).squeeze() for feature in features]

    # Pad sequences independently
    padded_rep1s = pad_sequence(rep1s, batch_first=True)  # [n_samples, time, embed_dim]
    padded_rep2s = pad_sequence(rep2s, batch_first=True)  # [n_samples, time, embed_dim]

    # Apply the fusion strategy
    _pool = lambda x, size: F.adaptive_max_pool1d(x.permute(0, 2, 1), output_size=size)
    time_dim1 = padded_rep1s.size(1)
    time_dim2 = padded_rep2s.size(1)

    if time_dim1 == time_dim2:
        collated = torch.cat([padded_rep1s, padded_rep2s], dim=-1)
    elif time_dim1 > time_dim2:
        pooled_rep1 = _pool(padded_rep1s, time_dim2)
        pooled_rep1 = pooled_rep1.permute(0, 2, 1)
        collated = torch.cat([pooled_rep1, padded_rep2s], dim=-1)
    elif time_dim2 > time_dim1:
        pooled_rep2 = _pool(padded_rep2s, time_dim1)
        pooled_rep2 = pooled_rep2.permute(0, 2, 1)
        collated = torch.cat([padded_rep1s, pooled_rep2], dim=-1)

    return collated, labels


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

# leave-one-speaker-out cross-validation
splits = utils.get_cv_splits(dataset, n_cv_groups=n_cv_groups)

for train_index, test_index in tqdm(splits):
    args = {
        "project": os.environ["WANDB_PROJECT"],
        "tags": ["svm", *model_names],
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

        X_train, y_train = collate(ds["train"])
        X_test, y_test = collate(ds["test"])

        # model
        svm = SVC(kernel='linear', C=1.0)
        svm.fit(X_train, y_train)

        for m, metric in utils.get_compute_metrics(metrics):
            wandb.log({m: metric.compute(svm.predict(X_test), y_test)})
