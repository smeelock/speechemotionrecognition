import os
import re
from pathlib import Path

import torch
from datasets import Dataset, Value, Features, ClassLabel, DatasetInfo, Audio

from .constants import DEFAULT_IEMOCAP_LABEL_LIST, DEFAULT_TARGET_SAMPLING_RATE, DEFAULT_CACHE_DIR


# inspired by https://github.com/pytorch/audio/blob/main/torchaudio/datasets/iemocap.py
def _get_dict(path):
    assert os.path.isdir(path), f"Dataset not found at: {path}"

    sessions = (1, 2, 3, 4, 5)
    tmp = {}  # will contain all wav file stems
    labels = []
    paths = []
    speakers = []

    for session in sessions:
        session_name = f"Session{session}"
        session_dir = path / session_name

        # get all wav paths in tmp
        wav_dir = session_dir / "sentences" / "wav"
        wav_paths = sorted(str(p) for p in wav_dir.glob("*/*.wav"))
        for wav_path in wav_paths:
            wav_stem = str(Path(wav_path).stem)
            tmp[wav_stem] = wav_path

        # add label and speaker information
        label_dir = session_dir / "dialog" / "EmoEvaluation"
        label_paths = label_dir.glob("*.txt")

        for label_path in label_paths:
            with open(label_path, "r") as f:
                for line in f:
                    if not line.startswith("["):
                        continue
                    line = re.split("[\t\n]", line)
                    wav_stem = line[1]
                    label = line[2]
                    if wav_stem not in tmp.keys():  # only use data that has a wav file
                        continue
                    if label not in DEFAULT_IEMOCAP_LABEL_LIST:  # only use labels that are in the default list
                        continue

                    paths.append(tmp[wav_stem])
                    labels.append(label)
                    speakers.append(wav_stem.split("_")[0])

    return {"audio": paths, "path": paths, "label": labels, "speaker": speakers}


def get_iemocap(root, processor, model, cache_dir=DEFAULT_CACHE_DIR):
    """Get the IEMOCAP dataset.
    This functions wraps the following steps:
    1. Load the raw dataset
    2. Process the dataset
    3. Get the representations
    """
    raw_dataset = load_iemocap(root)
    dataset = process_dataset(raw_dataset, processor, model)
    dataset = get_representations(dataset, model, cache_dir)
    return dataset


def load_iemocap(root, cache_dir=DEFAULT_CACHE_DIR, filename="iemocap_raw.arrow"):
    """Get the IEMOCAP dataset."""
    root = Path(root)
    features = Features({
        "audio": Audio(sampling_rate=DEFAULT_TARGET_SAMPLING_RATE),
        "path": Value(dtype='string', id=None),
        "label": ClassLabel(num_classes=len(DEFAULT_IEMOCAP_LABEL_LIST), names=DEFAULT_IEMOCAP_LABEL_LIST,
                            names_file=None, id=None),
        "speaker": Value(dtype='string', id=None)
    })
    info = DatasetInfo(description="A 🤗 datasets loader for the IEMOCAP dataset",
                       homepage="https://sail.usc.edu/iemocap/", features=features)
    dataset = Dataset.from_dict(_get_dict(root), info=info)

    def _merge_emotions(example):
        if example["label"] == "exc":
            example["label"] = "hap"
        return example

    description = "Merging emotions `exc` & `hap`"
    if not cache_dir:
        return dataset.map(_merge_emotions, desc=description)
    return dataset.map(_merge_emotions, desc=description, cache_file_name=os.path.join(cache_dir, filename))


def process_dataset(dataset, processor, cache_dir=DEFAULT_CACHE_DIR, filename="iemocap_processed.arrow"):
    """Process a dataset with a given processor."""
    target_sampling_rate = DEFAULT_TARGET_SAMPLING_RATE
    if hasattr(processor, "feature_encoder"):
        target_sampling_rate = processor.feature_encoder.sampling_rate

    def _process(batch):
        inputs = processor(batch["audio"]["array"], sampling_rate=target_sampling_rate, return_tensors="pt")
        if "input_features" in inputs:  # whisper
            batch["input_features"] = inputs.input_features[0]
        elif "input_values" in inputs:  # wav2vec2
            batch["input_features"] = inputs.input_values
        return batch

    description = f"Processing IEMOCAP dataset with {type(processor).__name__}"
    args = {"function": _process, "desc": description, "remove_columns": ["audio"]}
    if cache_dir:
        args["cache_file_name"] = os.path.join(cache_dir, filename)
    return dataset.map(**args)


def get_representations(dataset, model, cache_dir=DEFAULT_CACHE_DIR, filename="iemocap_representations.arrow"):
    """Get speech representations of a dataset using a given pretrained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def _get_speech_representations(batch):
        inputs = torch.Tensor(batch["input_features"]).to(device)
        batch["representations"] = model(inputs).last_hidden_state
        return batch

    description = f"Getting IEMOCAP dataset speech representations using {type(model).__name__}"
    args = {"function": _get_speech_representations, "desc": description, "remove_columns": ["input_features"]}
    if cache_dir:
        args["cache_file_name"] = os.path.join(cache_dir, filename)
    return dataset.map(**args)
