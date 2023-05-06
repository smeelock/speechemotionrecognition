import os
import re
from pathlib import Path

from datasets import Dataset, Value, Features, ClassLabel, DatasetInfo, Audio

from .constants import DEFAULT_IEMOCAP_LABEL_LIST, DEFAULT_TARGET_SAMPLING_RATE


# inspired by https://github.com/pytorch/audio/blob/main/torchaudio/datasets/iemocap.py
def _get_dict(path):
    assert os.path.isdir(path), "Dataset not found."

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


def get_iemocap(root):
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
    dataset = dataset.map(_merge_emotions, desc="Merging emotions: excited + happy")
    return dataset
