import os
import re
from pathlib import Path

from datasets import Dataset, Value, Features

from constants import DEFAULT_IEMOCAP_LABEL_LIST


# inspired by https://github.com/pytorch/audio/blob/main/torchaudio/datasets/iemocap.py
def _get_wavs_paths(data_dir):
    wav_dir = data_dir / "sentences" / "wav"
    wav_paths = sorted(str(p) for p in wav_dir.glob("*/*.wav"))
    relative_paths = []
    for wav_path in wav_paths:
        start = wav_path.find("Session")
        wav_path = wav_path[start:]
        relative_paths.append(wav_path)
    return relative_paths


def _get_metadata(root):
    root = Path(root)
    path = root / "IEMOCAP"
    assert os.path.isdir(path), "Dataset not found."

    sessions = (1, 2, 3, 4, 5)
    tmp = []  # will contain all wav file stems
    metadata = {}  # will contain all metadata

    for session in sessions:
        session_name = f"Session{session}"
        session_dir = path / session_name

        # get wav paths
        wav_paths = _get_wavs_paths(session_dir)
        for wav_path in wav_paths:
            wav_stem = str(Path(wav_path).stem)
            tmp.append(wav_stem)

        # add label and speaker information
        label_dir = session_dir / "dialog" / "EmoEvaluation"
        query = "*.txt"
        label_paths = label_dir.glob(query)

        for label_path in label_paths:
            with open(label_path, "r") as f:
                for line in f:
                    if not line.startswith("["):
                        continue
                    line = re.split("[\t\n]", line)
                    wav_stem = line[1]
                    label = line[2]
                    if wav_stem not in tmp:  # only use data that has a wav file
                        continue
                    if label not in DEFAULT_IEMOCAP_LABEL_LIST:  # only use labels that are in the default list
                        continue
                    metadata[wav_stem] = {}
                    metadata[wav_stem]["label"] = label
                    metadata[wav_stem]["speaker"] = wav_stem.split("_")[0]

        for wav_path in wav_paths:
            wav_stem = str(Path(wav_path).stem)
            if wav_stem in metadata:
                metadata[wav_stem]["file_name"] = wav_path

    return metadata


def get_iemocap(root):
    metadata = _get_metadata(root)
    data_files = [m["file_name"] for m in metadata.values()]

    features = Features({
        "wav": "",
        "path": Value(dtype='string', id=None),
        "sample_rate": Value(dtype='int', id=None),
        "label": Value(dtype='string', id=None),
        "speaker": Value(dtype='string', id=None)
    })
    return Dataset.from_dict("audiofolder", data_files=data_files)
